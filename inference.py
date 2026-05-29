"""
inference.py — Sampling from a trained 235M checkpoint.

Supports:
  - Greedy decoding (deterministic argmax at each step)
  - Top-p / top-k / temperature sampling

Usage examples:
    # Interactive greedy decoding
    python inference.py --checkpoint checkpoints/step_0152000.pt

    # Temperature sampling, top-p=0.9
    python inference.py \\
        --checkpoint checkpoints/step_0152000.pt \\
        --mode sampling \\
        --temperature 0.8 \\
        --top_p 0.9 \\
        --top_k 50 \\
        --max_new_tokens 256

    # Pass prompt directly (non-interactive)
    python inference.py \\
        --checkpoint checkpoints/step_0152000.pt \\
        --prompt "def quicksort(arr):" \\
        --mode sampling --temperature 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature. temperature=1.0 is identity."""
    if temperature == 1.0:
        return logits
    return logits / max(temperature, 1e-8)


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k largest."""
    if top_k <= 0:
        return logits
    top_k   = min(top_k, logits.size(-1))
    values  = torch.topk(logits, top_k).values
    cutoff  = values[..., -1, None]          # (batch, 1)
    return logits.masked_fill(logits < cutoff, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability mass exceeds top_p, zero out the rest."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs  = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Shift right so the token *at* the boundary is included
    cumprobs  = torch.cat([torch.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]], dim=-1)
    remove    = cumprobs >= top_p

    # Scatter back to original order
    remove    = remove.scatter(dim=-1, index=sorted_idx, src=remove)
    return logits.masked_fill(remove, float("-inf"))


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids:      torch.Tensor,           # (1, T) on the correct device
    max_new_tokens: int        = 200,
    eos_token_id:   Optional[int] = 2,
    max_seq_len:    int        = 2048,
) -> torch.Tensor:
    """
    Greedy (argmax) autoregressive generation.

    Returns the full token sequence (prompt + completion) as a 1-D LongTensor.
    Stops early if eos_token_id is generated.
    """
    ids = input_ids.clone()

    for _ in range(max_new_tokens):
        ctx    = ids[:, -max_seq_len:]          # honour context window
        logits = model(ctx)                      # (1, T, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, 1)
        ids    = torch.cat([ids, next_id], dim=1)

        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return ids[0]


@torch.inference_mode()
def sample_generate(
    model,
    input_ids:      torch.Tensor,
    max_new_tokens: int   = 200,
    temperature:    float = 1.0,
    top_k:          int   = 0,
    top_p:          float = 1.0,
    eos_token_id:   Optional[int] = 2,
    max_seq_len:    int   = 2048,
) -> torch.Tensor:
    """
    Stochastic generation with temperature, top-k, and top-p (nucleus) sampling.

    Sampling order:
        logits → temperature scale → top-k filter → top-p filter → softmax → multinomial

    Returns the full token sequence (prompt + completion) as a 1-D LongTensor.
    Stops early if eos_token_id is generated.
    """
    ids = input_ids.clone()

    for _ in range(max_new_tokens):
        ctx    = ids[:, -max_seq_len:]
        logits = model(ctx)                     # (1, T, V)
        logits = logits[:, -1, :]               # (1, V) — next-token logits only

        # Apply sampling filters
        logits = _apply_temperature(logits, temperature)
        logits = _apply_top_k(logits, top_k)
        logits = _apply_top_p(logits, top_p)

        probs   = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        ids     = torch.cat([ids, next_id], dim=1)

        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return ids[0]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load the full model from a training checkpoint.
    The checkpoint must have been saved by train.py::save_checkpoint().
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import get_config
    from model.transformer import Transformer

    cfg   = get_config()
    model = Transformer.from_config(cfg.model)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle checkpoints saved from a compiled model (keys may have _orig_mod prefix)
    raw_state = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
    model.load_state_dict(raw_state)

    model = model.to(device).eval()
    print(f"Loaded checkpoint from step {state.get('step', '?')} "
          f"(loss={state.get('loss', float('nan')):.4f})")
    return model, cfg


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    return sp


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive_loop(
    model,
    sp,
    cfg,
    device:      torch.device,
    mode:        str   = "greedy",
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
    max_new_tokens: int = 200,
):
    """Simple REPL: type a prompt, get a completion. Ctrl-C or empty input to quit."""
    print(f"\n{'=' * 60}")
    print(f"  235M Inference  |  mode={mode}", end="")
    if mode == "sampling":
        print(f"  temp={temperature}  top_k={top_k}  top_p={top_p}", end="")
    print(f"\n{'=' * 60}")
    print("  Enter a prompt (empty line to quit).\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not prompt:
            print("Bye.")
            break

        ids = sp.Encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        if mode == "greedy":
            out = greedy_generate(
                model, inp,
                max_new_tokens = max_new_tokens,
                eos_token_id   = cfg.data.eos_token_id,
                max_seq_len    = cfg.model.max_seq_len,
            )
        else:
            out = sample_generate(
                model, inp,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = top_k,
                top_p          = top_p,
                eos_token_id   = cfg.data.eos_token_id,
                max_seq_len    = cfg.model.max_seq_len,
            )

        completion = sp.Decode(out.tolist())
        print(f"\n{completion}\n")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Batch / one-shot generation (for use as a library or in scripts)
# ---------------------------------------------------------------------------

def generate(
    prompt:         str,
    checkpoint:     str,
    mode:           str   = "greedy",
    temperature:    float = 1.0,
    top_k:          int   = 0,
    top_p:          float = 1.0,
    max_new_tokens: int   = 200,
    device_str:     str   = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    One-shot generation. Convenient for scripting / notebooks.

    Args:
        prompt:         Input text.
        checkpoint:     Path to a .pt checkpoint saved by train.py.
        mode:           'greedy' or 'sampling'.
        temperature:    Sampling temperature (ignored in greedy mode).
        top_k:          Top-k filter (0 = disabled).
        top_p:          Nucleus probability threshold (1.0 = disabled).
        max_new_tokens: Maximum tokens to generate.
        device_str:     'cuda', 'cpu', etc.

    Returns:
        Full generated string (prompt + completion decoded by SentencePiece).
    """
    device      = torch.device(device_str)
    model, cfg  = load_model_from_checkpoint(checkpoint, device)
    sp          = load_tokenizer(cfg.data.tokenizer_path)

    ids = sp.Encode(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.autocast(device_type=device.type, dtype=cfg.train.dtype,
                        enabled=(device.type == "cuda")):
        if mode == "greedy":
            out = greedy_generate(
                model, inp,
                max_new_tokens = max_new_tokens,
                eos_token_id   = cfg.data.eos_token_id,
                max_seq_len    = cfg.model.max_seq_len,
            )
        else:
            out = sample_generate(
                model, inp,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = top_k,
                top_p          = top_p,
                eos_token_id   = cfg.data.eos_token_id,
                max_seq_len    = cfg.model.max_seq_len,
            )

    return sp.Decode(out.tolist())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Sample from a 235M checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt checkpoint (e.g. checkpoints/step_0152000.pt)")
    p.add_argument("--prompt", default=None,
                   help="Prompt text. If omitted, enters interactive REPL.")
    p.add_argument("--mode", choices=["greedy", "sampling"], default="greedy",
                   help="Decoding strategy.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (sampling mode only).")
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k filter (0 = disabled). Sampling mode only.")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus (top-p) threshold (1.0 = disabled). Sampling mode only.")
    p.add_argument("--max_new_tokens", type=int, default=200,
                   help="Maximum number of new tokens to generate.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to run inference on.")
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device(args.device)

    model, cfg = load_model_from_checkpoint(args.checkpoint, device)
    sp         = load_tokenizer(cfg.data.tokenizer_path)

    if args.prompt:
        # Single-shot mode
        ids = sp.Encode(args.prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=cfg.train.dtype,
                            enabled=(device.type == "cuda")):
            if args.mode == "greedy":
                out = greedy_generate(
                    model, inp,
                    max_new_tokens = args.max_new_tokens,
                    eos_token_id   = cfg.data.eos_token_id,
                    max_seq_len    = cfg.model.max_seq_len,
                )
            else:
                out = sample_generate(
                    model, inp,
                    max_new_tokens = args.max_new_tokens,
                    temperature    = args.temperature,
                    top_k          = args.top_k,
                    top_p          = args.top_p,
                    eos_token_id   = cfg.data.eos_token_id,
                    max_seq_len    = cfg.model.max_seq_len,
                )
        print(sp.Decode(out.tolist()))
    else:
        # Interactive REPL
        interactive_loop(
            model, sp, cfg, device,
            mode           = args.mode,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            max_new_tokens = args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

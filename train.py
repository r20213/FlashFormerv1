"""
train.py — Modal training entrypoint for the 235M pretraining run.

Usage:
    # One-time: save secrets to Modal
    python train.py secrets

    # Launch training on Modal H100
    python train.py

Architecture:
    - One H100 SXM on Modal (modal.gpu.H100())
    - Dual optimiser: Muon (2D block weights) + fused AdamW (embedding/norm/head)
    - Cosine LR schedule with linear warmup, shared across both optimisers
    - torch.compile(mode="reduce-overhead") for ~30% throughput gain
    - BF16 autocast, gradient clipping, gradient accumulation
    - WandB logging (optional), checkpoint save/resume
    - Eval: validation loss + greedy completions on fixed prompts
"""

import os
import sys
import math
import time
import shutil
from pathlib import Path
import torch
import modal
from modal.mount import Mount
# ---------------------------------------------------------------------------
# Modal image — everything the training job needs
# ---------------------------------------------------------------------------
# Exact matching wheel URL for flash-attn 2.6.3, torch 2.4, python 3.11
# Fully verified wheel URL mapping to your environment
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("packaging", "torch==2.4.0", "torchvision")
    # Stage 2: Install Flash Attention using the pre-compiled wheel
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        # # Flash Attention (pre-built wheel for CUDA 12.1 / torch 2.4)
        # "flash-attn==2.6.3",
        # Data pipeline
        "datasets==2.20.0",
        "sentencepiece==0.2.0",
        "huggingface-hub==0.24.0",
        # Quality filters
        "detoxify==0.5.2",
        "fasttext-wheel==0.9.2",
        # Logging
        "wandb==0.17.4",
        # Misc
        "numpy",
        "tqdm",
    )
    # Replaces the deprecated mounts configuration
    .add_local_dir("model", remote_path="/root/model")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("config.py", remote_path="/root/config.py")
    .add_local_file("tokenizer/spm.model", remote_path="/root/tokenizer/spm.model")
)

app = modal.App("pretrain-235m", image=image)

# Persistent volume: checkpoints survive container restarts
volume = modal.Volume.from_name("pretrain-235m-checkpoints", create_if_missing=True)
VOLUME_MOUNT = "/checkpoints"

# ---------------------------------------------------------------------------
# Secret management helpers — run locally with: python train.py secrets
# ---------------------------------------------------------------------------

def save_secrets():
    """
    Interactively save HF_TOKEN and WANDB_API_KEY to Modal secret store.
    Run once before training: python train.py secrets
    """
    import subprocess

    print("=" * 60)
    print("  Saving secrets to Modal")
    print("=" * 60)

    hf_token    = input("Enter HuggingFace token (HF_TOKEN): ").strip()
    wandb_key   = input("Enter WandB API key (WANDB_API_KEY): ").strip()

    # Build the modal secret create command
    cmd = [
        "modal", "secret", "create", "pretrain-secrets",
        f"HF_TOKEN={hf_token}",
        f"WANDB_API_KEY={wandb_key}",
        "--force",  # overwrite if already exists
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\n✓ Secrets saved to Modal secret store as 'pretrain-secrets'")
        print("  You can now run training with: python train.py")
    else:
        print(f"\n✗ Error saving secrets:\n{result.stderr}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Muon optimiser (self-contained, no external dependency)
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G, steps: int = 5):
    """Newton-Schulz iteration to compute G / ||G||_F (approximate)."""
    import torch
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Normalise
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class MuonOptimizer:
    """
    Muon: Momentum + Orthogonalised Update (Newton-Schulz) for 2D weights.
    Implements the algorithm from "Muon is Scalable for LLM Training" (2024).
    """

    def __init__(
        self,
        params,
        lr:           float = 0.02,
        momentum:     float = 0.95,
        ns_steps:     int   = 5,
        weight_decay: float = 0.1,
    ):
        import torch
        self.params       = list(params)
        self.lr           = lr
        self.momentum     = momentum
        self.ns_steps     = ns_steps
        self.weight_decay = weight_decay

        # Momentum buffers (FP32)
        self.momentum_bufs = [
            torch.zeros_like(p, dtype=torch.float32)
            for p in self.params
        ]

    @torch.no_grad()
    def step(self, scale: float = 0.2):
        import torch
        for p, buf in zip(self.params, self.momentum_bufs):
            if p.grad is None:
                continue
            g = p.grad.float()

            # Nesterov momentum update
            buf.mul_(self.momentum).add_(g)
            g_nesterov = g.add(buf, alpha=self.momentum)

            # Orthogonalise via Newton-Schulz (returns unit-scale update)
            g_orth = _zeropower_via_newtonschulz5(g_nesterov, steps=self.ns_steps)

            # Scale update to match RMS of a typical AdamW update
            rms = g_orth.norm() / math.sqrt(g_orth.numel())
            if rms > 1e-8:
                g_orth = g_orth * (scale / rms)

            # Weight decay + update
            if self.weight_decay > 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)
            p.data.add_(g_orth.to(p.dtype), alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def state_dict(self):
        return {
            "lr":           self.lr,
            "momentum":     self.momentum,
            "ns_steps":     self.ns_steps,
            "weight_decay": self.weight_decay,
            "bufs":         [b.cpu() for b in self.momentum_bufs],
        }

    def load_state_dict(self, state: dict):
        import torch
        self.lr           = state["lr"]
        self.momentum     = state["momentum"]
        self.ns_steps     = state["ns_steps"]
        self.weight_decay = state["weight_decay"]
        for buf, saved in zip(self.momentum_bufs, state["bufs"]):
            buf.copy_(saved)


# ---------------------------------------------------------------------------
# LR schedule (cosine with linear warmup, shared between both optimisers)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int,
           peak_lr: float, decay_factor: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (decay_factor + (1.0 - decay_factor) * cosine)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / f"step_{step:07d}.pt"


def save_checkpoint(
    step:       int,
    model,
    muon:       MuonOptimizer,
    adamw,
    cfg,
    loss:       float,
    ckpt_dir:   Path,
    keep_last_n: int = 3,
):
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(ckpt_dir, step)
    torch.save({
        "step":        step,
        "model":       model.state_dict(),
        "muon":        muon.state_dict(),
        "adamw":       adamw.state_dict(),
        "loss":        loss,
    }, path)
    print(f"  [ckpt] Saved {path}")

    # Prune old checkpoints
    all_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in all_ckpts[:-keep_last_n]:
        old.unlink()


def load_checkpoint(path: str, model, muon: MuonOptimizer, adamw) -> int:
    import torch
    state = torch.load(path, map_location="cuda")
    model.load_state_dict(state["model"])
    muon.load_state_dict(state["muon"])
    adamw.load_state_dict(state["adamw"])
    step = state["step"]
    print(f"  [ckpt] Resumed from step {step}  (loss={state['loss']:.4f})")
    return step


# ---------------------------------------------------------------------------
# Evaluation: validation loss + prompt completions
# ---------------------------------------------------------------------------

def run_eval(model, val_loader, cfg, step: int, device, use_wandb: bool):
    import torch
    import torch.nn.functional as F
    from model.loss import ntp_loss

    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for ids in val_loader:
            ids = ids.to(device)
            with torch.autocast(device_type="cuda", dtype=cfg.train.dtype):
                logits = model(ids)
                loss   = ntp_loss(logits, ids, pad_id=3)
            total_loss += loss.item()
            n_batches  += 1

    val_loss = total_loss / max(1, n_batches)
    print(f"\n  [eval] step={step}  val_loss={val_loss:.4f}  val_ppl={math.exp(val_loss):.2f}")

    # Greedy completions for qualitative check
    print("\n  [completions] ──────────────────────────────────────────")
    _sample_prompts(model, cfg, device)
    print("  ────────────────────────────────────────────────────────\n")

    if use_wandb:
        import wandb
        wandb.log({"val_loss": val_loss, "val_ppl": math.exp(val_loss)}, step=step)

    model.train()
    return val_loss


def _sample_prompts(model, cfg, device):
    """Greedy completions on the fixed eval prompts from TrainConfig."""
    import torch
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(cfg.data.tokenizer_path)

    model.eval()
    for prompt in cfg.train.eval_prompts:
        ids = sp.Encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(cfg.train.max_new_tokens):
                # Trim to model's context window
                inp_ctx = inp[:, -cfg.model.max_seq_len:]
                with torch.autocast(device_type="cuda", dtype=cfg.train.dtype):
                    logits = model(inp_ctx)          # (1, T, V)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy
                inp = torch.cat([inp, next_id], dim=1)

        completion = sp.Decode(inp[0].tolist())
        print(f"\n  Prompt : {prompt!r}")
        print(f"  Output : {completion!r}")
    model.train()


# ---------------------------------------------------------------------------
# Core training function (runs inside Modal)
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100:1",
    timeout=4 * 3600,  # 4-hour wall clock limit
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("pretrain-secrets")],
)
def train():
    import torch
    import wandb
    import sys
    sys.path.insert(0, "/root")

    from config import get_config
    from model.transformer import Transformer, classify_params
    from data.dataset import make_dataloader
    from model.loss import ntp_loss

    # ── Setup ────────────────────────────────────────────────────────────────
    cfg = get_config()
    device = torch.device("cuda")
    torch.manual_seed(cfg.train.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # HuggingFace token for gated datasets
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # ── WandB ────────────────────────────────────────────────────────────────
    use_wandb = bool(cfg.train.wandb_project)
    if use_wandb:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb.login(key=wandb_key)
        wandb.init(
            project=cfg.train.wandb_project,
            config={
                "d_model":        cfg.model.d_model,
                "n_layers":       cfg.model.n_layers,
                "n_heads_q":      cfg.model.n_heads_q,
                "n_heads_kv":     cfg.model.n_heads_kv,
                "d_ff":           cfg.model.d_ff,
                "target_tokens":  cfg.train.target_tokens,
                "micro_batch":    cfg.train.micro_batch_size,
                "grad_accum":     cfg.train.grad_accum_steps,
                "muon_lr":        cfg.optim.muon_lr,
                "adamw_lr":       cfg.optim.adamw_lr,
                "warmup_steps":   cfg.optim.warmup_steps,
            },
        )

    print(cfg.summary())

    # ── Model ────────────────────────────────────────────────────────────────
    model = Transformer.from_config(cfg.model).to(device)
    print(f"\n  Parameters: {model.num_parameters()/1e6:.1f}M")

    if cfg.train.compile_model:
        print("  Compiling model (this takes ~60-90s on first step)…")
        model = torch.compile(model, mode="reduce-overhead")

    # ── Optimisers ───────────────────────────────────────────────────────────
    muon_params, adamw_params = classify_params(
        model._orig_mod if hasattr(model, "_orig_mod") else model
    )
    muon  = MuonOptimizer(
        muon_params,
        lr           = cfg.optim.muon_lr,
        momentum     = cfg.optim.muon_momentum,
        ns_steps     = cfg.optim.muon_ns_steps,
        weight_decay = cfg.optim.muon_weight_decay,
    )
    adamw = torch.optim.AdamW(
        adamw_params,
        lr           = cfg.optim.adamw_lr,
        betas        = cfg.optim.adamw_betas,
        eps          = cfg.optim.adamw_eps,
        weight_decay = cfg.optim.adamw_weight_decay,
        fused        = cfg.optim.adamw_fused,
    )

    total_steps = cfg.total_steps
    print(f"  Total optimiser steps: {total_steps:,}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = make_dataloader(cfg, split="train")
    val_loader   = make_dataloader(cfg, split="validation")

    # ── Resume ───────────────────────────────────────────────────────────────
    ckpt_dir = Path(VOLUME_MOUNT) / cfg.train.checkpoint_dir
    start_step = 0
    resume_path = cfg.train.resume_from
    if resume_path is None:
        # Auto-resume from latest checkpoint in volume
        existing = sorted(ckpt_dir.glob("step_*.pt"))
        if existing:
            resume_path = str(existing[-1])

    if resume_path and Path(resume_path).exists():
        start_step = load_checkpoint(
            resume_path,
            model._orig_mod if hasattr(model, "_orig_mod") else model,
            muon, adamw,
        )

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    train_iter  = iter(train_loader)
    step        = start_step
    accum_loss  = 0.0
    t0          = time.perf_counter()

    print(f"\n  Starting training from step {step}\n")

    while step < total_steps:
        # ── LR schedule ──────────────────────────────────────────────────────
        lr = get_lr(
            step,
            warmup_steps  = cfg.optim.warmup_steps,
            total_steps   = total_steps,
            peak_lr       = cfg.optim.muon_lr,
            decay_factor  = cfg.optim.lr_decay_factor,
        )
        lr_adamw = get_lr(
            step,
            warmup_steps  = cfg.optim.warmup_steps,
            total_steps   = total_steps,
            peak_lr       = cfg.optim.adamw_lr,
            decay_factor  = cfg.optim.lr_decay_factor,
        )
        muon.lr = lr
        for pg in adamw.param_groups:
            pg["lr"] = lr_adamw

        # ── Gradient accumulation ─────────────────────────────────────────────
        muon.zero_grad()
        adamw.zero_grad()

        for micro_step in range(cfg.train.grad_accum_steps):
            try:
                ids = next(train_iter).to(device)
            except StopIteration:
                train_iter = iter(train_loader)
                ids = next(train_iter).to(device)

            with torch.autocast(device_type="cuda", dtype=cfg.train.dtype):
                logits = model(ids)
                loss   = ntp_loss(logits, ids, pad_id=3)
                loss   = loss / cfg.train.grad_accum_steps  # scale for accum

            loss.backward()
            accum_loss += loss.item()

        # ── Gradient clip ─────────────────────────────────────────────────────
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), cfg.optim.grad_clip
        )

        # ── Optimiser step ────────────────────────────────────────────────────
        muon.step(scale=cfg.optim.muon_update_scale)
        adamw.step()

        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % cfg.train.log_every == 0:
            dt    = time.perf_counter() - t0
            tok_s = (cfg.train.log_every * cfg.tokens_per_step) / dt
            print(
                f"  step={step:>7,}  loss={accum_loss:.4f}  "
                f"grad_norm={grad_norm:.3f}  "
                f"lr_muon={lr:.2e}  lr_adamw={lr_adamw:.2e}  "
                f"tok/s={tok_s:,.0f}"
            )
            if use_wandb:
                wandb.log({
                    "train_loss": accum_loss,
                    "grad_norm":  grad_norm,
                    "lr_muon":    lr,
                    "lr_adamw":   lr_adamw,
                    "tok_per_s":  tok_s,
                    "tokens_seen": step * cfg.tokens_per_step,
                }, step=step)
            accum_loss = 0.0
            t0 = time.perf_counter()

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % cfg.train.eval_every == 0:
            run_eval(model, val_loader, cfg, step, device, use_wandb)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % cfg.train.checkpoint_every == 0:
            save_checkpoint(
                step,
                raw_model,
                muon, adamw, cfg,
                loss=accum_loss,
                ckpt_dir=ckpt_dir,
                keep_last_n=cfg.train.keep_last_n,
            )
            volume.commit()  # flush to Modal persistent volume

    # ── Final checkpoint ──────────────────────────────────────────────────────
    print("\n  Training complete. Saving final checkpoint…")
    save_checkpoint(
        step,
        raw_model,
        muon, adamw, cfg,
        loss=0.0,
        ckpt_dir=ckpt_dir,
        keep_last_n=cfg.train.keep_last_n,
    )
    volume.commit()

    if use_wandb:
        wandb.finish()

    print("  Done.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "secrets":
        save_secrets()
    else:
        # Trigger Modal remote run
        with app.run():
            train.remote()

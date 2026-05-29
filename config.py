"""
config.py — Single source of truth for every hyperparameter and path.

Design principles:
  - All numbers live here. No magic constants anywhere else in the codebase.
  - Dataclasses give us free __repr__, type hints, and IDE completion.
  - ModelConfig and TrainConfig are intentionally separate — model arch is
    independent of training decisions (LR, batch size, etc.).
  - get_config() is the single entry point. Override fields via CLI args
    or environment variables if you need sweeps; don't fork the dataclass.

Parameter count estimate (verify with model.num_parameters()):
  Embedding      :  32768 × 1024              =  33.6M
  16 × Attention :  16 × 4 × 1024²            =  67.1M   (Q,K,V,O projections)
  16 × FFN       :  16 × 3 × 1024 × 2730      =  134.3M  (gate, up, down)
  16 × RMSNorm   :  16 × 2 × 1024             =  0.03M   (pre-attn, pre-ffn)
  Final RMSNorm  :  1024                       =  0.001M
  Output head    :  tied to embedding          =  0M
  ─────────────────────────────────────────────────────
  Total                                        ≈  235M params
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # --- Dimensions ---
    vocab_size: int = 32_768          # SentencePiece BPE vocabulary
    d_model: int = 1024               # Residual stream / embedding dimension
    n_layers: int = 16                # Number of transformer blocks

    # --- Attention (GQA) ---
    # Rule: n_heads_q must be divisible by n_heads_kv.
    # KV compression ratio here is 4:1 (16Q / 4KV).
    # head_dim = d_model // n_heads_q = 64 — well within FA3/FA4 support.
    n_heads_q: int = 16              # Query heads
    n_heads_kv: int = 4              # Key/Value heads (GQA)
    head_dim: int = 64               # d_model // n_heads_q; set explicitly for clarity

    # --- FFN (SwiGLU) ---
    # SwiGLU uses 3 matrices (gate, up, down). To keep parameter count
    # iso to a standard 4× FFN, the hidden dim is ⌊8/3 × d_model⌋ rounded
    # to the nearest multiple of 256 for CUDA memory alignment.
    d_ff: int = 2816                  # 8/3 × 1024 = 2730.6 → rounded to 2816

    # --- Positional encoding (RoPE) ---
    max_seq_len: int = 2048          # Training context window
    rope_theta: float = 10_000.0     # Base frequency for RoPE

    # --- Regularisation ---
    # No dropout during pretraining — data volume is our regulariser.
    # No bias terms anywhere — standard modern practice (GPT-NeoX, Llama, etc.)
    dropout: float = 0.0
    use_bias: bool = False

    # --- Tied embeddings ---
    # Output projection (d_model → vocab) shares weights with the input
    # embedding matrix. Saves 33.6M params and often improves convergence.
    tie_embeddings: bool = True

    # --- Numerical ---
    dtype: torch.dtype = torch.bfloat16   # BF16 throughout; no FP8 this run

    def __post_init__(self):
        assert self.d_model % self.n_heads_q == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads_q ({self.n_heads_q})"
        )
        assert self.n_heads_q % self.n_heads_kv == 0, (
            f"n_heads_q ({self.n_heads_q}) must be divisible by n_heads_kv ({self.n_heads_kv})"
        )
        assert self.head_dim == self.d_model // self.n_heads_q, (
            f"head_dim ({self.head_dim}) must equal d_model // n_heads_q "
            f"({self.d_model // self.n_heads_q})"
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # --- Source datasets (HuggingFace Hub) ---
    # Weights are normalised internally by the interleaver — they don't need
    # to sum to 1. All starcoderdata splits use text_column="content".
    #
    # Language selection rationale (3-tier strategy):
    #
    #   Tier A — Core competence (6 langs, ~300M+ tokens each):
    #     Python, JavaScript, Java, C++, C, SQL
    #     Model will be genuinely useful for these. Non-negotiable.
    #
    #   Tier B — Reasoning signal (7 langs, 46–120M tokens each):
    #     Go, Rust, Shell, TypeScript, CUDA, Lean, Haskell
    #     Each teaches a structurally distinct reasoning pattern.
    #     ~50M tokens minimum is enough for signal transfer at 235M params.
    #     Lean: highest value/token — formal logic proof chains transfer to math.
    #     CUDA: parallelism + memory hierarchy, cognitively distinct from all others.
    #     Go: concurrency (goroutines/channels) = distributed systems reasoning.
    #     TypeScript stays (not dropped for JS): type annotations are distinct signal.
    #
    #   Dropped (noise at this scale):
    #     HTML   — noisy minified markup, poor signal/token
    #     VHDL   — tiny corpus, near-zero semantic overlap with target reasoning
    #     Erlang — OTP patterns need 200M+ tokens to emerge; looks like noise here
    #     Perl   — write-only regex soup; Python covers scripting better
    #     R      — tiny corpus, domain-specific; Python covers data science
    #     Dockerfile — ~5M tokens total; too sparse for any competence
    #
    # Token exposure estimates use StarCoder paper proportions over 6.6B code tokens.
    # "Utilisation" = estimated tokens sampled ÷ estimated corpus size.
    sources: list = field(default_factory=lambda: [

        # ── Tier 0: Natural language prose (33.7%) ──────────────────────────
        # 378M rows — never exhausted. Synthetic edu rewrites of FineWeb at L3
        # quality. Multi-style register: expository, conversational, instructional.
        {
            "path": "openbmb/Ultra-FineWeb-L3",
            "name": "Ultra-FineWeb-L3-en-Multi-Style-Synthetic",
            "split": "train",
            "text_column": "content",
            "weight": 0.337,
        },

        # ── Tier A: Core competence languages ───────────────────────────────

        # Python — highest weight; docstrings = implicit NL supervision.
        # ~18B tok available. ~594M sampled → ~0.03× (tiny fraction, no repeat).
        {
            "path": "bigcode/starcoderdata",
            "name": "python",
            "split": "train",
            "text_column": "content",
            "weight": 0.090,
        },
        # JavaScript — largest raw token pool (~25B). Covers frontend + Node.
        {
            "path": "bigcode/starcoderdata",
            "name": "javascript",
            "split": "train",
            "text_column": "content",
            "weight": 0.065,
        },
        # Java — verbose typing teaches structured/OOP reasoning.
        {
            "path": "bigcode/starcoderdata",
            "name": "java",
            "split": "train",
            "text_column": "content",
            "weight": 0.045,
        },
        # C++ — low-level + OOP patterns; distinct from C in abstraction style.
        {
            "path": "bigcode/starcoderdata",
            "name": "cpp",
            "split": "train",
            "text_column": "content",
            "weight": 0.038,
        },
        # C — systems/memory reasoning; manual resource management patterns.
        {
            "path": "bigcode/starcoderdata",
            "name": "c",
            "split": "train",
            "text_column": "content",
            "weight": 0.032,
        },
        # SQL — structured query logic; distinct register from imperative code.
        # Small corpus (~4B tok). ~99M sampled → ~2.5× repetition, acceptable.
        {
            "path": "bigcode/starcoderdata",
            "name": "sql",
            "split": "train",
            "text_column": "content",
            "weight": 0.015,
        },

        # ── Tier B: Reasoning signal languages ──────────────────────────────

        # Go — concurrency primitives (goroutines, channels, select).
        # Best available proxy for distributed systems reasoning patterns.
        {
            "path": "bigcode/starcoderdata",
            "name": "go",
            "split": "train",
            "text_column": "content",
            "weight": 0.018,        # ~119M tokens sampled
        },
        # Rust — ownership/lifetimes force explicit resource reasoning chains.
        # Borrow checker logic = closest thing to formal verification in systems code.
        {
            "path": "bigcode/starcoderdata",
            "name": "rust",
            "split": "train",
            "text_column": "content",
            "weight": 0.016,        # ~106M tokens sampled
        },
        # Shell — tool composition, pipelines, process substitution.
        # Teaches "glue logic" reasoning distinct from all other languages.
        {
            "path": "bigcode/starcoderdata",
            "name": "shell",
            "split": "train",
            "text_column": "content",
            "weight": 0.012,        # ~79M tokens sampled
        },
        # TypeScript — type annotations are structurally distinct from plain JS.
        # Interface definitions, generics, conditional types = type-level reasoning.
        {
            "path": "bigcode/starcoderdata",
            "name": "typescript",
            "split": "train",
            "text_column": "content",
            "weight": 0.010,        # ~66M tokens sampled
        },
        # CUDA — parallel thread indexing, shared memory, barrier synchronisation.
        # Cognitively distinct from everything else; GPU memory hierarchy forces
        # explicit spatial reasoning about data layout. Tiny corpus but dense.
        {
            "path": "bigcode/starcoderdata",
            "name": "cuda",
            "split": "train",
            "text_column": "content",
            "weight": 0.008,        # ~53M tokens sampled
        },
        # Lean — formal theorem proofs. Highest reasoning signal per token of
        # anything on GitHub. Strict logical deduction chains transfer directly
        # to mathematical reasoning. Tiny corpus; every token counts.
        {
            "path": "bigcode/starcoderdata",
            "name": "lean",
            "split": "train",
            "text_column": "content",
            "weight": 0.007,        # ~46M tokens sampled; small corpus ~1× utilisation
        },
        # Haskell — pure functional patterns, lazy evaluation, type inference chains.
        # Monads/functors teach compositional reasoning unlike any imperative lang.
        {
            "path": "bigcode/starcoderdata",
            "name": "haskell",
            "split": "train",
            "text_column": "content",
            "weight": 0.007,        # ~46M tokens sampled
        },

        # ── Tier 0: Math pretraining corpus (22.0%) ─────────────────────────
        # 6.3M rows, ~9.6B tokens of raw mathematical web content.
        # LaTeX equations, proofs, worked solutions, math forum posts.
        # NOT SFT — teaches mathematical prose as a writing style, not as an
        # answer format. load_dataset("HuggingFaceTB/finemath","infiwebmath-4plus")
        {
            "path": "HuggingFaceTB/finemath",
            "name": "infiwebmath-4plus",
            "split": "train",
            "text_column": "text",
            "weight": 0.220,
        },

        # ── Tier 0: Instruction / reasoning SFT (1.2%) ──────────────────────
        # 289k rows × ~800 tok/row ≈ 231M tokens total.
        # Weight calibrated for exactly-once traversal: 231M / 20B = 0.01155.
        # Effective sampling ≈ 1.3% of steps. DO NOT increase this weight —
        # doing so reintroduces repetition without adding signal.
        {
            "path": "openbmb/UltraInteract_sft",
            "name": None,
            "split": "train",
            "text_column": None,    # Special: formatted from instruction+response columns
            "weight": 0.012,
        },
    ])

    # --- Tokeniser ---
    tokenizer_path: str = "tokenizer/spm.model"   # Output of tokenizer/train.py
    eos_token_id: int = 2            # Standard SP EOS; verified in train_tokenizer.py
    bos_token_id: int = 1            # Standard SP BOS

    # --- Sequence packing ---
    # Documents are concatenated with EOS separators and sliced into
    # fixed-length chunks. No padding. Position IDs reset per document.
    seq_len: int = 2048              # Must match ModelConfig.max_seq_len
    pack_sequences: bool = True

    # --- Filtering (inline, stateless per row) ---
    # MinHash dedup is run as a separate offline pass (see data/dedup.py).
    # These inline filters are applied during streaming.
    min_doc_tokens: int = 64         # Drop documents shorter than this
    max_doc_tokens: int = 8192       # Drop documents longer than this (pre-pack)
    toxicity_threshold: float = 0.5  # Detoxify score above which row is dropped
    lang_threshold: float = 0.85     # fasttext language confidence (English)

    # --- Tokeniser training sample ---
    tokenizer_sample_size: int = 15_000_000   # tokens; ~15M for good BPE coverage
    tokenizer_vocab_size: int = 32_768        # Must match ModelConfig.vocab_size

    # --- Workers ---
    num_proc: int = 4                # DataLoader workers for prefetching


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

@dataclass
class OptimConfig:
    # --- Muon (applies to all 2D Linear weight matrices except embedding + head) ---
    muon_lr: float = 0.02            # Higher than AdamW; NS orthogonalisation normalises magnitude
    muon_momentum: float = 0.95      # Nesterov momentum coefficient
    muon_ns_steps: int = 5           # Newton-Schulz iteration steps; 5 is sweet spot (FLOP < 1%)
    muon_update_scale: float = 0.2   # RMS-match to AdamW update norm (Moonshot calibration)
    muon_weight_decay: float = 0.1   # Weight decay for Muon params (from "Muon is Scalable" paper)

    # --- AdamW (embedding, RMSNorm scales, output head) ---
    adamw_lr: float = 3e-4
    adamw_betas: tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.1
    adamw_fused: bool = True          # Use PyTorch fused AdamW kernel on CUDA

    # --- Shared schedule ---
    # One cosine scheduler governs LR for both optimisers simultaneously.
    # The Muon LR and AdamW LR decay at the same relative rate.
    warmup_steps: int = 2_000        # Linear ramp from 0 → peak LR
    lr_decay_factor: float = 0.1     # Final LR = peak × decay_factor (i.e. 10× decay)
    # Total steps is computed at runtime from token budget ÷ tokens_per_step

    # --- Gradient clipping ---
    grad_clip: float = 1.0           # Max gradient norm; standard for transformer pretraining


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # --- Compute budget ---
    # Target: 20B tokens. With Muon ~1.35× efficiency ≈ 27B AdamW-equivalent.
    # At ~1.5M tok/sec on H100 → ~3.7 hrs. Fits within the 3-hr window with
    # some buffer for compilation, checkpointing, and eval passes.
    target_tokens: int = 20_000_000_000   # 20B tokens total

    # --- Batch configuration ---
    # Effective batch = micro_batch × grad_accum × seq_len
    # = 16 × 4 × 2048 = 131,072 tokens/step ≈ 128k token batch
    # This is in the standard range for 200M-scale models.
    micro_batch_size: int = 16       # Sequences per GPU step
    grad_accum_steps: int = 4        # Gradient accumulation before optimizer.step()
    # tokens_per_step = micro_batch_size × grad_accum_steps × seq_len
    # = 16 × 4 × 2048 = 131,072

    # --- Precision ---
    dtype: torch.dtype = torch.bfloat16
    # torch.autocast is applied in the training loop around forward + loss.
    # Muon and AdamW states are kept in FP32 internally.

    # --- Compilation ---
    compile_model: bool = True       # torch.compile(mode="reduce-overhead")
    # First step will be slow (~60-90s for compilation). Subsequent steps ~normal.

    # --- Logging ---
    log_every: int = 10              # Log loss + grad norm every N steps
    eval_every: int = 500            # Run validation loss every N steps
    sample_every: int = 2_000        # Greedy completion of fixed prompts every N steps
    wandb_project: Optional[str] = "pretrain-235m"  # Set to None to disable wandb

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 2_000    # Save checkpoint every N steps
    keep_last_n: int = 3             # Only keep the last N checkpoints (disk budget)
    resume_from: Optional[str] = None   # Path to checkpoint to resume from

    # --- Evaluation ---
    val_tokens: int = 50_000         # Tokens in the held-out validation shard
    # Fixed prompts for qualitative coherence check (one per domain)
    eval_prompts: list = field(default_factory=lambda: [
        "def binary_search(arr, target):",                          # code
        "The French Revolution began in 1789 when",                 # natural language
        "Solve step by step: If 3x + 7 = 22, then x =",            # math
        "Instruction: Summarise the following in one sentence.\n",  # instruction
        "Question: What is the capital of Australia? Answer:",       # factual
    ])
    max_new_tokens: int = 128        # Max tokens to generate per prompt during eval

    # --- Seed ---
    seed: int = 42


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    root: Path = Path(".")
    tokenizer_dir: Path = Path("tokenizer")
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")

    def make_dirs(self):
        for d in [self.tokenizer_dir, self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # --- Derived properties (computed, not set by user) ---

    @property
    def tokens_per_step(self) -> int:
        """Effective token throughput per optimizer step."""
        return (
            self.train.micro_batch_size
            * self.train.grad_accum_steps
            * self.data.seq_len
        )

    @property
    def total_steps(self) -> int:
        """Total optimizer steps for the full token budget."""
        return self.train.target_tokens // self.tokens_per_step

    @property
    def gqa_groups(self) -> int:
        """Number of GQA groups = n_heads_q // n_heads_kv."""
        return self.model.n_heads_q // self.model.n_heads_kv

    def validate(self):
        """Cross-field consistency checks. Call before training starts."""
        assert self.data.seq_len == self.model.max_seq_len, (
            f"data.seq_len ({self.data.seq_len}) must match "
            f"model.max_seq_len ({self.model.max_seq_len})"
        )
        assert self.data.tokenizer_vocab_size == self.model.vocab_size, (
            f"data.tokenizer_vocab_size ({self.data.tokenizer_vocab_size}) must match "
            f"model.vocab_size ({self.model.vocab_size})"
        )
        assert self.model.dtype == self.train.dtype, (
            "model.dtype and train.dtype must match — both should be bfloat16"
        )
        # Warn if token budget is very ambitious for a 3-hour window
        est_hours = self.train.target_tokens / (1_500_000 * 3600)
        if est_hours > 3.5:
            import warnings
            warnings.warn(
                f"Token budget of {self.train.target_tokens/1e9:.1f}B tokens "
                f"may take ~{est_hours:.1f} hours at 1.5M tok/sec. "
                f"Consider reducing target_tokens or increasing micro_batch_size."
            )

    def summary(self) -> str:
        """Human-readable training plan summary."""
        lines = [
            "=" * 60,
            "  PRETRAINING CONFIG SUMMARY",
            "=" * 60,
            f"  Model        : {self.model.n_layers}L × {self.model.d_model}d  "
                              f"| {self.model.n_heads_q}Q/{self.model.n_heads_kv}KV heads  "
                              f"| d_ff={self.model.d_ff}",
            f"  Vocab        : {self.model.vocab_size:,}  | tied embeddings={self.model.tie_embeddings}",
            f"  Sequence     : {self.data.seq_len} tokens  | packed={self.data.pack_sequences}",
            f"  Batch        : {self.train.micro_batch_size} micro × {self.train.grad_accum_steps} accum "
                              f"= {self.tokens_per_step:,} tok/step",
            f"  Token budget : {self.train.target_tokens/1e9:.0f}B tokens",
            f"  Total steps  : {self.total_steps:,}",
            f"  Warmup steps : {self.optim.warmup_steps:,}",
            f"  Muon LR      : {self.optim.muon_lr}  |  AdamW LR : {self.optim.adamw_lr}",
            f"  Precision    : {self.train.dtype}",
            f"  torch.compile: {self.train.compile_model}",
            f"  Est. 1.5M tok/sec → {self.train.target_tokens/1.5e6/3600:.1f} hours",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_config() -> Config:
    """
    Returns the default Config. To customise for a sweep or experiment,
    modify fields after calling this function, then call cfg.validate().

    Example:
        cfg = get_config()
        cfg.train.target_tokens = 10_000_000_000  # 10B for a quick test run
        cfg.train.wandb_project = None             # disable wandb
        cfg.validate()
    """
    cfg = Config()
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    print(cfg.summary())
    print()

    # Verify model config internal consistency
    print("Model config checks:")
    print(f"  head_dim             = {cfg.model.head_dim}  (d_model // n_heads_q)")
    print(f"  GQA groups           = {cfg.gqa_groups}  (n_heads_q // n_heads_kv)")
    print(f"  KV compression ratio = {cfg.model.n_heads_q // cfg.model.n_heads_kv}×")
    print(f"  tokens_per_step      = {cfg.tokens_per_step:,}")
    print(f"  total_steps          = {cfg.total_steps:,}")
    print()

    # Rough parameter count
    vocab, d, n, dff = (
        cfg.model.vocab_size,
        cfg.model.d_model,
        cfg.model.n_layers,
        cfg.model.d_ff,
    )
    p_embed = vocab * d
    p_attn  = n * 4 * d * d     # Q, K, V, O  (K and V are smaller with GQA but close)
    p_ffn   = n * 3 * d * dff   # gate, up, down
    p_norm  = n * 2 * d         # pre-attn + pre-ffn RMSNorm scales
    p_total = p_embed + p_attn + p_ffn + p_norm
    print("Estimated parameter counts:")
    print(f"  Embedding  : {p_embed/1e6:.1f}M")
    print(f"  Attention  : {p_attn/1e6:.1f}M  (approx; ignores GQA KV reduction)")
    print(f"  FFN        : {p_ffn/1e6:.1f}M")
    print(f"  Norms      : {p_norm/1e6:.2f}M")
    print(f"  Total      : {p_total/1e6:.1f}M  (output head tied → no extra params)")
    print()
    print("All checks passed. Config is valid.")

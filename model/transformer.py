"""
model/transformer.py — RMSNorm, TransformerBlock, and full Transformer

Architecture (per block, × n_layers):
  x → RMSNorm → GQA → Add(residual) → RMSNorm → SwiGLUFFN → Add(residual)

Design notes:
  - Pre-norm architecture throughout (norm before each sub-layer).
  - No bias on any layer (use_bias=False).
  - Tied embeddings: output projection weight = embedding weight transposed.
    Saves 33.6M parameters and generally improves convergence at this scale.
  - RoPE is instantiated once and shared across all transformer blocks
    (the cos/sin tables are identical for every layer).
  - model.parameters() correctly excludes tied output head weight from the
    parameter set (it's the same tensor as the embedding).
  - Muon optimizer targets all 2D weight matrices EXCEPT embed/norm/head.
    The classify_params() helper in this module returns param groups.
  - dtype is kept at BF16 by the training loop's autocast; this module
    does not call .to(dtype) internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope      import RotaryEmbedding
from model.attention import GroupedQueryAttention
from model.ffn       import SwiGLUFFN


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation — no mean subtraction, no bias.

    out = x / RMS(x) * scale
    where RMS(x) = sqrt(mean(x²) + eps)

    Args:
        dim: Feature dimension to normalise over (last dimension of input).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))  # learned scale (no bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in FP32 for numerical stability, then cast back
        x_fp32 = x.float()
        rms     = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(x.dtype) * self.scale


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer block: pre-norm attention + pre-norm FFN.

    Residual connections added AFTER each sub-layer (post-add, pre-norm style).

    Args:
        d_model, n_heads_q, n_heads_kv, head_dim, d_ff: architecture dims.
        rope:     Shared RotaryEmbedding instance.
        use_bias: Propagated to attention and FFN.
        dropout:  Attention dropout (0.0 for pretraining).
    """

    def __init__(
        self,
        d_model:    int,
        n_heads_q:  int,
        n_heads_kv: int,
        head_dim:   int,
        d_ff:       int,
        rope:       RotaryEmbedding,
        use_bias:   bool  = False,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn      = GroupedQueryAttention(
            d_model    = d_model,
            n_heads_q  = n_heads_q,
            n_heads_kv = n_heads_kv,
            head_dim   = head_dim,
            rope       = rope,
            use_bias   = use_bias,
            dropout    = dropout,
        )
        self.norm_ffn  = RMSNorm(d_model)
        self.ffn       = SwiGLUFFN(d_model=d_model, d_ff=d_ff, use_bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # Attention sub-layer: pre-norm + residual
        x = x + self.attn(self.norm_attn(x))
        # FFN sub-layer: pre-norm + residual
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ---------------------------------------------------------------------------
# Full Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Decoder-only transformer for causal language modelling.

    Forward pass:
        token_ids → embedding → N × TransformerBlock → RMSNorm → logits

    Tied weights: lm_head.weight is the same tensor as embedding.weight.
    Output logits are raw (no softmax); the training loss applies log-softmax
    internally via cross-entropy.

    Args (all sourced from ModelConfig):
        vocab_size, d_model, n_layers,
        n_heads_q, n_heads_kv, head_dim,
        d_ff, max_seq_len, rope_theta,
        use_bias, dropout, tie_embeddings.
    """

    def __init__(
        self,
        vocab_size:     int,
        d_model:        int,
        n_layers:       int,
        n_heads_q:      int,
        n_heads_kv:     int,
        head_dim:       int,
        d_ff:           int,
        max_seq_len:    int   = 2048,
        rope_theta:     float = 10_000.0,
        use_bias:       bool  = False,
        dropout:        float = 0.0,
        tie_embeddings: bool  = True,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_layers  = n_layers

        # ── Token embedding ──────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ── Shared RoPE (one instance, shared across all blocks) ─────────
        self.rope = RotaryEmbedding(
            head_dim    = head_dim,
            max_seq_len = max_seq_len,
            theta       = rope_theta,
        )

        # ── Transformer blocks ───────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model    = d_model,
                n_heads_q  = n_heads_q,
                n_heads_kv = n_heads_kv,
                head_dim   = head_dim,
                d_ff       = d_ff,
                rope       = self.rope,
                use_bias   = use_bias,
                dropout    = dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Final layer norm ─────────────────────────────────────────────
        self.norm_final = RMSNorm(d_model)

        # ── Output projection (no bias, no softmax) ──────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding matrix with the output projection.
        # After tying, lm_head.weight IS embedding.weight — same object in memory.
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # ── Initialisation ───────────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        GPT-2 / Llama-style initialisation:
          - Embedding: normal(0, 0.02)
          - All Linear weights: normal(0, 0.02)
          - Output projections (attn o_proj, ffn down_proj): scaled by
            1/sqrt(2 * n_layers) to control residual stream variance growth.
          - RMSNorm scales: already ones (nn.Parameter default).
        """
        std = 0.02
        scaled_std = std / (2 * self.n_layers) ** 0.5

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Scale down residual-path output projections
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=scaled_std)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) long tensor of token IDs

        Returns:
            logits: (B, T, vocab_size) — raw (no softmax)
        """
        x = self.embedding(input_ids)   # (B, T, d_model)

        for block in self.blocks:
            x = block(x)                # (B, T, d_model)

        x = self.norm_final(x)          # (B, T, d_model)
        logits = self.lm_head(x)        # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count parameters, correctly handling tied weights (no double-count)."""
        seen   = set()
        total  = 0
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    @classmethod
    def from_config(cls, cfg) -> "Transformer":
        """Construct a Transformer from a ModelConfig dataclass."""
        return cls(
            vocab_size     = cfg.vocab_size,
            d_model        = cfg.d_model,
            n_layers       = cfg.n_layers,
            n_heads_q      = cfg.n_heads_q,
            n_heads_kv     = cfg.n_heads_kv,
            head_dim       = cfg.head_dim,
            d_ff           = cfg.d_ff,
            max_seq_len    = cfg.max_seq_len,
            rope_theta     = cfg.rope_theta,
            use_bias       = cfg.use_bias,
            dropout        = cfg.dropout,
            tie_embeddings = cfg.tie_embeddings,
        )


# ---------------------------------------------------------------------------
# Optimizer parameter grouping
# ---------------------------------------------------------------------------

def classify_params(model: Transformer):
    """
    Split model parameters into two groups for the dual-optimizer setup:

      muon_params  — all 2-D weight matrices inside transformer blocks
                     (Q/K/V/O projections, gate/up/down FFN weights).
                     These are updated by Muon with Nesterov orthogonalisation.

      adamw_params — everything else: embedding, RMSNorm scales, lm_head weight
                     (which is tied to the embedding, so it's already in the
                     embedding set), and any 1-D tensors (norms, biases if used).
                     Updated by fused AdamW.

    Returns:
        muon_params:  list[nn.Parameter]
        adamw_params: list[nn.Parameter]

    Note: tied embedding/lm_head weight appears only ONCE (in adamw_params).
    """
    muon_params  = []
    adamw_params = []
    seen         = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        # Muon targets: 2D weights inside transformer blocks only
        is_block_weight = (
            param.ndim == 2
            and any(f"blocks.{i}." in name for i in range(model.n_layers))
            # Exclude norm scales (they're 1D anyway, but be explicit)
            and "norm" not in name
        )
        if is_block_weight:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params

"""
model/attention.py — Grouped-Query Attention with Flash Attention

Design notes:
  - Uses flash_attn_func directly (flash-attn v2/v3 compatible API).
  - GQA implemented by expanding KV heads to match Q heads ONLY for the
    matmuls — this is the standard repeat_kv trick that avoids materialising
    extra KV projections.
  - RoPE is applied to Q and K (in (B, T, H, D) layout) before passing to
    flash_attn_func.
  - flash_attn_func expects inputs in (B, T, H, D) — we do NOT reshape to
    (B, H, T, D). The function handles the head dimension internally.
  - No bias on any projection (use_bias=False in config).
  - causal=True always during training (autoregressive LM).
  - Softmax scale is applied by flash_attn internally (1/sqrt(head_dim)).
  - BF16 throughout; no mixed precision inside this module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) and Flash Attention.

    Args:
        d_model:     Residual stream dimension.
        n_heads_q:   Number of query heads.
        n_heads_kv:  Number of key/value heads. Must divide n_heads_q evenly.
        head_dim:    Dimension per head (= d_model // n_heads_q).
        rope:        Shared RotaryEmbedding instance (precomputed tables).
        use_bias:    Whether to add bias to projections (False for modern LLMs).
        dropout:     Attention dropout rate (0.0 during pretraining).
    """

    def __init__(
        self,
        d_model:    int,
        n_heads_q:  int,
        n_heads_kv: int,
        head_dim:   int,
        rope:       RotaryEmbedding,
        use_bias:   bool = False,
        dropout:    float = 0.0,
    ):
        super().__init__()
        assert n_heads_q % n_heads_kv == 0, (
            f"n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
        )
        assert d_model == n_heads_q * head_dim, (
            f"d_model ({d_model}) must equal n_heads_q × head_dim ({n_heads_q}×{head_dim})"
        )

        self.d_model    = d_model
        self.n_heads_q  = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.head_dim   = head_dim
        self.gqa_groups = n_heads_q // n_heads_kv  # repeat factor for KV
        self.dropout    = dropout
        self.rope       = rope

        # Projections
        # Q: d_model → n_heads_q × head_dim
        self.q_proj = nn.Linear(d_model, n_heads_q  * head_dim, bias=use_bias)
        # K, V: d_model → n_heads_kv × head_dim  (GQA: fewer KV heads)
        self.k_proj = nn.Linear(d_model, n_heads_kv * head_dim, bias=use_bias)
        self.v_proj = nn.Linear(d_model, n_heads_kv * head_dim, bias=use_bias)
        # Output: n_heads_q × head_dim → d_model
        self.o_proj = nn.Linear(n_heads_q * head_dim, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)  — pre-normed residual stream

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # ── 1. Linear projections ──────────────────────────────────────────
        q = self.q_proj(x)  # (B, T, n_heads_q  * head_dim)
        k = self.k_proj(x)  # (B, T, n_heads_kv * head_dim)
        v = self.v_proj(x)  # (B, T, n_heads_kv * head_dim)

        # ── 2. Reshape to (B, T, n_heads, head_dim) ───────────────────────
        q = q.view(B, T, self.n_heads_q,  self.head_dim)
        k = k.view(B, T, self.n_heads_kv, self.head_dim)
        v = v.view(B, T, self.n_heads_kv, self.head_dim)

        # ── 3. Apply RoPE to Q and K only ─────────────────────────────────
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)

        # ── 4. Expand KV heads to match Q heads (GQA → MHA view for FA) ───
        # flash_attn_func accepts mismatched head counts natively in FA2+.
        # We still expand for FA3/FA4 compatibility and clarity.
        # k, v: (B, T, n_heads_kv, head_dim) → (B, T, n_heads_q, head_dim)
        if self.gqa_groups > 1:
            k = k.repeat_interleave(self.gqa_groups, dim=2)
            v = v.repeat_interleave(self.gqa_groups, dim=2)

        # ── 5. Flash Attention ─────────────────────────────────────────────
        # flash_attn_func(q, k, v, ...) expects (B, T, H, D)
        # Returns (B, T, H, D); causal=True for autoregressive LM.
        attn_drop = self.dropout if self.training else 0.0
        out = flash_attn_func(
            q, k, v,
            dropout_p=attn_drop,
            causal=True,
        )  # (B, T, n_heads_q, head_dim)

        # ── 6. Merge heads and project ────────────────────────────────────
        out = out.reshape(B, T, self.n_heads_q * self.head_dim)  # (B, T, d_model)
        out = self.o_proj(out)  # (B, T, d_model)

        return out

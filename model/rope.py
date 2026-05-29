"""
model/rope.py — Rotary Position Embedding (RoPE)

Design notes:
  - Precomputes cos/sin tables once at construction for max_seq_len.
  - apply_rope() operates on (B, T, n_heads, head_dim) tensors — the layout
    that flash_attn_func expects before it reshapes internally.
  - Only Q and K are rotated; V is left untouched.
  - rotate_half splits the last dimension in half and applies the standard
    (-x2, x1) rotation, matching the Llama/GPT-NeoX convention.
  - Tables are registered as buffers (not parameters) — they move to the
    correct device with .to(device) / .cuda() but are excluded from grad.
  - BF16 safe: tables are cast to match the input dtype on the fly so no
    precision mismatch even when inputs are BF16.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Precomputed RoPE cos/sin tables for sequences up to `max_seq_len`.

    Usage:
        rope  = RotaryEmbedding(head_dim=64, max_seq_len=2048, theta=10_000.0)
        q_rot = rope(q, seq_len=T)   # (B, T, n_heads_q, head_dim)
        k_rot = rope(k, seq_len=T)   # (B, T, n_heads_kv, head_dim)
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.theta       = theta

        # Inverse frequencies: shape (head_dim // 2,)
        # inv_freq[i] = 1 / theta^(2i / head_dim)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute and cache cos/sin tables: shape (max_seq_len, head_dim)
        self._build_cache(max_seq_len)

    # ------------------------------------------------------------------
    # Cache construction
    # ------------------------------------------------------------------

    def _build_cache(self, seq_len: int) -> None:
        """Build or extend the cos/sin cache to cover `seq_len` positions."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # outer product → (seq_len, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # emb = [freqs, freqs] along last dim → (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply RoPE to `x`.

        Args:
            x:       (B, T, n_heads, head_dim) — Q or K tensor
            seq_len: T (used for slicing the precomputed tables)

        Returns:
            Rotated tensor with the same shape as `x`.
        """
        if seq_len > self.max_seq_len:
            # Lazily extend the cache if someone passes a longer sequence
            # (shouldn't happen during standard training but good to be safe).
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]  # (T, head_dim)
        sin = self.sin_cached[:seq_len]  # (T, head_dim)

        # Cast tables to match input dtype (e.g. bfloat16)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # Broadcast over batch and head dimensions:
        # x    : (B, T, n_heads, head_dim)
        # cos/sin need shape (1, T, 1, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim)

        return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension by 90° using the (-x2, x1) convention.

    Splits x into two halves along the last dimension and returns:
        [-x2, x1]  (concatenated back to the same shape)
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # first half
    x2 = x[..., half:]   # second half
    return torch.cat([-x2, x1], dim=-1)

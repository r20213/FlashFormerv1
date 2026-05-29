"""
model/ffn.py — SwiGLU Feed-Forward Network

Design notes:
  - Three separate projections: gate_proj, up_proj, down_proj.
    This is the Llama/PaLM-style decomposition; NOT the fused (gate+up) variant.
    Keeping them separate makes weight shapes cleaner and avoids the fused split.
  - Forward: out = down_proj( silu(gate_proj(x)) * up_proj(x) )
  - No bias anywhere (use_bias=False).
  - No activation after down_proj — the residual add happens in TransformerBlock.
  - d_ff = ⌊8/3 × d_model⌋, rounded to a multiple of 256 in config.py.
    For d_model=1024: 8/3×1024 ≈ 2730 → 2816 (nearest multiple of 256).
  - Parameter count: 3 × d_model × d_ff = 3 × 1024 × 2816 ≈ 8.65M per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    Args:
        d_model:  Input/output dimension (residual stream width).
        d_ff:     Hidden dimension. Typically ⌊8/3 × d_model⌋, rounded up.
        use_bias: Add bias to linear layers? (False for modern LLMs.)
    """

    def __init__(self, d_model: int, d_ff: int, use_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=use_bias)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=use_bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # gate branch: silu(W_gate · x)
        gate = F.silu(self.gate_proj(x))   # (B, T, d_ff)
        # up branch: W_up · x
        up   = self.up_proj(x)             # (B, T, d_ff)
        # element-wise gating then project down
        return self.down_proj(gate * up)   # (B, T, d_model)

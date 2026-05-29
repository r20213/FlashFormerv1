"""
model/loss.py — Next-Token Prediction loss

Design notes:
  - Standard cross-entropy over a causal shift: target[t] = input[t+1].
  - The shift is done in-place via slicing: logits[:, :-1] and ids[:, 1:].
    This matches the diagram notation: cross_entropy(logits[:,:-1], ids[:,1:]).
  - Padding tokens (token_id == pad_id) are masked out with ignore_index so
    they contribute zero gradient. EOS tokens between packed documents ARE
    included in the loss (they teach the model to predict sequence boundaries).
  - logits are passed as FP32 or BF16; F.cross_entropy upcasts internally
    to FP32 for the log-softmax, so there's no precision risk.
  - reduction="mean" averages over all unmasked token positions in the batch.
    This means the effective learning signal scales correctly with sequence
    packing density.
"""

import torch
import torch.nn.functional as F


def ntp_loss(
    logits:  torch.Tensor,
    ids:     torch.Tensor,
    pad_id:  int = 3,
) -> torch.Tensor:
    """
    Next-token prediction cross-entropy loss.

    Args:
        logits: (B, T, vocab_size) — raw output from the model (no softmax).
        ids:    (B, T)             — input token IDs (same tensor that was fed
                                    to the model as input_ids).
        pad_id: Token ID to ignore in the loss (default 3 = <pad> from SPM).

    Returns:
        Scalar loss (mean cross-entropy over all non-padding target positions).

    Shift logic:
        Input position t predicts position t+1.
        So we use logits[:, :-1, :] as predictions and ids[:, 1:] as targets.
        This discards the last logit and the first token as label, which is
        the standard causal LM formulation.
    """
    # Shift: predict next token at every position except the last
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
    shift_labels = ids[:, 1:].contiguous()          # (B, T-1)

    # Flatten to (B*(T-1), vocab_size) and (B*(T-1),) for F.cross_entropy
    B, T_minus_1, V = shift_logits.shape
    shift_logits = shift_logits.view(B * T_minus_1, V)
    shift_labels = shift_labels.view(B * T_minus_1)

    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        ignore_index=pad_id,
        reduction="mean",
    )
    return loss

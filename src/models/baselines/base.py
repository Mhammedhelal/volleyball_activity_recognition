"""
src/models/baselines/base.py
----------------------------
Shared utilities and the BaselineModel ABC.

Every baseline must declare:
    INPUT_TYPE: str   "crops"  → forward(crops: [N, T, C, H, W])
                      "frame"  → forward(frame: [T, C, H, W])

    HAS_PERSON_LOSS: bool   True  → trainer feeds person_labels and adds aux loss
                            False → no person supervision (B1, B2, B4, B5)

The trainer/evaluator reads these attributes to route the correct tensor
without any if/else on model class names.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Literal


class BaselineModel(nn.Module):
    """
    Abstract base for all baseline models.

    Subclasses set class-level flags; the engine reads them.
    """
    # ── contract flags ───────────────────────────────────────────────────────
    INPUT_TYPE:      Literal["crops", "frame"] = "crops"
    HAS_PERSON_LOSS: bool = False          # override to True in B3/B6/B7

    # ── helpers ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns group_logits: [num_classes]  (single sample, no batch dim)

        Optionally also returns person_logits: [N, person_classes]
        when HAS_PERSON_LOSS is True.
        """
        raise NotImplementedError


def pool_persons(
    x:    torch.Tensor,
    mode: Literal["max", "avg"] = "max",
) -> torch.Tensor:
    """
    Aggregate per-person feature vectors into one scene-level vector.

    Args:
        x    : (N, D)   — one feature vector per person
        mode : pooling strategy

    Returns:
        (D,)  — scene-level descriptor
    """
    if x.shape[0] == 0:
        raise ValueError("pool_persons: got 0 persons")
    if mode == "max":
        return x.max(dim=0).values      # (D,)
    return x.mean(dim=0)                # (D,)
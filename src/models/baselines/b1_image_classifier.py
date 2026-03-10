"""
src/models/baselines/b1_image_classifier.py
-------------------------------------------
B1 — Image Classification

Architecture:
    Full Frame [T, C, H, W]
        → pick center frame [C, H, W]
        → AlexNet fc7  [D]
        → Linear       [num_classes]

INPUT_TYPE = "frame"   (receives full-frame sequence, picks center)
HAS_PERSON_LOSS = False
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable

from src.models.baselines.base import BaselineModel
from src.models.cnn_backbones import build_alexnet_fc7


class B1_ImageClassifier(BaselineModel):
    INPUT_TYPE      = "frame"
    HAS_PERSON_LOSS = False

    def __init__(
        self,
        num_classes:  int,
        backbone_fn:  Callable = build_alexnet_fc7,
    ):
        super().__init__()
        self.backbone, feat_dim = backbone_fn(freeze=False)   # fine-tuned per paper
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame : [T, C, H, W]  full-frame temporal sequence

        Returns:
            group_logits : [num_classes]
        """
        # B1 is a single-frame model — use the center frame
        t_center = frame.shape[0] // 2
        img = frame[t_center]                   # [C, H, W]
        img = img.unsqueeze(0)                  # [1, C, H, W]  (batch dim for backbone)
        feats = self.backbone(img)              # [1, D]
        return self.classifier(feats).squeeze(0)  # [num_classes]
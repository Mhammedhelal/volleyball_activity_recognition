"""
src/models/baselines/b4_temporal_image.py
------------------------------------------
B4 — Temporal Model with Image Features  (≈ LRCN / Donahue et al.)

Architecture:
    Full Frames [T, C, H, W]
        → AlexNet fc7 per frame  [T, D]
        → LSTM over T            hidden: [H]
        → Linear                 [num_classes]

Note: On the Volleyball Dataset this performs *worse* than B1 because
camera motion corrupts the temporal signal at the full-frame level.

INPUT_TYPE = "frame"
HAS_PERSON_LOSS = False
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable

from src.models.baselines.base import BaselineModel
from src.models.cnn_backbones import build_alexnet_fc7


class B4_TemporalImageModel(BaselineModel):
    INPUT_TYPE      = "frame"
    HAS_PERSON_LOSS = False

    def __init__(
        self,
        num_classes:  int,
        backbone_fn:  Callable = build_alexnet_fc7,
        lstm_hidden:  int = 512,
        lstm_layers:  int = 1,
    ):
        super().__init__()
        self.backbone, feat_dim = backbone_fn(freeze=False)
        self.lstm       = nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame : [T, C, H, W]  full-frame temporal sequence

        Returns:
            group_logits : [num_classes]
        """
        T, C, H, W = frame.shape
        feats = self.backbone(frame)            # [T, D]  (T acts as batch here)
        feats = feats.unsqueeze(0)              # [1, T, D]
        _, (h_n, _) = self.lstm(feats)          # h_n: [layers, 1, H]
        return self.classifier(h_n[-1].squeeze(0))  # [num_classes]
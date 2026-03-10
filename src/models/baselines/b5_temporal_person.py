"""
src/models/baselines/b5_temporal_person.py
-------------------------------------------
B5 — Temporal Model with Person Features

Architecture:
    Person crops [N, T, C, H, W]
        → AlexNet fc7 per (person × frame)  [N, T, D]
        → pool over N at each timestep       [T, D]   ← persons collapsed BEFORE LSTM
        → LSTM over T                        hidden: [H]
        → Linear                             [num_classes]

The key limitation vs B7: person-level temporal dynamics are
collapsed *before* the LSTM, so the LSTM reasons about the
group sequence, not individual person sequences.

INPUT_TYPE = "crops"
HAS_PERSON_LOSS = False
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Literal

from src.models.baselines.base import BaselineModel, pool_persons
from src.models.cnn_backbones import build_alexnet_fc7


class B5_TemporalPersonModel(BaselineModel):
    INPUT_TYPE      = "crops"
    HAS_PERSON_LOSS = False

    def __init__(
        self,
        num_classes:  int,
        backbone_fn:  Callable = build_alexnet_fc7,
        pool:         Literal["max", "avg"] = "max",
        lstm_hidden:  int = 512,
        lstm_layers:  int = 1,
    ):
        super().__init__()
        self.backbone, feat_dim = backbone_fn(freeze=False)
        self.pool       = pool
        self.lstm       = nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops : [N, T, C, H, W]

        Returns:
            group_logits : [num_classes]
        """
        N, T, C, H, W = crops.shape

        # Extract features for every (person, frame) pair
        flat  = crops.view(N * T, C, H, W)
        feats = self.backbone(flat)             # [N*T, D]
        feats = feats.view(N, T, -1)            # [N, T, D]

        # Pool over persons at each timestep → one vector per frame
        # feats[:, t, :] is [N, D] → pool_persons → [D]
        pooled_seq = torch.stack(
            [pool_persons(feats[:, t, :], self.pool) for t in range(T)],
            dim=0,
        )                                       # [T, D]

        pooled_seq = pooled_seq.unsqueeze(0)    # [1, T, D]
        _, (h_n, _) = self.lstm(pooled_seq)     # h_n: [layers, 1, H]
        return self.classifier(h_n[-1].squeeze(0))  # [num_classes]
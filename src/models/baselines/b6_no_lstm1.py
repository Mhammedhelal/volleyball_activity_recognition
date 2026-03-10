"""
src/models/baselines/b6_no_lstm1.py
-------------------------------------
B6 — Two-stage Model without LSTM 1

Architecture:
    Person crops [N, T, C, H, W]
        → fine-tuned AlexNet fc7 per (person × frame)  [N, T, D]
        → pool over N at each timestep                  [T, D]   (no person LSTM)
        → LSTM2 over T                                  hidden: [H]
        → Linear                                        [num_classes]

Group-level temporal dynamics ARE modeled (LSTM2).
Person-level temporal dynamics are NOT modeled (no LSTM1).

INPUT_TYPE = "crops"
HAS_PERSON_LOSS = False
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Literal

from src.models.baselines.base import BaselineModel, pool_persons
from src.models.cnn_backbones import build_alexnet_fc7


class B6_NoPersonLSTM(BaselineModel):
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
        # CNN is fine-tuned (freeze=False) — matches paper's B6 description
        self.backbone, feat_dim = backbone_fn(freeze=False)
        self.pool       = pool
        self.lstm2      = nn.LSTM(feat_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops : [N, T, C, H, W]

        Returns:
            group_logits : [num_classes]
        """
        N, T, C, H, W = crops.shape

        flat  = crops.view(N * T, C, H, W)
        feats = self.backbone(flat)             # [N*T, D]
        feats = feats.view(N, T, -1)            # [N, T, D]

        # Frame descriptors Z_t — pool persons at each timestep
        z_seq = torch.stack(
            [pool_persons(feats[:, t, :], self.pool) for t in range(T)],
            dim=0,
        )                                       # [T, D]

        z_seq = z_seq.unsqueeze(0)              # [1, T, D]
        _, (h_n, _) = self.lstm2(z_seq)         # h_n: [layers, 1, H]
        return self.classifier(h_n[-1].squeeze(0))  # [num_classes]
"""
src/models/baselines/b7_no_lstm2.py
-------------------------------------
B7 — Two-stage Model without LSTM 2   (strongest baseline: 80.2% on Volleyball)

Architecture:
    Person crops [N, T, C, H, W]
        → fine-tuned AlexNet fc7  [N, T, D]      (shared CNN)
        → LSTM1 per person        [N, T, H_p]    (shared weights across persons)
        → concat CNN + LSTM1      [N, T, D+H_p]  (P_tk = x_tk ⊕ h_tk)
        → use last timestep T     [N, D+H_p]
        → pool over N             [D+H_p]
        → Linear                  [num_classes]

Person dynamics ARE captured (LSTM1).
Group dynamics across time are NOT further modeled (no LSTM2).

INPUT_TYPE = "crops"
HAS_PERSON_LOSS = False
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Literal

from src.models.baselines.base import BaselineModel, pool_persons
from src.models.cnn_backbones import build_alexnet_fc7


class B7_NoGroupLSTM(BaselineModel):
    INPUT_TYPE      = "crops"
    HAS_PERSON_LOSS = False

    def __init__(
        self,
        num_classes:  int,
        backbone_fn:  Callable = build_alexnet_fc7,
        pool:         Literal["max", "avg"] = "max",
        lstm1_hidden: int = 512,
        lstm1_layers: int = 1,
    ):
        super().__init__()
        self.backbone, feat_dim = backbone_fn(freeze=False)
        self.pool   = pool
        self.lstm1  = nn.LSTM(feat_dim, lstm1_hidden, lstm1_layers, batch_first=True)
        # P_tk = x_tk ⊕ h_tk → concatenated dim
        self.classifier = nn.Linear(feat_dim + lstm1_hidden, num_classes)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops : [N, T, C, H, W]

        Returns:
            group_logits : [num_classes]
        """
        N, T, C, H, W = crops.shape

        flat      = crops.view(N * T, C, H, W)
        cnn_feats = self.backbone(flat)                     # [N*T, D]
        cnn_feats = cnn_feats.view(N, T, -1)                # [N, T, D]

        # LSTM1: shared weights, each person gets its own temporal sequence
        lstm_out, _ = self.lstm1(cnn_feats)                 # [N, T, H_p]

        # P_tk = x_tk ⊕ h_tk — use the last timestep for classification
        p_T = torch.cat([cnn_feats[:, -1, :],
                         lstm_out[:, -1, :]], dim=-1)       # [N, D+H_p]

        pooled = pool_persons(p_T, self.pool)                # [D+H_p]
        return self.classifier(pooled.unsqueeze(0)).squeeze(0)  # [num_classes]
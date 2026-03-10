"""
src/models/baselines/b2_b3_person_classifier.py
------------------------------------------------
B2 — Person Classification
B3 — Fine-tuned Person Classification  (B2 with freeze=False)

Architecture (both):
    Person crops [N, T, C, H, W]
        → pick center frame  [N, C, H, W]
        → AlexNet fc7 per person  [N, D]
        → pool over N persons     [D]
        → Linear                  [num_classes]

B2: backbone frozen (generic ImageNet features)
B3: backbone unfrozen (fine-tuned on person action labels)

INPUT_TYPE = "crops"
HAS_PERSON_LOSS = False  (no person-level supervision in B2/B3 group classifier)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Literal

from src.models.baselines.base import BaselineModel, pool_persons
from src.models.cnn_backbones import build_alexnet_fc7


class B2_PersonClassifier(BaselineModel):
    INPUT_TYPE      = "crops"
    HAS_PERSON_LOSS = False

    def __init__(
        self,
        num_classes: int,
        backbone_fn: Callable = build_alexnet_fc7,
        pool:        Literal["max", "avg"] = "max",
        freeze:      bool = True,           # True → B2,  False → B3
    ):
        super().__init__()
        self.backbone, feat_dim = backbone_fn(freeze=freeze)
        self.pool       = pool
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops : [N, T, C, H, W]

        Returns:
            group_logits : [num_classes]
        """
        N, T, C, H, W = crops.shape
        t_center = T // 2

        center = crops[:, t_center, :, :, :]    # [N, C, H, W]
        feats  = self.backbone(center)           # [N, D]
        pooled = pool_persons(feats, self.pool)  # [D]
        return self.classifier(pooled.unsqueeze(0)).squeeze(0)  # [num_classes]


def B3_FineTunedPersonClassifier(
    num_classes: int,
    backbone_fn: Callable = build_alexnet_fc7,
    pool:        Literal["max", "avg"] = "max",
) -> B2_PersonClassifier:
    """
    B3 is B2 with freeze=False — backbone fine-tuned on person action labels.
    Returns a B2_PersonClassifier instance (same class, different init flag).
    """
    model = B2_PersonClassifier(
        num_classes=num_classes,
        backbone_fn=backbone_fn,
        pool=pool,
        freeze=False,
    )
    return model
"""
src/models/baselines/__init__.py
---------------------------------
Baseline registry.

Usage
-----
    from src.models.baselines import BASELINES

    model = BASELINES["B4"](num_classes=8)
    # model.INPUT_TYPE  → "frame" or "crops"
    # model.HAS_PERSON_LOSS → False

    # Override backbone or pool strategy:
    from src.models.cnn_backbones import build_resnet50
    model = BASELINES["B2"](num_classes=8, backbone_fn=build_resnet50, pool="avg")

Available keys
--------------
    "B1"  ImageClassifier          INPUT_TYPE=frame,  HAS_PERSON_LOSS=False
    "B2"  PersonClassifier         INPUT_TYPE=crops,  HAS_PERSON_LOSS=False
    "B3"  FineTunedPersonClassifier INPUT_TYPE=crops, HAS_PERSON_LOSS=False
    "B4"  TemporalImageModel       INPUT_TYPE=frame,  HAS_PERSON_LOSS=False
    "B5"  TemporalPersonModel      INPUT_TYPE=crops,  HAS_PERSON_LOSS=False
    "B6"  NoPersonLSTM             INPUT_TYPE=crops,  HAS_PERSON_LOSS=False
    "B7"  NoGroupLSTM              INPUT_TYPE=crops,  HAS_PERSON_LOSS=False
"""

from src.models.baselines.b1_image_classifier    import B1_ImageClassifier
from src.models.baselines.b2_b3_person_classifier import (
    B2_PersonClassifier,
    B3_FineTunedPersonClassifier,
)
from src.models.baselines.b4_temporal_image      import B4_TemporalImageModel
from src.models.baselines.b5_temporal_person     import B5_TemporalPersonModel
from src.models.baselines.b6_no_lstm1            import B6_NoPersonLSTM
from src.models.baselines.b7_no_lstm2            import B7_NoGroupLSTM

BASELINES: dict = {
    "B1": B1_ImageClassifier,
    "B2": B2_PersonClassifier,
    "B3": B3_FineTunedPersonClassifier,
    "B4": B4_TemporalImageModel,
    "B5": B5_TemporalPersonModel,
    "B6": B6_NoPersonLSTM,
    "B7": B7_NoGroupLSTM,
}

__all__ = [
    "BASELINES",
    "B1_ImageClassifier",
    "B2_PersonClassifier",
    "B3_FineTunedPersonClassifier",
    "B4_TemporalImageModel",
    "B5_TemporalPersonModel",
    "B6_NoPersonLSTM",
    "B7_NoGroupLSTM",
]
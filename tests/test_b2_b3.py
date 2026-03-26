# tests/test_b2_b3.py  (B2: frozen backbone) (B3: fine-tuned — freeze=False)

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines.b2_b3_person_classifier import B2_PersonClassifier, B3_FineTunedPersonClassifier
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


N, T, C, H, W = 12, 9, 3, 224, 224
NUM_CLASSES = 8


@pytest.fixture(scope="module")
def sample_crops():
    return torch.randn(N, T, C, H, W)


class TestB2B3Shapes:
    @pytest.mark.parametrize("model_cls", [B2_PersonClassifier, B3_FineTunedPersonClassifier])
    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_output_shape(self, model_cls, backbone_fn, sample_crops):
        model = model_cls(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        logits = model(sample_crops)
        assert logits.shape == (NUM_CLASSES,)

    def test_variable_N(self, sample_crops):
        model = B2_PersonClassifier(num_classes=NUM_CLASSES)
        for n in (6, 10, 12):
            x = torch.randn(n, T, C, H, W)
            logits = model(x)
            assert logits.shape == (NUM_CLASSES,)


class TestB2B3Gradients:
    def test_gradient_to_classifier(self):
        model = B2_PersonClassifier(num_classes=NUM_CLASSES)
        x = torch.randn(N, T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        for name, param in model.classifier.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0


# For B3 specifically (fine-tuned)
class TestB3Gradients:
    def test_backbone_receives_gradients(self):
        model = B3_FineTunedPersonClassifier(num_classes=NUM_CLASSES)
        x = torch.randn(N, T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Backbone should have gradients because freeze=False
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                      for p in model.backbone.parameters())
        assert has_grad
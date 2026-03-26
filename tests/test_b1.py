import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.baselines.b1_image_classifier import B1_ImageClassifier
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

T, C, H, W           = 9, 3, 224, 224
CNN_DIM              = 4096
NUM_CLASSES          = 8


# ─────────────────────────────────────────────
# Output shape tests
# ─────────────────────────────────────────────




# ─────────────────────────────────────────────
# CNN frozen tests
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestB1Gradients:

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_lstm(self, feature_extractor):
        """Loss backward must produce gradients in LSTM1 weights."""
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, feature_extractor=feature_extractor)
        x     = torch.randn(T, C, H, W)
        logits, _ = model(x)
        loss  = logits.sum()
        loss.backward()

        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_no_gradient_in_cnn(self, feature_extractor):
        """CNN backbone must not accumulate gradients."""
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, feature_extractor=feature_extractor)
        x     = torch.randn(T, C, H, W)
        logits, _ = model(x)
        logits.sum().backward()

        for name, param in model.cnn.named_parameters():
            assert param.grad is None, (
                f"Unexpected gradient in frozen CNN param '{name}'"
            )


# ─────────────────────────────────────────────
# Device Consistency tests
# ─────────────────────────────────────────────
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestB1Eval:
    def test_forward_backward(self, device):
        x = torch.randn(T, C, H, W, device=device)
        model = B1_ImageClassifier(num_classes=NUM_CLASSES).to(device)
        logits = model(x)
        assert logits.device == device

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad.device == device


# ─────────────────────────────────────────────
# eval mode determinism tests
# ─────────────────────────────────────────────
class TestB1Eval:
    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_eval_mode_determinism(self, feature_extractor):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, feature_extractor=feature_extractor)
        x     = torch.randn(T, C, H, W)
        model.eval()
        logits_1= model(x)
        logits_2 = model(x)

        assert torch.allclose(logits_1, logits_2, atol=1e-6), "eval mode is not deteministic."
        
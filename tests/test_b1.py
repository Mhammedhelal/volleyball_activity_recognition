"""
tests/test_b1.py
----------------
Tests for B1 — Image Classifier (single center frame + AlexNet)
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines.b1_image_classifier import B1_ImageClassifier
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


T, C, H, W = 9, 3, 224, 224
NUM_CLASSES = 8


@pytest.fixture(scope="module")
def sample_frame():
    return torch.randn(T, C, H, W)


class TestB1Shapes:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_output_shape(self, backbone_fn, sample_frame):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        logits = model(sample_frame)
        assert logits.shape == (NUM_CLASSES,)
        assert logits.dim() == 1
        del model
        torch.cuda.empty_cache()

    def test_variable_T(self, sample_frame):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES)
        for t in (5, 9, 11):
            x = torch.randn(t, C, H, W)
            logits = model(x)
            assert logits.shape == (NUM_CLASSES,)
        del model
        torch.cuda.empty_cache()


class TestB1Gradients:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_classifier(self, backbone_fn):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        x = torch.randn(T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Classifier must have gradients
        for name, param in model.classifier.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.abs().sum() > 0
        del model
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_backbone(self, backbone_fn):
        """Since B1 uses freeze=False, backbone should receive gradients."""
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        x = torch.randn(T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        has_grad = False
        for param in model.backbone.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to backbone (expected since freeze=False)"
        del model
        torch.cuda.empty_cache()


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestB1Device:

    def test_forward_backward(self, device):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES).to(device)
        x = torch.randn(T, C, H, W, device=device)
        logits = model(x)
        assert logits.device == torch.device(device)

        loss = logits.sum()
        loss.backward()
        del model
        torch.cuda.empty_cache()


class TestB1Eval:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_eval_mode_determinism(self, backbone_fn):
        model = B1_ImageClassifier(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        x = torch.randn(T, C, H, W)
        model.eval()
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Eval mode is not deterministic"
        del model
        torch.cuda.empty_cache()
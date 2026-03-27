# tests/test_b6.py
"""Tests for B6 — No Person LSTM (pool persons + LSTM over time)"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines.b6_no_lstm1 import B6_NoPersonLSTM
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


N, T, C, H, W = 4, 5, 3, 64, 64
NUM_CLASSES = 8


@pytest.fixture
def sample_crops():
    return torch.randn(N, T, C, H, W)


class TestB6Shapes:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_output_shape(self, backbone_fn, pool, sample_crops):
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES, backbone_fn=backbone_fn, pool=pool)
        logits = model(sample_crops)
        assert logits.shape == (NUM_CLASSES,)
        del model
        torch.cuda.empty_cache()

    def test_variable_N(self, sample_crops):
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES)
        for n in (6, 10, 12):
            x = torch.randn(n, T, C, H, W)
            logits = model(x)
            assert logits.shape == (NUM_CLASSES,)
        del model
        torch.cuda.empty_cache()


class TestB6Gradients:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_lstm2_and_classifier(self, backbone_fn):
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        x = torch.randn(N, T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # LSTM2
        for name, param in model.lstm2.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

        # Classifier
        for name, param in model.classifier.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        del model
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_backbone(self, backbone_fn):
        """Since B6 uses freeze=False, backbone should receive gradients."""
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES, backbone_fn=backbone_fn)
        x = torch.randn(N, T, C, H, W)
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
class TestB6Device:

    def test_forward_backward(self, device):
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES).to(device)
        x = torch.randn(N, T, C, H, W, device=device)
        logits = model(x)
        assert logits.device == torch.device(device)

        loss = logits.sum()
        loss.backward()
        del model
        torch.cuda.empty_cache()


class TestB6Eval:

    @pytest.mark.parametrize("backbone_fn", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_eval_mode_determinism(self, backbone_fn, pool):
        model = B6_NoPersonLSTM(num_classes=NUM_CLASSES, backbone_fn=backbone_fn, pool=pool)
        x = torch.randn(N, T, C, H, W)
        model.eval()
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Eval mode is not deterministic"
        del model
        torch.cuda.empty_cache()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
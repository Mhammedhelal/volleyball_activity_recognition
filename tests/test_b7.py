# tests/test_b7.py
"""Tests for B7 — No Group LSTM (person LSTM + concat + pool)"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines.b7_no_lstm2 import B7_NoGroupLSTM
from src.models.cnn_backbones import build_alexnet_fc7


N, T, C, H, W = 12, 9, 3, 224, 224
NUM_CLASSES = 8


@pytest.fixture(scope="module")
def sample_crops():
    return torch.randn(N, T, C, H, W)


class TestB7Shapes:

    def test_output_shape(self, sample_crops):
        model = B7_NoGroupLSTM(num_classes=NUM_CLASSES)
        logits = model(sample_crops)
        assert logits.shape == (NUM_CLASSES,)

    def test_variable_N(self, sample_crops):
        model = B7_NoGroupLSTM(num_classes=NUM_CLASSES)
        for n in (8, 12):
            x = torch.randn(n, T, C, H, W)
            logits = model(x)
            assert logits.shape == (NUM_CLASSES,)


class TestB7Gradients:

    def test_gradient_flows_to_lstm1_and_classifier(self):
        model = B7_NoGroupLSTM(num_classes=NUM_CLASSES)
        x = torch.randn(N, T, C, H, W)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # LSTM1
        for name, param in model.lstm1.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0

        # Classifier
        for name, param in model.classifier.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0
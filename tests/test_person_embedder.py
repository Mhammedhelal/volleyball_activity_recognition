"""
tests/test_person_embedder.py
-----------------------------
Unit tests for PersonEmbedder.
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.person_embedder import PersonEmbedder


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

N, T, C, H, W       = 12, 9, 3, 224, 224
CNN_DIM              = 4096
LSTM_HIDDEN          = 512
PERSON_CLASSES       = 9
EMBED_DIM            = CNN_DIM + LSTM_HIDDEN   # 4608


# ─────────────────────────────────────────────
# Output shape tests
# ─────────────────────────────────────────────

class TestPersonEmbedderShapes:

    def setup_method(self):
        """Create embedder and sample input for each test."""
        self.embedder = PersonEmbedder(
            cnn_output_size=CNN_DIM,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        self.sample_input = torch.randn(N, T, C, H, W)

    def test_person_logits_shape(self):
        person_logits, _ = self.embedder(self.sample_input)
        assert person_logits.shape == (N, PERSON_CLASSES), (
            f"Expected [{N}, {PERSON_CLASSES}], got {person_logits.shape}"
        )

    def test_P_shape(self):
        _, P = self.embedder(self.sample_input)
        assert P.shape == (N, T, EMBED_DIM), (
            f"Expected [{N}, {T}, {EMBED_DIM}], got {P.shape}"
        )

    def test_P_temporal_dim_preserved(self):
        """P must retain all T timesteps — not collapsed to last frame."""
        _, P = self.embedder(self.sample_input)
        assert P.shape[1] == T, (
            f"Temporal dimension collapsed: expected {T}, got {P.shape[1]}"
        )

    def test_variable_N(self):
        """Output shapes must scale with N."""
        for n in (1, 6, 12):
            x             = torch.randn(n, T, C, H, W)
            logits, P     = self.embedder(x)
            assert logits.shape == (n, PERSON_CLASSES)
            assert P.shape      == (n, T, EMBED_DIM)

    def test_variable_T(self):
        """Output shapes must scale with T."""
        for t in (5, 9):
            x         = torch.randn(N, t, C, H, W)
            _, P      = self.embedder(x)
            assert P.shape[1] == t, f"Expected T={t}, got {P.shape[1]}"


# ─────────────────────────────────────────────
# CNN frozen tests
# ─────────────────────────────────────────────

class TestPersonEmbedderCNN:

    def setup_method(self):
        """Create embedder for each test."""
        self.embedder = PersonEmbedder(
            cnn_output_size=CNN_DIM,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )

    def test_cnn_frozen(self):
        """AlexNet backbone must be frozen — no gradients."""
        for name, param in self.embedder.cnn.named_parameters():
            assert not param.requires_grad, (
                f"CNN param '{name}' has requires_grad=True (should be frozen)"
            )

    def test_lstm_trainable(self):
        """LSTM1 weights must be trainable."""
        for name, param in self.embedder.lstm.named_parameters():
            assert param.requires_grad, (
                f"LSTM param '{name}' has requires_grad=False"
            )

    def test_person_fc_trainable(self):
        for name, param in self.embedder.person_fc.named_parameters():
            assert param.requires_grad, (
                f"person_fc param '{name}' has requires_grad=False"
            )


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestPersonEmbedderGradients:

    def test_gradient_flows_to_lstm(self):
        """Loss backward must produce gradients in LSTM1 weights."""
        model = PersonEmbedder(CNN_DIM, LSTM_HIDDEN, PERSON_CLASSES)
        x     = torch.randn(4, T, C, H, W)
        logits, _ = model(x)
        loss  = logits.sum()
        loss.backward()

        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for LSTM param '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero gradient for LSTM param '{name}'"

    def test_no_gradient_in_cnn(self):
        """CNN backbone must not accumulate gradients."""
        model = PersonEmbedder(CNN_DIM, LSTM_HIDDEN, PERSON_CLASSES)
        x     = torch.randn(4, T, C, H, W)
        logits, _ = model(x)
        logits.sum().backward()

        for name, param in model.cnn.named_parameters():
            assert param.grad is None, (
                f"Unexpected gradient in frozen CNN param '{name}'"
            )

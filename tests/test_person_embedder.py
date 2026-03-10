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

from src.models.person_embedder import PersonEmbedder, build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


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
        """Create sample input for each test."""
        self.sample_input = torch.randn(N, T, C, H, W)

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_person_logits_shape(self, feature_extractor):
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        person_logits, _ = embedder(self.sample_input)
        assert person_logits.shape == (N, PERSON_CLASSES), (
            f"Expected [{N}, {PERSON_CLASSES}], got {person_logits.shape}"
        )

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_P_shape(self, feature_extractor):
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        _, P = embedder(self.sample_input)
        embed_dim = embedder.cnn_dim + LSTM_HIDDEN
        assert P.shape == (N, T, embed_dim), (
            f"Expected [{N}, {T}, {embed_dim}], got {P.shape}"
        )

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_P_temporal_dim_preserved(self, feature_extractor):
        """P must retain all T timesteps — not collapsed to last frame."""
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        _, P = embedder(self.sample_input)
        assert P.shape[1] == T, (
            f"Temporal dimension collapsed: expected {T}, got {P.shape[1]}"
        )

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_output_shapes(self, feature_extractor):
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        person_logits, P = embedder(self.sample_input)
        embed_dim = embedder.cnn_dim + LSTM_HIDDEN
        assert person_logits.shape == (N, PERSON_CLASSES)
        assert P.shape == (N, T, embed_dim)

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_variable_N(self, feature_extractor):
        """Output shapes must scale with N."""
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        embed_dim = embedder.cnn_dim + LSTM_HIDDEN
        for n in (1, 6, 12):
            x             = torch.randn(n, T, C, H, W)
            logits, P     = embedder(x)
            assert logits.shape == (n, PERSON_CLASSES)
            assert P.shape      == (n, T, embed_dim)

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_variable_T(self, feature_extractor):
        """Output shapes must scale with T."""
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        for t in (5, 9):
            x         = torch.randn(N, t, C, H, W)
            _, P      = embedder(x)
            assert P.shape[1] == t, f"Expected T={t}, got {P.shape[1]}"


# ─────────────────────────────────────────────
# CNN frozen tests
# ─────────────────────────────────────────────

class TestPersonEmbedderCNN:

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_cnn_frozen(self, feature_extractor):
        """CNN backbone must be frozen — no gradients."""
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        for name, param in embedder.cnn.named_parameters():
            assert not param.requires_grad, (
                f"CNN param '{name}' has requires_grad=True (should be frozen)"
            )

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_lstm_trainable(self, feature_extractor):
        """LSTM1 weights must be trainable."""
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        for name, param in embedder.lstm.named_parameters():
            assert param.requires_grad, (
                f"LSTM param '{name}' has requires_grad=False"
            )

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_person_fc_trainable(self, feature_extractor):
        embedder = PersonEmbedder(
            feature_extractor=feature_extractor,
            lstm_hidden=LSTM_HIDDEN,
            person_classes=PERSON_CLASSES,
            n_layers=1,
        )
        for name, param in embedder.person_fc.named_parameters():
            assert param.requires_grad, (
                f"person_fc param '{name}' has requires_grad=False"
            )


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestPersonEmbedderGradients:

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_gradient_flows_to_lstm(self, feature_extractor):
        """Loss backward must produce gradients in LSTM1 weights."""
        model = PersonEmbedder(feature_extractor=feature_extractor, lstm_hidden=LSTM_HIDDEN, person_classes=PERSON_CLASSES)
        x     = torch.randn(4, T, C, H, W)
        logits, _ = model(x)
        loss  = logits.sum()
        loss.backward()

        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for LSTM param '{name}'"
            assert param.grad.abs().sum() > 0, f"Zero gradient for LSTM param '{name}'"

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_no_gradient_in_cnn(self, feature_extractor):
        """CNN backbone must not accumulate gradients."""
        model = PersonEmbedder(feature_extractor=feature_extractor, lstm_hidden=LSTM_HIDDEN, person_classes=PERSON_CLASSES)
        x     = torch.randn(4, T, C, H, W)
        logits, _ = model(x)
        logits.sum().backward()

        for name, param in model.cnn.named_parameters():
            assert param.grad is None, (
                f"Unexpected gradient in frozen CNN param '{name}'"
            )

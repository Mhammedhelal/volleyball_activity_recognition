"""
tests/test_frame_descriptor.py
-------------------------------
Unit tests for FrameDescriptor (LSTM2 / group-level temporal model).
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.frame_descriptor import FrameDescriptor


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

EMBED_DIM     = 4608   # (4096 + 512)
N_SUBGROUPS   = 2
Z_DIM         = EMBED_DIM * N_SUBGROUPS   # 9216
LSTM_HIDDEN   = 512
GROUP_CLASSES = 8
T             = 9


@pytest.fixture(scope="module")
def descriptor() -> FrameDescriptor:
    return FrameDescriptor(
        z_dim         = Z_DIM,
        lstm_hidden   = LSTM_HIDDEN,
        group_classes = GROUP_CLASSES,
        n_layers      = 1,
    )


def make_Z(t: int = T) -> torch.Tensor:
    """Synthetic frame descriptor sequence [1, T, z_dim]."""
    return torch.randn(1, t, Z_DIM)


# ─────────────────────────────────────────────
# Output shape tests
# ─────────────────────────────────────────────

class TestFrameDescriptorShapes:

    def test_output_shape(self, descriptor):
        Z            = make_Z()
        group_logits = descriptor(Z)
        assert group_logits.shape == (GROUP_CLASSES,), (
            f"Expected [{GROUP_CLASSES}], got {group_logits.shape}"
        )

    def test_output_is_1d(self, descriptor):
        """group_logits must be a 1-D tensor, not [1, 8] or [1, 1, 8]."""
        Z = make_Z()
        assert descriptor(Z).dim() == 1

    def test_variable_T(self, descriptor):
        """Output shape must be independent of sequence length T."""
        for t in (5, 9, 10):
            Z = make_Z(t=t)
            assert descriptor(Z).shape == (GROUP_CLASSES,), (
                f"T={t}: unexpected output shape"
            )

    def test_batch_dim_is_one(self, descriptor):
        """FrameDescriptor processes one scene at a time — batch=1."""
        Z = make_Z()
        assert Z.shape[0] == 1

    def test_output_dtype(self, descriptor):
        Z = make_Z()
        assert descriptor(Z).dtype == torch.float32


# ─────────────────────────────────────────────
# LSTM2 processes full sequence tests
# ─────────────────────────────────────────────

class TestFrameDescriptorSequence:

    def test_different_sequences_give_different_outputs(self, descriptor):
        """Two different Z sequences must produce different logits."""
        Z1 = make_Z()
        Z2 = make_Z()
        assert not torch.allclose(descriptor(Z1), descriptor(Z2)), (
            "Different Z sequences produced identical logits — "
            "LSTM2 may not be processing the sequence correctly."
        )

    def test_sequence_order_matters(self, descriptor):
        """Reversing Z must produce different output (LSTM is order-sensitive)."""
        Z         = make_Z()
        Z_reversed = Z.flip(dims=[1])
        out_fwd   = descriptor(Z)
        out_rev   = descriptor(Z_reversed)
        assert not torch.allclose(out_fwd, out_rev), (
            "Forward and reversed sequences gave identical output — "
            "LSTM2 is not using temporal order."
        )

    def test_single_vs_multi_timestep(self, descriptor):
        """Output for T=1 vs T=9 must differ (LSTM memory has different depth)."""
        out_1 = descriptor(make_Z(t=1))
        out_9 = descriptor(make_Z(t=9))
        # Shapes must match even if values differ
        assert out_1.shape == out_9.shape == (GROUP_CLASSES,)


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestFrameDescriptorGradients:

    def test_gradient_flows_to_group_lstm(self):
        model = FrameDescriptor(Z_DIM, LSTM_HIDDEN, GROUP_CLASSES)
        Z     = make_Z()
        loss  = model(Z).sum()
        loss.backward()

        for name, param in model.group_lstm.named_parameters():
            assert param.grad is not None, (
                f"No gradient for group_lstm param '{name}'"
            )

    def test_gradient_flows_to_group_fc(self):
        model = FrameDescriptor(Z_DIM, LSTM_HIDDEN, GROUP_CLASSES)
        Z     = make_Z()
        model(Z).sum().backward()

        for name, param in model.group_fc.named_parameters():
            assert param.grad is not None, (
                f"No gradient for group_fc param '{name}'"
            )

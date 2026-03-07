"""
tests/test_hierarchical.py
--------------------------
Integration tests for HierarchicalGroupActivityModel.

These tests treat the full model as a black box and verify:
  - output shapes
  - output types
  - gradient flow end-to-end
  - stage-1 freeze behaviour
  - subgroup configuration variants
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.hierarchical_model import HierarchicalGroupActivityModel


# ─────────────────────────────────────────────
# Constants — paper hyperparameters (small for tests)
# ─────────────────────────────────────────────

CNN_DIM       = 4096
LSTM_HIDDEN_P = 512
LSTM_HIDDEN_G = 512
PERSON_CLS    = 9
GROUP_CLS     = 8
N, T, C, H, W = 12, 9, 3, 224, 224


@pytest.fixture(scope="module")
def model() -> HierarchicalGroupActivityModel:
    return HierarchicalGroupActivityModel(
        cnn_output_size = CNN_DIM,
        lstm_hidden_p   = LSTM_HIDDEN_P,
        lstm_hidden_g   = LSTM_HIDDEN_G,
        person_classes  = PERSON_CLS,
        group_classes   = GROUP_CLS,
        n_subgroups     = 2,
        pool            = "max",
    )


@pytest.fixture(scope="module")
def sample() -> torch.Tensor:
    return torch.randn(N, T, C, H, W)


# ─────────────────────────────────────────────
# Output shape tests
# ─────────────────────────────────────────────

class TestHierarchicalShapes:

    def test_group_logits_shape(self, model, sample):
        group_logits, _ = model(sample)
        assert group_logits.shape == (GROUP_CLS,), (
            f"Expected [{GROUP_CLS}], got {group_logits.shape}"
        )

    def test_person_logits_shape(self, model, sample):
        _, person_logits = model(sample)
        assert person_logits.shape == (N, PERSON_CLS), (
            f"Expected [{N}, {PERSON_CLS}], got {person_logits.shape}"
        )

    def test_group_logits_1d(self, model, sample):
        group_logits, _ = model(sample)
        assert group_logits.dim() == 1

    def test_variable_N(self, model):
        for n in (6, 12):
            x = torch.randn(n, T, C, H, W)
            group_logits, person_logits = model(x)
            assert group_logits.shape  == (GROUP_CLS,)
            assert person_logits.shape == (n, PERSON_CLS)

    @pytest.mark.parametrize("n_sub", [1, 2, 4])
    def test_subgroup_variants(self, n_sub):
        m = HierarchicalGroupActivityModel(
            cnn_output_size = CNN_DIM,
            lstm_hidden_p   = LSTM_HIDDEN_P,
            lstm_hidden_g   = LSTM_HIDDEN_G,
            n_subgroups     = n_sub,
        )
        x = torch.randn(12, T, C, H, W)
        group_logits, person_logits = m(x)
        assert group_logits.shape  == (GROUP_CLS,)
        assert person_logits.shape == (12, PERSON_CLS)

    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_pool_variants(self, pool):
        m = HierarchicalGroupActivityModel(
            cnn_output_size = CNN_DIM,
            lstm_hidden_p   = LSTM_HIDDEN_P,
            lstm_hidden_g   = LSTM_HIDDEN_G,
            pool            = pool,
        )
        group_logits, _ = m(torch.randn(N, T, C, H, W))
        assert group_logits.shape == (GROUP_CLS,)


# ─────────────────────────────────────────────
# Output value tests
# ─────────────────────────────────────────────

class TestHierarchicalValues:

    def test_group_logits_finite(self, model, sample):
        group_logits, _ = model(sample)
        assert torch.isfinite(group_logits).all(), "group_logits contains NaN or Inf"

    def test_person_logits_finite(self, model, sample):
        _, person_logits = model(sample)
        assert torch.isfinite(person_logits).all(), "person_logits contains NaN or Inf"

    def test_group_softmax_sums_to_one(self, model, sample):
        group_logits, _ = model(sample)
        prob_sum = group_logits.softmax(dim=-1).sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, f"Softmax sum: {prob_sum}"

    def test_different_inputs_give_different_outputs(self, model):
        """Model must not return constant output regardless of input."""
        x1 = torch.randn(N, T, C, H, W)
        x2 = torch.randn(N, T, C, H, W)
        g1, _ = model(x1)
        g2, _ = model(x2)
        assert not torch.allclose(g1, g2), (
            "Different inputs produced identical group logits"
        )


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestHierarchicalGradients:

    def test_group_loss_grad_reaches_lstm2(self):
        """Group-activity loss must flow back to LSTM2 (group_lstm) weights."""
        m = HierarchicalGroupActivityModel(
            CNN_DIM, LSTM_HIDDEN_P, LSTM_HIDDEN_G, n_subgroups=2
        )
        group_logits, _ = m(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in m.frame_descriptor.group_lstm.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, (
                f"No gradient reached group_lstm param '{name}'"
            )

    def test_person_loss_grad_reaches_lstm1(self):
        """Person-action loss must flow back to LSTM1 (person_embedder.lstm)."""
        m = HierarchicalGroupActivityModel(
            CNN_DIM, LSTM_HIDDEN_P, LSTM_HIDDEN_G, n_subgroups=2
        )
        _, person_logits = m(torch.randn(N, T, C, H, W))
        person_logits.sum().backward()

        for name, param in m.person_embedder.lstm.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, (
                f"No gradient reached person lstm param '{name}'"
            )

    def test_stage2_freeze_blocks_person_embedder_grad(self):
        """After freezing person_embedder, group loss must NOT update its weights."""
        m = HierarchicalGroupActivityModel(
            CNN_DIM, LSTM_HIDDEN_P, LSTM_HIDDEN_G, n_subgroups=2
        )

        # Simulate stage-2 freeze (as done in train.py)
        for param in m.person_embedder.parameters():
            param.requires_grad = False

        group_logits, _ = m(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in m.person_embedder.named_parameters():
            assert param.grad is None, (
                f"Gradient leaked into frozen person_embedder param '{name}'"
            )

    def test_stage2_freeze_still_trains_lstm2(self):
        """After freezing person_embedder, LSTM2 must still receive gradients."""
        m = HierarchicalGroupActivityModel(
            CNN_DIM, LSTM_HIDDEN_P, LSTM_HIDDEN_G, n_subgroups=2
        )
        for param in m.person_embedder.parameters():
            param.requires_grad = False

        group_logits, _ = m(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in m.frame_descriptor.group_lstm.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0, (
                f"LSTM2 param '{name}' has no gradient after stage-2 freeze"
            )

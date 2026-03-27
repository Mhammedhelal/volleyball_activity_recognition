"""
tests/test_hierarchical.py
--------------------------
Integration tests for HierarchicalGroupActivityModel.
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CNN_DIM       = 4096
LSTM_HIDDEN_P = 512
LSTM_HIDDEN_G = 512
PERSON_CLS    = 9
GROUP_CLS     = 8
N, T, C, H, W = 4, 5, 3, 64, 64


@pytest.fixture(scope="module")
def model() -> HierarchicalGroupActivityModel:
    return HierarchicalGroupActivityModel(
        feature_extractor = build_alexnet_fc7,
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
        assert group_logits.shape == (GROUP_CLS,)

    def test_person_logits_shape(self, model, sample):
        _, person_logits = model(sample)
        assert person_logits.shape == (N, PERSON_CLS)

    def test_group_logits_1d(self, model, sample):
        group_logits, _ = model(sample)
        assert group_logits.dim() == 1

    def test_variable_N(self, model):
        for n in (6, 12):
            x = torch.randn(n, T, C, H, W)
            group_logits, person_logits = model(x)
            assert group_logits.shape  == (GROUP_CLS,)
            assert person_logits.shape == (n, PERSON_CLS)

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    @pytest.mark.parametrize("n_sub", [1, 2, 4])
    def test_subgroup_variants(self, feature_extractor, n_sub):
        m = HierarchicalGroupActivityModel(
            feature_extractor = feature_extractor,
            lstm_hidden_p   = LSTM_HIDDEN_P,
            lstm_hidden_g   = LSTM_HIDDEN_G,
            n_subgroups     = n_sub,
        )
        x = torch.randn(12, T, C, H, W)
        group_logits, person_logits = m(x)
        assert group_logits.shape  == (GROUP_CLS,)
        assert person_logits.shape == (12, PERSON_CLS)

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_pool_variants(self, feature_extractor, pool):
        m = HierarchicalGroupActivityModel(
            feature_extractor = build_alexnet_fc7,
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
        assert torch.isfinite(group_logits).all()

    def test_person_logits_finite(self, model, sample):
        _, person_logits = model(sample)
        assert torch.isfinite(person_logits).all()

    def test_group_softmax_sums_to_one(self, model, sample):
        group_logits, _ = model(sample)
        prob_sum = group_logits.softmax(dim=-1).sum().item()
        assert abs(prob_sum - 1.0) < 1e-5

    def test_different_inputs_give_different_outputs(self, model):
        x1 = torch.randn(N, T, C, H, W)
        x2 = torch.randn(N, T, C, H, W)
        g1, _ = model(x1)
        g2, _ = model(x2)
        assert not torch.allclose(g1, g2)


# ─────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────

class TestHierarchicalGradients:

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_group_loss_grad_reaches_lstm2(self, feature_extractor):
        model = HierarchicalGroupActivityModel(
            feature_extractor=feature_extractor, lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G, n_subgroups=2
        )
        group_logits, _ = model(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in model.person_embedder.lstm.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.abs().sum() > 0

        for name, param in model.frame_descriptor.group_lstm.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.abs().sum() > 0
        del model
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_person_loss_grad_reaches_lstm1(self, feature_extractor):
        model = HierarchicalGroupActivityModel(
            feature_extractor=feature_extractor, lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G, n_subgroups=2
        )
        _, person_logits = model(torch.randn(N, T, C, H, W))
        person_logits.sum().backward()

        for name, param in model.person_embedder.lstm.named_parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.abs().sum() > 0
        del model
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_stage2_freeze_blocks_person_embedder_grad(self, feature_extractor):
        m = HierarchicalGroupActivityModel(
            feature_extractor=feature_extractor, lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G, n_subgroups=2
        )
        for param in m.person_embedder.parameters():
            param.requires_grad = False

        group_logits, _ = m(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in m.person_embedder.named_parameters():
            assert param.grad is None, (
                f"Gradient leaked into frozen person_embedder param '{name}'"
            )
        del m
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_stage2_freeze_still_trains_lstm2(self, feature_extractor):
        m = HierarchicalGroupActivityModel(
            feature_extractor=feature_extractor, lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G, n_subgroups=2
        )
        for param in m.person_embedder.parameters():
            param.requires_grad = False

        group_logits, _ = m(torch.randn(N, T, C, H, W))
        group_logits.sum().backward()

        for name, param in m.frame_descriptor.group_lstm.named_parameters():
            assert param.grad is not None and param.grad.abs().sum() > 0
        del m
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Device tests  — compare torch.device objects, not str vs device
# ─────────────────────────────────────────────

@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
class TestHierarchicalDevice:

    def test_forward_backward(self, device_str):
        device = torch.device(device_str)
        x = torch.randn(N, T, C, H, W, device=device)
        model = HierarchicalGroupActivityModel(
            feature_extractor=build_alexnet_fc7,
            lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G,
            n_subgroups=2,
        ).to(device)

        g_logits, p_logits = model(x)

        # Compare torch.device objects — not str vs torch.device
        assert g_logits.device.type == device.type
        assert p_logits.device.type == device.type

        loss = g_logits.sum() + p_logits.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.device.type == device.type
        del model
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Eval mode determinism
# ─────────────────────────────────────────────

class TestHierarchicalEval:

    @pytest.mark.parametrize("feature_extractor", [build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large])
    def test_eval_mode_determinism(self, feature_extractor):
        model = HierarchicalGroupActivityModel(
            feature_extractor=feature_extractor,
            lstm_hidden_p=LSTM_HIDDEN_P,
            lstm_hidden_g=LSTM_HIDDEN_G,
            n_subgroups=2,
        )
        x = torch.randn(4, T, C, H, W)
        model.eval()
        g1, p1 = model(x)
        g2, p2 = model(x)
        assert torch.allclose(g1, g2, atol=1e-6)
        assert torch.allclose(p1, p2, atol=1e-6)
        del model
        torch.cuda.empty_cache()
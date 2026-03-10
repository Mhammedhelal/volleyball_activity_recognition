"""
tests/test_subgroup_pooler.py
-----------------------------
Unit tests for SubGroupPooler and make_subgroup_indices.
"""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.subgroup_pooler import SubGroupPooler
from src.utils.subgroups import make_subgroup_indices


# ─────────────────────────────────────────────
# make_subgroup_indices tests
# ─────────────────────────────────────────────

class TestMakeSubgroupIndices:

    def test_single_group(self):
        idx = make_subgroup_indices(12, 1)
        assert idx == [list(range(12))]

    def test_two_groups_even(self):
        idx = make_subgroup_indices(12, 2)
        assert idx == [list(range(6)), list(range(6, 12))]

    def test_two_groups_odd(self):
        """11 players → groups of 6 and 5."""
        idx = make_subgroup_indices(11, 2)
        assert len(idx[0]) == 6
        assert len(idx[1]) == 5

    def test_four_groups(self):
        idx = make_subgroup_indices(12, 4)
        assert len(idx) == 4
        assert all(len(g) == 3 for g in idx)

    def test_covers_all_players(self):
        """Union of all subgroup indices must equal {0, …, N-1}."""
        for n, d in [(12, 2), (11, 2), (12, 4), (10, 4)]:
            idx  = make_subgroup_indices(n, d)
            flat = [i for group in idx for i in group]
            assert sorted(flat) == list(range(n)), (
                f"n={n}, d={d}: indices don't cover all players"
            )

    def test_no_overlap(self):
        """No player should appear in more than one subgroup."""
        for n, d in [(12, 2), (12, 4)]:
            idx  = make_subgroup_indices(n, d)
            flat = [i for group in idx for i in group]
            assert len(flat) == len(set(flat)), (
                f"n={n}, d={d}: duplicate player indices found"
            )

    def test_contiguous(self):
        """Each subgroup must contain a contiguous range of indices."""
        idx = make_subgroup_indices(12, 4)
        for group in idx:
            assert group == list(range(group[0], group[-1] + 1))


# ─────────────────────────────────────────────
# SubGroupPooler tests
# ─────────────────────────────────────────────

N, T, EMBED_DIM = 12, 9, 4608   # (4096 + 512)


def make_P(n=N, t=T, d=EMBED_DIM) -> torch.Tensor:
    return torch.randn(n, t, d)


class TestSubGroupPooler:

    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_output_shape_single_group(self, pool):
        pooler = SubGroupPooler(pool=pool)
        P      = make_P()
        idx    = make_subgroup_indices(N, 1)
        Z      = pooler(P, idx)
        assert Z.shape == (1, T, EMBED_DIM), (
            f"Expected [1, {T}, {EMBED_DIM}], got {Z.shape}"
        )

    @pytest.mark.parametrize("pool", ["max", "avg"])
    def test_output_shape_two_groups(self, pool):
        pooler = SubGroupPooler(pool=pool)
        P      = make_P()
        idx    = make_subgroup_indices(N, 2)
        Z      = pooler(P, idx)
        assert Z.shape == (1, T, EMBED_DIM * 2), (
            f"Expected [1, {T}, {EMBED_DIM * 2}], got {Z.shape}"
        )

    @pytest.mark.parametrize("n_sub", [1, 2, 4])
    def test_z_dim_scales_with_subgroups(self, n_sub):
        pooler = SubGroupPooler(pool="max")
        P      = make_P()
        idx    = make_subgroup_indices(N, n_sub)
        Z      = pooler(P, idx)
        expected_z_dim = EMBED_DIM * n_sub
        assert Z.shape == (1, T, expected_z_dim), (
            f"n_sub={n_sub}: expected z_dim={expected_z_dim}, got {Z.shape[-1]}"
        )

    def test_temporal_dim_preserved(self):
        """Z must have T timesteps — one frame descriptor per frame."""
        pooler = SubGroupPooler(pool="max")
        for t in (5, 9):
            P   = make_P(t=t)
            idx = make_subgroup_indices(N, 2)
            Z   = pooler(P, idx)
            assert Z.shape[1] == t, f"Expected T={t}, got {Z.shape[1]}"

    def test_max_pool_selects_maximum(self):
        """Max pool output must equal the element-wise max over the subgroup."""
        pooler = SubGroupPooler(pool="max")
        P      = torch.arange(6 * 1 * 4, dtype=torch.float).reshape(6, 1, 4)
        idx    = [list(range(6))]
        Z      = pooler(P, idx)                      # [1, 1, 4]
        expected = P[:, 0, :].max(dim=0).values      # [4]
        assert torch.allclose(Z[0, 0, :], expected)

    def test_avg_pool_computes_mean(self):
        """Avg pool output must equal the element-wise mean over the subgroup."""
        pooler = SubGroupPooler(pool="avg")
        P      = torch.arange(6 * 1 * 4, dtype=torch.float).reshape(6, 1, 4)
        idx    = [list(range(6))]
        Z      = pooler(P, idx)                      # [1, 1, 4]
        expected = P[:, 0, :].mean(dim=0)            # [4]
        assert torch.allclose(Z[0, 0, :], expected)

    def test_invalid_pool_raises(self):
        with pytest.raises(AssertionError):
            SubGroupPooler(pool="sum")

    def test_output_batch_dim_is_one(self):
        """Z must always have batch dim = 1 (one scene at a time)."""
        pooler = SubGroupPooler(pool="max")
        Z      = pooler(make_P(), make_subgroup_indices(N, 2))
        assert Z.shape[0] == 1

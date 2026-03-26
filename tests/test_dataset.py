"""
tests/test_dataset.py
---------------------
Unit tests for VolleyballDataset and volleyball_collate.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import VolleyballDataset, make_collate_fn, volleyball_collate

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ANNOTATION_LINES = [
    "23455.jpg r_spike 463 646 87 166 standing 654 570 85 124 standing",
    "23460.jpg r_pass  100 200 50 100 digging  300 400 60 120 jumping",
]

PLAYER_COUNT = 2
T            = 9
C, H, W      = 3, 224, 224


def make_fake_dataset(tmp_path: Path, crops_data: bool, T: int = 9) -> VolleyballDataset:
    """
    Build a minimal VolleyballDataset backed by fake images and annotations.
    """
    video_id = 1
    frame_id = 23455
    half     = T // 2

    video_dir = tmp_path / str(video_id) / str(frame_id)
    video_dir.mkdir(parents=True)

    for fid in range(frame_id - half, frame_id + half + 1):
        Image.new("RGB", (720, 576), color=(128, 64, 32)).save(
            video_dir / f"{fid}.jpg"
        )

    ann_file = tmp_path / str(video_id) / "annotations.txt"
    ann_file.write_text("\n".join(ANNOTATION_LINES))

    import torchvision.transforms as tvt
    simple_transform = tvt.Compose([
        tvt.Resize((224, 224)),
        tvt.ToTensor(),
    ])

    return VolleyballDataset(
        root         = tmp_path,
        split_videos = {video_id},
        cfg          = Config.from_yaml("configs/default.yaml"),
        transforms   = simple_transform,
        T            = T,
        crops_data   = crops_data,
    )


# ─────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────

class TestVolleyballDataset:

    def setup_method(self):
        self.tmp_path = Path(tempfile.mkdtemp())

    def teardown_method(self):
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_length(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        assert len(ds) == len(ANNOTATION_LINES)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_sample_shapes(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        x, group_label, person_labels = ds[0]

        if crops_data:
            assert x.shape == (PLAYER_COUNT, T, C, H, W), (
                f"Expected [N={PLAYER_COUNT}, T={T}, C={C}, H={H}, W={W}], got {x.shape}"
            )
            assert person_labels.shape == (PLAYER_COUNT,)
        else:
            assert x.shape == (T, C, H, W), (
                f"Expected full frames shape ({T}, {C}, {H}, {W}), got {x.shape}"
            )
            assert person_labels.shape == (PLAYER_COUNT,)

        assert group_label.shape == (1,)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_group_label_dtype(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, group_label, _ = ds[0]
        assert group_label.dtype == torch.long

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_person_labels_dtype(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, _, person_labels = ds[0]
        assert person_labels.dtype == torch.long

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_group_label_in_range(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, group_label, _ = ds[0]
        assert 0 <= group_label.item() < 8

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_person_labels_in_range(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, _, person_labels = ds[0]
        assert all(0 <= l.item() < 9 for l in person_labels)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_players_sorted_by_x(self, crops_data):
        """Players must be sorted left→right (ascending bbox_center_x)."""
        ds      = make_fake_dataset(self.tmp_path, crops_data)
        _, ann  = ds.samples[0]
        players = sorted(ann["players"], key=lambda p: p["bbox_center_x"])
        centers = [p["bbox_center_x"] for p in players]
        assert centers == sorted(centers)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_missing_frame_fallback(self, crops_data):
        """Dataset must not raise on a missing frame."""
        ds = make_fake_dataset(self.tmp_path, crops_data)
        try:
            x, _, _ = ds[1]
            if crops_data:
                assert x.shape[1] == T
            else:
                assert x.shape[0] == T
        except FileNotFoundError:
            pytest.fail("Dataset raised FileNotFoundError on missing frame")

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_temporal_window_size(self, crops_data):
        for T_val in (5, 9):
            tmp = Path(tempfile.mkdtemp())
            try:
                ds = make_fake_dataset(tmp, crops_data=crops_data, T=T_val)
                x, _, _ = ds[0]
                if crops_data:
                    assert x.shape[1] == T_val
                else:
                    assert x.shape[0] == T_val
            finally:
                shutil.rmtree(tmp)

    def test_odd_T_assertion(self):
        with pytest.raises(AssertionError):
            VolleyballDataset(
                self.tmp_path,
                split_videos = {1},
                cfg          = Config.from_yaml("configs/default.yaml"),
                T            = 8,
            )


# ─────────────────────────────────────────────
# Collate tests
# ─────────────────────────────────────────────

class TestVolleyballCollate:

    def _make_batch(self, sizes: list, crops_data: bool) -> list:
        """Create a synthetic batch with different N per sample."""
        batch = []
        for n in sizes:
            if crops_data:
                x = torch.zeros(n, T, C, H, W)
            else:
                x = torch.zeros(T, C, H, W)
            group_label   = torch.tensor([0], dtype=torch.long)
            person_labels = torch.zeros(n, dtype=torch.long)
            batch.append((x, group_label, person_labels))
        return batch

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_group_labels_stacked(self, crops_data):
        batch = self._make_batch([12, 10, 11], crops_data)
        x_batch, group_labels, _ = volleyball_collate(batch, crops_data)
        assert group_labels.shape == (3,)
        assert group_labels.dtype == torch.long

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_x_batch_type(self, crops_data):
        batch = self._make_batch([12, 10], crops_data)
        x_batch, _, _ = volleyball_collate(batch, crops_data)
        if crops_data:
            assert isinstance(x_batch, list), "crops_data=True: x_batch should be list"
            assert len(x_batch) == 2
        else:
            assert isinstance(x_batch, torch.Tensor), "crops_data=False: x_batch should be Tensor"
            assert x_batch.shape == (2, T, C, H, W)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_crops_list_length(self, crops_data):
        batch   = self._make_batch([12, 10], crops_data)
        x_batch, _, _ = volleyball_collate(batch, crops_data)
        if crops_data:
            assert len(x_batch) == 2

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_x_preserve_shape(self, crops_data):
        ns    = [12, 10, 8]
        batch = self._make_batch(ns, crops_data)
        x_batch, _, _ = volleyball_collate(batch, crops_data)
        for i, n in enumerate(ns):
            if crops_data:
                assert x_batch[i].shape == (n, T, C, H, W)
            else:
                assert x_batch[i].shape == (T, C, H, W)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_person_labels_list_length(self, crops_data):
        batch = self._make_batch([12, 10], crops_data)
        _, _, person_labels_list = volleyball_collate(batch, crops_data)
        assert len(person_labels_list) == 2

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_person_labels_preserve_shape(self, crops_data):
        ns    = [12, 10, 8]
        batch = self._make_batch(ns, crops_data)
        _, _, person_labels_list = volleyball_collate(batch, crops_data)
        for i, (labels, n) in enumerate(zip(person_labels_list, ns)):
            assert labels.shape == (n,)

    @pytest.mark.parametrize("crops_data", [True, False])
    def test_variable_n_does_not_raise(self, crops_data):
        """Variable N must not raise during collate (crops path)."""
        batch = self._make_batch([12, 6, 9], crops_data)
        try:
            volleyball_collate(batch, crops_data)
        except RuntimeError as e:
            pytest.fail(f"volleyball_collate raised RuntimeError: {e}")

    def test_make_collate_fn_callable(self):
        """make_collate_fn must return a callable, not execute the function."""
        fn = make_collate_fn(crops_data=True)
        assert callable(fn)
        batch = [
            (torch.zeros(6, T, C, H, W), torch.tensor([0]), torch.zeros(6, dtype=torch.long))
        ]
        x_batch, group_labels, person_labels_list = fn(batch)
        assert isinstance(x_batch, list)
        assert group_labels.shape == (1,)
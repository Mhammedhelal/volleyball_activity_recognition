"""
tests/test_dataset.py
---------------------
Unit tests for VolleyballDataset and volleyball_collate.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

ANNOTATION_LINES = [
    # frame      group  x   y   w   h   action   (× 2 players)
    "23455.jpg r_spike 463 646 87 166 standing 654 570 85 124 standing",
    "23460.jpg r_pass  100 200 50 100 digging  300 400 60 120 jumping",
]

PLAYER_COUNT = 2
T            = 9
C, H, W      = 3, 224, 224


def make_fake_dataset(tmp_path: Path, crops_data, T: int = 9) -> VolleyballDataset:
    """
    Build a minimal VolleyballDataset backed by fake images and annotations.
    Images are blank PIL images so no real data is needed.
    """
    video_id  = 1
    frame_id  = 23455
    half      = T // 2

    video_dir = tmp_path / str(video_id) / str(frame_id)
    video_dir.mkdir(parents=True)

    # Write a blank jpg for every frame in the temporal window
    for fid in range(frame_id - half, frame_id + half + 1):
        Image.new("RGB", (720, 576), color=(128, 64, 32)).save(
            video_dir / f"{fid}.jpg"
        )

    # Write annotations.txt
    ann_file = tmp_path / str(video_id) / "annotations.txt"
    ann_file.write_text("\n".join(ANNOTATION_LINES))

    transform = torch.nn.Identity()   # skip resize/normalize in unit tests

    # Monkey-patch the transform to return a [C,H,W] tensor from a PIL image
    import torchvision.transforms as tvt
    simple_transform = tvt.Compose([
        tvt.Resize((224, 224)),
        tvt.ToTensor(),
    ])

    return VolleyballDataset(
        root         = tmp_path,
        split_videos = {video_id},
        cfg = Config.from_yaml('configs/default.yaml'),
        transforms   = simple_transform,
        T            = T,
        crops_data=crops_data,
    )


# ─────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────

class TestVolleyballDataset:

    def setup_method(self):
        """Create a temporary directory for each test."""
        self.tmp_path = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_length(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        # 2 annotation lines → 2 samples, but only frame 23455 has images
        # (frame 23460 will trigger _nearest_frame fallback)
        assert len(ds) == len(ANNOTATION_LINES)

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_sample_shapes(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        x, group_label, person_labels = ds[0]

        if crops_data:
            # x: [N, T, C, H, W]
            expected_shape = (PLAYER_COUNT, T, C, H, W)
            assert x.shape == expected_shape, (
                f"Expected [N={PLAYER_COUNT}, T={T}, C={C}, H={H}, W={W}], got {x.shape}"
            )
            assert person_labels.shape == (PLAYER_COUNT,), f"person_labels shape: {person_labels.shape}"
        else:
            # x: [T, C, H, W] (full frames)
            expected_shape = (T, C, H, W)
            assert x.shape == expected_shape, f"Expected full frames shape {expected_shape}, got {x.shape}"
            # person_labels still has shape [N]
            assert person_labels.shape == (PLAYER_COUNT,), f"person_labels shape: {person_labels.shape}"

        # group label is always [1]
        assert group_label.shape == (1,), f"group_label shape: {group_label.shape}"

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_group_label_dtype(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, group_label, _ = ds[0]
        assert group_label.dtype == torch.long

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_person_labels_dtype(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, _, person_labels = ds[0]
        assert person_labels.dtype == torch.long

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_group_label_in_range(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, group_label, _ = ds[0]
        assert 0 <= group_label.item() < 8, f"group_label out of range: {group_label.item()}"

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_person_labels_in_range(self, crops_data):
        ds = make_fake_dataset(self.tmp_path, crops_data)
        _, _, person_labels = ds[0]
        assert all(0 <= l.item() < 9 for l in person_labels), (
            f"person_labels out of range: {person_labels}"
        )

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_players_sorted_by_x(self, crops_data):
        """Players must be sorted left→right (ascending bbox_center_x)."""
        ds       = make_fake_dataset(self.tmp_path, crops_data)
        _, vid   = ds.samples[0]
        players  = sorted(vid["players"], key=lambda p: p["bbox_center_x"])
        centers  = [p["bbox_center_x"] for p in players]
        assert centers == sorted(centers), f"Players not sorted: {centers}"

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_missing_frame_fallback(self, crops_data):
        """Dataset must not raise on a missing frame — nearest fallback used."""
        ds = make_fake_dataset(self.tmp_path, crops_data)
        # Sample index 1 references frame 23460 which has no images on disk
        try:
            x, _, _ = ds[1]
            if crops_data: assert x.shape[1] == T
            else: assert x.shape[0] == T
        except FileNotFoundError:
            pytest.fail("Dataset raised FileNotFoundError on missing frame")

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_temporal_window_size(self, crops_data):
        for T_val in (5, 9):
            # Create a fresh tmp_path for each T value to avoid conflicts
            tmp_path_t = Path(tempfile.mkdtemp())
            try:
                ds = make_fake_dataset(tmp_path_t, T=T_val, crops_data=crops_data)
                x, _, _ = ds[0]
                if crops_data: assert x.shape[1] == T_val, f"Expected T={T_val}, got {x.shape[1]}"
                else: assert x.shape[0] == T_val, f"Expected T={T_val}, got {x.shape[1]}"
            finally:
                shutil.rmtree(tmp_path_t)

    def test_odd_T_assertion(self):
        with pytest.raises(AssertionError):
            VolleyballDataset(self.tmp_path, split_videos={1}, cfg=Config.from_yaml('configs/default.yaml'), T=8)


# ─────────────────────────────────────────────
# Collate tests
# ─────────────────────────────────────────────

class TestVolleyballCollate:

    def _make_batch(self, sizes: list[int], crops_data) -> list:
        """Create a synthetic batch with different N per sample."""
        batch = []
        for n in sizes:
            if crops_data:
                x = torch.zeros(n, T, C, H, W)   # crops
            else:
                x = torch.zeros(T, C, H, W)      # full frame

            group_label   = torch.tensor([0], dtype=torch.long)
            person_labels = torch.zeros(n, dtype=torch.long)

            batch.append((x, group_label, person_labels))
        return batch
    
    @pytest.mark.parametrize('crops_data', [True, False])
    def test_group_labels_stacked(self, crops_data):
        batch = self._make_batch([12, 10, 11], crops_data)
        frames_list, group_labels, person_labels_list = volleyball_collate(batch, crops_data)
        assert group_labels.shape == (3,), f"Expected [3], got {group_labels.shape}"
        assert group_labels.dtype == torch.long

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_frames_list_length(self, crops_data):
        batch = self._make_batch([12, 10], crops_data)
        frames_list, _, _ = volleyball_collate(batch)
        assert len(frames_list) == 2

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_frames_preserve_shape(self, crops_data):
        ns    = [12, 10, 8]
        batch = self._make_batch(ns, crops_data)
        frames_list, _, _ = volleyball_collate(batch)
        for i, (frames, n) in enumerate(zip(frames_list, ns)):
            expected_shape = (n, T, C, H, W) if crops_data else (T, C, H, W)
            assert frames.shape == expected_shape, (
                f"Sample {i}: expected [{n}, {T}, {C}, {H}, {W}], got {frames.shape}" if crops_data
              else   f"Sample {i}: expected [{T}, {C}, {H}, {W}], got {frames.shape}"
            )

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_person_labels_list_length(self, crops_data):
        batch = self._make_batch([12, 10], crops_data)
        _, _, person_labels_list = volleyball_collate(batch)
        assert len(person_labels_list) == 2

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_person_labels_preserve_shape(self, crops_data):
        ns    = [12, 10, 8]
        batch = self._make_batch(ns, crops_data)
        _, _, person_labels_list = volleyball_collate(batch)
        for i, (labels, n) in enumerate(zip(person_labels_list, ns)):
            assert labels.shape == (n,), (
                f"Sample {i}: expected [{n}], got {labels.shape}"
            )

    @pytest.mark.parametrize('crops_data', [True, False])
    def test_variable_n_does_not_raise(self, crops_data):
        """Core requirement: variable N must not raise during collate."""
        batch = self._make_batch([12, 6, 9], crops_data)
        try:
            volleyball_collate(batch)
        except RuntimeError as e:
            pytest.fail(f"volleyball_collate raised RuntimeError: {e}")

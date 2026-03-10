"""
src/data/dataset.py
-------------------
PyTorch Dataset for volleyball activity recognition.

__getitem__ returns a 4-tuple:
    crops        [N, T, C, H, W]  – per-person temporal crops  (used by B2/B3/B5/B6/B7 + full model)
    full_frame   [T, C, H, W]     – full-frame temporal sequence (used by B1/B4)
    group_label  [1]              – team activity class index
    person_labels[N]              – individual action class indices

volleyball_collate handles the variable-N dimension and the new full_frame field.
"""

from torch.utils.data import Dataset
from pathlib import Path
import torch
from PIL import Image

import sys
from src.config import Config


class VolleyballDataset(Dataset):
    def __init__(
        self,
        root:         Path,
        split_videos: set[int],
        cfg:          Config,
        transforms=None,
        T:            int = 9,
    ):
        assert T % 2 == 1, f"T must be odd for a symmetric window, got {T}"

        self.root       = Path(root)
        self.cfg        = cfg
        self.transforms = transforms
        self.T          = T
        self.half       = T // 2

        self.samples = []   # list of (video_id, annotation_dict)

        for video_id in split_videos:
            ann_file = self.root / str(video_id) / "annotations.txt"
            for ann in self._parse_annotations(ann_file):
                self.samples.append((video_id, ann))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ann = self.samples[idx]

        frame_id    = ann["frame_id"]
        group_label = ann["group_label"]
        players     = ann["players"]

        # Sort players left → right (required by subgroup logic)
        players = sorted(players, key=lambda p: p["bbox_center_x"])

        # Symmetric temporal window around frame_id
        frame_ids = list(range(frame_id - self.half, frame_id + self.half + 1))
        clip_dir  = self.root / str(video_id) / str(frame_id)

        # Pre-load all T frames once to avoid re-opening the same file per player
        frames_pil: dict[int, Image.Image] = {}
        for fid in frame_ids:
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                frames_pil[fid] = Image.open(img_path).convert("RGB")
            else:
                frames_pil[fid] = self._nearest_frame(clip_dir, fid, frame_ids)

        assert len(players) > 0, f"Sample must have at least 1 player"

        # ── person crops: [N, T, C, H, W] ────────────────────────────────────
        person_crops = []
        for p in players:
            x, y, w, h = p["bbox"]
            t_crops = []
            for fid in frame_ids:
                img    = frames_pil[fid]
                iw, ih = img.size
                x1 = max(0, x);       y1 = max(0, y)
                x2 = min(iw, x + w);  y2 = min(ih, y + h)
                if x2 <= x1 or y2 <= y1:
                    x1, y1 = 0, 0
                    x2, y2 = min(1, iw), min(1, ih)
                crop = img.crop((x1, y1, x2, y2))
                if self.transforms:
                    crop = self.transforms(crop)        # [C, H, W]
                t_crops.append(crop)
            person_crops.append(torch.stack(t_crops, dim=0))   # [T, C, H, W]

        crops = torch.stack(person_crops, dim=0)        # [N, T, C, H, W]

        # ── full frames: [T, C, H, W] ─────────────────────────────────────────
        # Used by frame-level baselines (B1, B4).  We apply the same transforms
        # as for crops so the backbone sees the same normalisation.
        full_frames = []
        for fid in frame_ids:
            img = frames_pil[fid]
            if self.transforms:
                img = self.transforms(img)              # [C, H, W]
            full_frames.append(img)
        full_frame = torch.stack(full_frames, dim=0)    # [T, C, H, W]

        # ── labels ────────────────────────────────────────────────────────────
        person_labels = torch.tensor(
            [p["action_id"] for p in players], dtype=torch.long
        )                                               # [N]

        group_label_tensor = torch.tensor([group_label], dtype=torch.long)  # [1]

        assert crops.shape[0] == len(person_labels), "Mismatch: N players"
        assert crops.shape[1] == self.T,             "Mismatch: temporal window"

        return crops, full_frame, group_label_tensor, person_labels

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_annotations(self, ann_file: Path) -> list[dict]:
        samples = []
        with open(ann_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            tokens    = line.split()
            frame_str = tokens[0]
            group_str = tokens[1]
            frame_id  = int(frame_str.replace(".jpg", ""))
            group_label = self.cfg.labels.group_activities.index(group_str)

            player_tokens = tokens[2:]
            assert len(player_tokens) % 5 == 0, (
                f"Invalid annotation in {ann_file}, line: {line!r}"
            )

            players = []
            for i in range(0, len(player_tokens), 5):
                x, y, w, h = map(int, player_tokens[i : i + 4])
                action     = player_tokens[i + 4]
                players.append({
                    "bbox":          (x, y, w, h),
                    "bbox_center_x": x + w / 2,
                    "action":        action,
                    "action_id":     self.cfg.labels.person_actions.index(action),
                })

            samples.append({
                "frame_id":    frame_id,
                "group_label": group_label,
                "players":     players,
            })

        return samples

    @staticmethod
    def _nearest_frame(
        clip_dir:    Path,
        missing_fid: int,
        frame_ids:   list[int],
    ) -> Image.Image:
        for fid in sorted(frame_ids, key=lambda f: abs(f - missing_fid)):
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        return Image.new("RGB", (224, 224), color=0)


def volleyball_collate(batch: list) -> tuple:
    """
    Custom collate for the Volleyball dataset.

    Each sample from VolleyballDataset.__getitem__ is:
        crops         [N_i, T, C, H, W]   person crops
        full_frame    [T, C, H, W]         full-frame sequence
        group_label   [1]                  team activity class index
        person_labels [N_i]                individual action class indices

    Returns
    -------
    crops_list         : list[B] of [N_i, T, C, H, W]   variable N — kept as list
    full_frames        : FloatTensor [B, T, C, H, W]     fixed shape — stacked
    group_labels       : LongTensor  [B]
    person_labels_list : list[B] of [N_i]                variable N — kept as list
    """
    crops_list         = [s[0] for s in batch]
    full_frames        = torch.stack([s[1] for s in batch], dim=0)     # [B, T, C, H, W]
    group_labels       = torch.cat([s[2] for s in batch], dim=0)       # [B]
    person_labels_list = [s[3] for s in batch]

    return crops_list, full_frames, group_labels, person_labels_list
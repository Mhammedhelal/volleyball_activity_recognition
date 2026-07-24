"""
src/data/dataset.py
-------------------
PyTorch Dataset for volleyball activity recognition.

__getitem__ returns a 3-tuple whose first element depends on *crops_data*:

    crops_data=True  (default, used by B2/B3/B5/B6/B7 and the full model)
        x            [N, T, C, H, W]  – per-person temporal crops
        group_label  [1]              – team activity class index
        person_labels[N]              – individual action class indices

    crops_data=False  (used by B1 / B4)
        x            [T, C, H, W]    – full-frame temporal sequence
        group_label  [1]
        person_labels[N]

volleyball_collate handles the variable-N dimension and both x shapes.
The *crops_data* flag must be consistent between the dataset and the collate
function.  Use build_loader() in scripts/common.py to ensure this.
"""

from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import Config


class VolleyballDataset(Dataset):
    def __init__(
        self,
        root:         Path,
        split_videos: set,
        cfg:          Config,
        transforms=None,
        T:            int  = 9,
        crops_data:   bool = True,
    ):
        assert T % 2 == 1, f"T must be odd for a symmetric window, got {T}"

        self.root        = Path(root)
        self.cfg         = cfg
        self.transforms  = transforms
        self.T           = T
        self.half        = T // 2
        self.crops_data  = crops_data

        self.samples = []   # list of (video_id, annotation_dict)

        for video_id in split_videos:
            ann_file = self.root / str(video_id) / "annotations.txt"
            for ann in self._parse_annotations(ann_file):
                self.samples.append((video_id, ann))

    # ------------------------------------------------------------------

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
        frames_pil: dict = {}
        for fid in frame_ids:
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                frames_pil[fid] = Image.open(img_path).convert("RGB")
            else:
                frames_pil[fid] = self._nearest_frame(clip_dir, fid, frame_ids)

        assert len(players) > 0, "Sample must have at least 1 player"

        if self.crops_data:
            # ── person crops: [N, T, C, H, W] ────────────────────────────
            person_crops = []
            for p in players:
                x_b, y_b, w_b, h_b = p["bbox"]
                t_crops = []
                for fid in frame_ids:
                    img    = frames_pil[fid]
                    iw, ih = img.size
                    x1 = max(0, x_b);         y1 = max(0, y_b)
                    x2 = min(iw, x_b + w_b);  y2 = min(ih, y_b + h_b)
                    if x2 <= x1 or y2 <= y1:
                        x1, y1 = 0, 0
                        x2, y2 = min(1, iw), min(1, ih)
                    crop = img.crop((x1, y1, x2, y2))
                    if self.transforms:
                        crop = self.transforms(crop)   # [C, H, W]
                    t_crops.append(crop)
                person_crops.append(torch.stack(t_crops, dim=0))  # [T, C, H, W]

            x = torch.stack(person_crops, dim=0)   # [N, T, C, H, W]

            assert x.shape[0] == len(players), "Mismatch: N players"
            assert x.shape[1] == self.T,       "Mismatch: temporal window"

        else:
            # ── full frames: [T, C, H, W] ─────────────────────────────────
            full_frames = []
            for fid in frame_ids:
                img = frames_pil[fid]
                if self.transforms:
                    img = self.transforms(img)   # [C, H, W]
                full_frames.append(img)
            x = torch.stack(full_frames, dim=0)  # [T, C, H, W]

            assert x.shape[0] == self.T, "Mismatch: temporal window"

        # ── labels ────────────────────────────────────────────────────────
        person_labels      = torch.tensor(
            [p["action_id"] for p in players], dtype=torch.long
        )                                               # [N]
        group_label_tensor = torch.tensor(
            [group_label], dtype=torch.long
        )                                               # [1]

        return x, group_label_tensor, person_labels

    # ── helpers ───────────────────────────────────────────────────────────

    def _parse_annotations(self, ann_file: Path) -> list:
        samples = []
        with open(ann_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            tokens    = line.split()
            frame_str = tokens[0]
            group_str = tokens[1]
            group_str = tokens[1].replace("-", "_")
            frame_id  = int(frame_str.replace(".jpg", ""))
            group_label = self.cfg.labels.group_activities.index(group_str)

            player_tokens = tokens[2:]
            assert len(player_tokens) % 5 == 0, (
                f"Invalid annotation in {ann_file}, line: {line!r}"
            )

            players = []
            for i in range(0, len(player_tokens), 5):
                x, y, w, h = map(int, player_tokens[i : i + 4])
                action      = player_tokens[i + 4]
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
        frame_ids:   list,
    ) -> Image.Image:
        for fid in sorted(frame_ids, key=lambda f: abs(f - missing_fid)):
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        return Image.new("RGB", (224, 224), color=0)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def volleyball_collate(batch: list, crops_data: bool = True) -> tuple:
    """
    Custom collate for the Volleyball dataset.

    Each sample is a 3-tuple  (x, group_label[1], person_labels[N]).

    When crops_data=True
        x has shape [N_i, T, C, H, W] — variable N, kept as list.

    When crops_data=False
        x has shape [T, C, H, W] — fixed shape, stacked into [B, T, C, H, W].

    Returns
    -------
    x_batch            : list[Tensor [N_i, T, C, H, W]]  if crops_data
                       : Tensor [B, T, C, H, W]           otherwise
    group_labels       : LongTensor [B]
    person_labels_list : list[B] of LongTensor [N_i]
    """
    if crops_data:
        x_batch = [s[0] for s in batch]                        # variable N — list
    else:
        x_batch = torch.stack([s[0] for s in batch], dim=0)   # [B, T, C, H, W]

    group_labels       = torch.cat([s[1] for s in batch], dim=0)   # [B]
    person_labels_list = [s[2] for s in batch]

    return x_batch, group_labels, person_labels_list


def make_collate_fn(crops_data: bool = True):
    """Return a collate callable with *crops_data* baked in.

    Use this instead of passing volleyball_collate directly to DataLoader so
    that the crops_data flag is captured in a closure:

        loader = DataLoader(ds, collate_fn=make_collate_fn(crops_data=True))
    """
    return partial(volleyball_collate, crops_data=crops_data)
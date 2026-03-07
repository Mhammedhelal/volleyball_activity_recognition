from torch.utils.data import Dataset
from pathlib import Path
import torch
from PIL import Image


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config


class VolleyballDataset(Dataset):
    def __init__(
        self,
        root:         Path,
        split_videos: set[int],
        cfg: Config,
        transforms=None,
        T: int = 9,
    ):
        assert T % 2 == 1, f"T must be odd for a symmetric window, got {T}"

        self.root       = Path(root)
        self.cfg = cfg
        self.transforms = transforms
        self.T          = T
        self.half       = T // 2

        self.samples = []   # list of (video_id, annotation)

        for video_id in split_videos:
            ann_file = self.root / str(video_id) / "annotations.txt"
            for ann in self._parse_annotations(ann_file):
                # FIX 1: _parse_annotations returned bare dicts, but the
                # original __init__ called self.samples.extend() expecting
                # (video_id, ann) tuples.  The parser never attached video_id,
                # so __getitem__ had no way to build the correct path.
                self.samples.append((video_id, ann))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ann = self.samples[idx]

        frame_id    = ann["frame_id"]
        group_label = ann["group_label"]
        players     = ann["players"]

        # 1. sort players left → right (required by make_subgroup_indices)
        players = sorted(players, key=lambda p: p["bbox_center_x"])

        # 2. symmetric temporal window around frame_id
        frame_ids = list(range(frame_id - self.half, frame_id + self.half + 1))

        # FIX 2: original loop accumulated [T][N] then stacked to [T, N, C, H, W]
        # which torch.stack(crops, dim=1) would incorrectly produce [T, N, C, H, W].
        # Correct target is [N, T, C, H, W] so we accumulate per-person across time.
        clip_dir = self.root / str(video_id) / str(frame_id)

        # Pre-load all T frames once to avoid re-opening the same file per player
        frames_pil = {}
        for fid in frame_ids:
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                frames_pil[fid] = Image.open(img_path).convert("RGB")
            else:
                # FIX 3: original raised FileNotFoundError on missing frames.
                # Substitute the nearest available frame instead.
                frames_pil[fid] = self._nearest_frame(clip_dir, fid, frame_ids)

        person_crops = []                               # [N] list of [T, C, H, W]
        for p in players:
            x, y, w, h = p["bbox"]
            t_crops = []
            for fid in frame_ids:
                img    = frames_pil[fid]
                iw, ih = img.size
                # Clamp bbox to image boundaries
                x1 = max(0, x);       y1 = max(0, y)
                x2 = min(iw, x + w);  y2 = min(ih, y + h)
                crop = img.crop((x1, y1, x2, y2))
                if self.transforms:
                    crop = self.transforms(crop)        # [C, H, W]
                t_crops.append(crop)
            person_crops.append(torch.stack(t_crops, dim=0))   # [T, C, H, W]

        x = torch.stack(person_crops, dim=0)            # [N, T, C, H, W]

        person_labels = torch.tensor(
            [p["action_id"] for p in players], dtype=torch.long
        )                                               # [N]

        # FIX 4: group_label was returned as a raw int.
        # Wrap in a [1] tensor so volleyball_collate can torch.cat cleanly.
        group_label_tensor = torch.tensor([group_label], dtype=torch.long)  # [1]

        assert x.shape[0] == len(person_labels), "Mismatch: N players"
        assert x.shape[1] == self.T,             "Mismatch: temporal window"

        return x, group_label_tensor, person_labels

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_annotations(self, ann_file: Path) -> list[dict]:
        """
        Parse one annotations.txt file.

        Line format:
            <frame>.jpg  <group_label>  x y w h action  x y w h action  ...
        """
        samples = []

        with open(ann_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            tokens = line.split()

            frame_str = tokens[0]
            group_str = tokens[1]

            frame_id    = int(frame_str.replace(".jpg", ""))


            group_label = self.cfg.labesls.group_activities.index(group_str)

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
        """Return the closest available frame when a specific one is missing."""
        for fid in sorted(frame_ids, key=lambda f: abs(f - missing_fid)):
            img_path = clip_dir / f"{fid}.jpg"
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        return Image.new("RGB", (720, 1280), color=0)   # blank fallback


def volleyball_collate(batch: list) -> tuple:
    """
    Custom collate for the Volleyball dataset.

    The standard DataLoader collate assumes every tensor in a batch shares
    the same shape. N (number of players) varies across samples, so we
    cannot stack along a batch dimension.

    Each sample from VolleyballDataset.__getitem__ is:
        x              [N_i, T, C, H, W]   person crops
        group_label    [1]                 team activity class index
        person_labels  [N_i]               individual action class indices

    Returns
    -------
    frames_list        : list[B] of [N_i, T, C, H, W]
        One tensor per sample. Kept as a list because N_i differs.

    group_labels       : LongTensor [B]
        Safe to stack — always exactly one label per sample.

    person_labels_list : list[B] of [N_i]
        One tensor per sample. Kept as a list because N_i differs.
    """
    frames_list        = [sample[0] for sample in batch]    # list of [N_i, T, C, H, W]
    group_labels       = torch.cat([sample[1] for sample in batch], dim=0)  # [B]
    person_labels_list = [sample[2] for sample in batch]    # list of [N_i]

    return frames_list, group_labels, person_labels_list
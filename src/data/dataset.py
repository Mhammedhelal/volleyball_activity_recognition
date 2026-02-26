from torch.utils.data import Dataset
from pathlib import Path
import torch
from PIL import Image
from labels import GROUP_ACTIONS, PERSON_ACTIONS

class VolleyballDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split_videos: set[int],
        transforms=None,
        T: int = 9,
    ):
        self.root = root
        self.transforms = transforms
        self.T = T

        self.samples = []  # ← list of (video_id, annotation)

        for video_id in split_videos:
            ann_file = root / str(video_id) / "annotations.txt"
            self.samples.extend(
                self._parse_annotations(ann_file)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, ann = self.samples[idx]

        frame_id     = ann["frame_id"]
        group_label  = ann["group_label"]
        players      = ann["players"]

        # 1. sort players left → right
        players = sorted(players, key=lambda p: p["bbox_center_x"])

        # 2. temporal window
        frames = range(frame_id - 4, frame_id + 5)

        crops = []
        for t in frames:
            img = Image.open(
                self.root / str(video_id) / str(frame_id) / f"{t}.jpg"
            ).convert("RGB")

            frame_crops = []
            for p in players:
                x, y, w, h = p["bbox"]
                crop = img.crop((x, y, x + w, y + h))
                if self.transforms:
                    crop = self.transforms(crop)
                frame_crops.append(crop)

            crops.append(torch.stack(frame_crops))  # [N, C, H, W]

        x = torch.stack(crops, dim=1)  # [N, T, C, H, W]

        person_labels = torch.tensor(
            [p["action_id"] for p in players], dtype=torch.long
        )
        
        assert x.shape[0] == len(person_labels), "Mismatch N players"
        assert x.shape[1] == self.T, "Temporal window mismatch"

        return x, person_labels, group_label
    
    def _parse_annotations(self, ann_file: Path):
        samples = []

        with open(ann_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            tokens = line.split()

            # ---- header ----
            frame_str   = tokens[0]
            group_label = tokens[1]

            frame_id = int(frame_str.replace(".jpg", ""))

            # ---- players ----
            player_tokens = tokens[2:]
            assert len(player_tokens) % 5 == 0, "Invalid annotation format"

            players = []
            for i in range(0, len(player_tokens), 5):
                x, y, w, h = map(int, player_tokens[i:i+4])
                action     = player_tokens[i+4]

                players.append({
                    "bbox": (x, y, w, h),
                    "bbox_center_x": x + w / 2,
                    "action": action,
                    "action_id": PERSON_ACTIONS.index(action),
                })

            samples.append({
                "frame_id": frame_id,
                "group_label": GROUP_ACTIONS.index(group_label),
                "players": players,
            })

        return samples
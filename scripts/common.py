"""
scripts/common.py
-----------------
Shared helper functions used by both scripts/train.py and scripts/evaluate.py.
"""

import inspect
import random
from collections import Counter
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import VolleyballDataset, make_collate_fn
from src.data.transforms import eval_transforms
from src.models.cnn_backbones import (
    build_alexnet_fc7,
    build_mobilenet_v3_large,
    build_resnet50,
)
from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.baselines import BASELINES


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def validate_config(cfg: Config) -> None:
    """
    Validate the config before any training/evaluation starts.
    Raises ValueError with a clear message on the first problem found.
    """
    errors = []

    # Label lengths
    if len(cfg.labels.group_activities) != cfg.labels.num_group_classes:
        errors.append(
            f"labels.group_activities has {len(cfg.labels.group_activities)} entries "
            f"but num_group_classes={cfg.labels.num_group_classes}"
        )
    if len(cfg.labels.person_actions) != cfg.labels.num_person_classes:
        errors.append(
            f"labels.person_actions has {len(cfg.labels.person_actions)} entries "
            f"but num_person_classes={cfg.labels.num_person_classes}"
        )

    # Pooling strategy
    if cfg.pooling.strategy not in ("max", "avg"):
        errors.append(
            f"pooling.strategy must be 'max' or 'avg', got '{cfg.pooling.strategy}'"
        )

    # Num subgroups
    if cfg.pooling.num_subgroups not in (1, 2, 4):
        errors.append(
            f"pooling.num_subgroups must be 1, 2, or 4, got {cfg.pooling.num_subgroups}"
        )

    # Temporal window must be odd
    if cfg.dataset.num_frames % 2 == 0:
        errors.append(
            f"dataset.num_frames must be odd, got {cfg.dataset.num_frames}"
        )

    # Device
    device = getattr(cfg.training, "device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        errors.append(
            "training.device='cuda' but CUDA is not available on this machine. "
            "Set training.device='cpu' or run on a GPU node."
        )

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Config validation failed:\n{msg}")

    print("✔  Config validated OK")


# ---------------------------------------------------------------------------
# Startup data sanity check
# ---------------------------------------------------------------------------

def check_data_integrity(data_root: Path, videos: List[int]) -> None:
    """
    Lightweight startup check — verifies that each expected video folder
    exists, has an annotations.txt, and contains at least one .jpg.

    Does NOT exhaustively validate every frame (that would be too slow).
    Raises FileNotFoundError with a clear message on the first problem.
    """
    print(f"\n── Data integrity check ({'─'*40})")
    missing_dirs   = []
    missing_ann    = []
    empty_dirs     = []

    for vid in sorted(videos):
        vid_dir = data_root / str(vid)
        if not vid_dir.is_dir():
            missing_dirs.append(vid)
            continue
        ann_file = vid_dir / "annotations.txt"
        if not ann_file.exists():
            missing_ann.append(vid)
            continue
        jpgs = list(vid_dir.rglob("*.jpg"))
        if not jpgs:
            empty_dirs.append(vid)

    problems = []
    if missing_dirs:
        problems.append(f"Missing video directories: {missing_dirs}")
    if missing_ann:
        problems.append(f"Missing annotations.txt in: {missing_ann}")
    if empty_dirs:
        problems.append(f"No .jpg images found in: {empty_dirs}")

    if problems:
        msg = "\n".join(f"  • {p}" for p in problems)
        raise FileNotFoundError(
            f"Data integrity check failed under {data_root}:\n{msg}\n"
            "Fix the data paths or video IDs in configs/default.yaml."
        )

    print(f"  ✔  {len(videos)} video(s) verified under {data_root}")


# ---------------------------------------------------------------------------
# Class distribution analysis
# ---------------------------------------------------------------------------

def log_class_distribution(cfg: Config, videos: List[int], split_name: str) -> None:
    """
    Parse annotations for *videos* and print class counts.
    Helps detect severe class imbalance before training starts.
    """
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent.parent / data_root

    group_counts  = Counter()
    person_counts = Counter()
    total_samples = 0

    for vid in videos:
        ann_file = data_root / str(vid) / "annotations.txt"
        if not ann_file.exists():
            continue
        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens      = line.split()
                group_label = tokens[1]
                group_counts[group_label] += 1
                total_samples += 1
                player_tokens = tokens[2:]
                for i in range(4, len(player_tokens), 5):
                    person_counts[player_tokens[i]] += 1

    if total_samples == 0:
        print(f"  ⚠  No samples found for {split_name} split")
        return

    print(f"\n── {split_name} class distribution ({total_samples} samples) ──────────")
    print(f"  {'Group Activity':<20} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*20} {'-'*6}  {'-'*6}")
    for label in cfg.labels.group_activities:
        cnt = group_counts.get(label, 0)
        pct = 100.0 * cnt / total_samples if total_samples else 0
        flag = "  ⚠ rare" if pct < 5 else ""
        print(f"  {label:<20} {cnt:>6}  {pct:>5.1f}%{flag}")

    print(f"\n  {'Person Action':<20} {'Count':>7}")
    print(f"  {'-'*20} {'-'*7}")
    total_persons = sum(person_counts.values())
    for label in cfg.labels.person_actions:
        cnt = person_counts.get(label, 0)
        pct = 100.0 * cnt / total_persons if total_persons else 0
        flag = "  ⚠ rare" if pct < 3 else ""
        print(f"  {label:<20} {cnt:>7}  {pct:>5.1f}%{flag}")


# ---------------------------------------------------------------------------
# Video resolution
# ---------------------------------------------------------------------------

def resolve_videos(data_root: Path, requested: List[int], split_name: str) -> List[int]:
    available: set = set()
    if data_root.is_dir():
        for subdir in sorted(data_root.iterdir()):
            if subdir.is_dir() and subdir.name.isdigit():
                if (subdir / "annotations.txt").exists():
                    available.add(int(subdir.name))

    if not available:
        raise FileNotFoundError(
            f"No video folders with annotations.txt found under: {data_root}\n"
            f"Expected structure: {data_root}/<video_id>/annotations.txt"
        )

    requested_set = set(requested)
    matched       = sorted(available & requested_set)
    missing       = sorted(requested_set - available)
    extra         = sorted(available - requested_set)

    print(f"\n── {split_name} videos ──────────────────────────────")
    print(f"  Requested in config : {sorted(requested_set)}")
    print(f"  Found on disk       : {sorted(available)}")
    if matched:
        print(f"  ✔ Using            : {matched}")
    if missing:
        print(f"  ✘ Missing (skipped): {missing}")
    if extra:
        print(f"  ℹ  Extra on disk   : {extra}  (not in this split)")

    if not matched:
        print(f"\n  ⚠  None of the {split_name} IDs exist on disk.")
        print(f"     Falling back to ALL available: {sorted(available)}")
        matched = sorted(available)

    print()
    return matched


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def get_backbone_fn(backbone_name: str):
    backbone_map = {
        "alexnet":            build_alexnet_fc7,
        "resnet50":           build_resnet50,
        "mobilenet_v3_large": build_mobilenet_v3_large,
    }
    fn = backbone_map.get(backbone_name)
    if fn is None:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choices: {list(backbone_map.keys())}"
        )
    return fn


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_full_model(cfg: Config) -> HierarchicalGroupActivityModel:
    feature_extractor = get_backbone_fn(cfg.cnn.backbone)
    return HierarchicalGroupActivityModel(
        feature_extractor = feature_extractor,
        lstm_hidden_p     = cfg.person_lstm.hidden_dim,
        lstm_hidden_g     = cfg.group_lstm.hidden_dim,
        person_classes    = cfg.labels.num_person_classes,
        group_classes     = cfg.labels.num_group_classes,
        n_subgroups       = cfg.pooling.num_subgroups,
        pool              = cfg.pooling.strategy,
        n_layers_p        = cfg.person_lstm.num_layers,
        n_layers_g        = cfg.group_lstm.num_layers,
    )


def build_baseline_model(
    cfg:          Config,
    baseline_key: str,
    pool:         str | None = None,
    lstm_hidden:  int | None = None,
) -> object:
    key = baseline_key.upper()
    if key not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{baseline_key}'. Choose from: {list(BASELINES.keys())}"
        )

    backbone_fn = get_backbone_fn(cfg.cnn.backbone)
    num_classes = cfg.labels.num_group_classes
    pool        = pool        or cfg.pooling.strategy
    lstm_hidden = lstm_hidden or cfg.person_lstm.hidden_dim

    cls    = BASELINES[key]
    sig    = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    kwargs: dict = {"num_classes": num_classes}
    if "backbone_fn"  in params: kwargs["backbone_fn"]  = backbone_fn
    if "pool"         in params: kwargs["pool"]          = pool
    if "lstm_hidden"  in params: kwargs["lstm_hidden"]   = lstm_hidden
    if "lstm1_hidden" in params: kwargs["lstm1_hidden"]  = lstm_hidden

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loader(
    cfg:        Config,
    videos:     List[int],
    transform,
    shuffle:    bool,
    batch_size: int,
    crops_data: bool,
) -> DataLoader:
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_root    = project_root / data_root

    dataset = VolleyballDataset(
        root         = data_root,
        split_videos = set(videos),
        cfg          = cfg,
        transforms   = transform,
        T            = cfg.dataset.num_frames,
        crops_data   = crops_data,
    )

    # Validate crops_data matches model expectation before burning GPU time
    assert isinstance(crops_data, bool), "crops_data must be a bool"

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = make_collate_fn(crops_data=crops_data),
        num_workers = cfg.dataset.num_workers,
        pin_memory  = cfg.dataset.pin_memory and torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
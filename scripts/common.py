"""
src/scripts/common.py
---------------------
Shared helper functions used by both scripts/train.py and scripts/evaluate.py.
"""

from pathlib import Path
from typing import List
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate
from src.data.transforms import eval_transforms
from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large
from src.models.baselines import BASELINES
from src.engine.trainer import Trainer



def resolve_videos(data_root: Path, requested: List[int], split_name: str) -> List[int]:
    available: set[int] = set()
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


def get_backbone_fn(backbone_name: str):
    backbone_map = {
        "alexnet":            build_alexnet_fc7,
        "resnet50":           build_resnet50,
        "mobilenet_v3_large": build_mobilenet_v3_large,
    }
    return backbone_map.get(backbone_name, build_alexnet_fc7)


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


def build_baseline_model(cfg: Config, baseline_key: str, **run_args):
    key = baseline_key.upper()
    if key not in BASELINES:
        raise ValueError(f"Unknown baseline '{baseline_key}'. Choose from: {list(BASELINES.keys())}")

    backbone_fn = get_backbone_fn(cfg.cnn.backbone)
    num_classes = cfg.labels.num_group_classes
    pool        = run_args.get("pool") or cfg.pooling.strategy
    lstm_hidden = run_args.get("lstm_hidden") or cfg.person_lstm.hidden_dim

    cls = BASELINES[key]
    import inspect
    sig    = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    kwargs = {"num_classes": num_classes}
    if "backbone_fn" in params: kwargs["backbone_fn"] = backbone_fn
    if "pool" in params: kwargs["pool"] = pool
    if "lstm_hidden" in params: kwargs["lstm_hidden"] = lstm_hidden
    if "lstm1_hidden" in params: kwargs["lstm1_hidden"] = lstm_hidden

    return cls(**kwargs)


def build_loader(cfg: Config, videos: list[int], transform, shuffle: bool, batch_size: int, crops_data) -> DataLoader:
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
        crops_data=crops_data
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = volleyball_collate(crops_data=crops_data),
        num_workers = cfg.dataset.num_workers,
        pin_memory  = cfg.dataset.pin_memory,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

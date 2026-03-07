"""
scripts/evaluate.py
-------------------
Evaluation entrypoint for the hierarchical group activity model.

Loads a saved checkpoint, runs inference on the test (or val) split,
and prints a full per-class accuracy report.

Usage
-----
    # Evaluate on test split with a specific checkpoint
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final_model.pt

    # Evaluate on val split instead
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final_model.pt --split val

    # Override device
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final_model.pt --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate
from src.data.transforms import eval_transforms
from src.engine.evaluator import Evaluator
from src.models.hierarchical_model import HierarchicalGroupActivityModel


# ---------------------------------------------
# Helpers  (duplicated from train.py intentionally —
#           evaluate.py must be runnable standalone)
# ---------------------------------------------

def build_model(cfg: Config) -> HierarchicalGroupActivityModel:
    return HierarchicalGroupActivityModel(
        cnn_output_size = cfg.cnn.feature_dim,
        lstm_hidden_p   = cfg.person_lstm.hidden_dim,
        lstm_hidden_g   = cfg.group_lstm.hidden_dim,
        person_classes  = cfg.labels.num_person_classes,
        group_classes   = cfg.labels.num_group_classes,
        n_subgroups     = cfg.pooling.num_subgroups,
        pool            = cfg.pooling.strategy,
        n_layers_p      = cfg.person_lstm.num_layers,
        n_layers_g      = cfg.group_lstm.num_layers,
    )


def build_loader(
    cfg:        Config,
    videos:     list[int],
    batch_size: int,
) -> DataLoader:
    dataset = VolleyballDataset(
        root         = Path(cfg.paths.data_root),
        split_videos = set(videos),
        cfg = cfg,
        transforms   = eval_transforms,      # no augmentation at eval time
        T            = cfg.dataset.num_frames,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,                 # deterministic order for eval
        collate_fn  = volleyball_collate,
        num_workers = cfg.dataset.num_workers,
        pin_memory  = cfg.dataset.pin_memory,
    )


# ---------------------------------------------
# CLI
# ---------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the hierarchical group activity model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file used during training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda | cpu  (overrides config)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size  (overrides config)",
    )
    return parser.parse_args()


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Config ---------------------------------------------
    cfg = Config.from_yaml(args.config)

    overrides: dict = {}
    if args.device     is not None: overrides.setdefault("training", {})["device"]        = args.device
    if args.batch_size is not None: overrides.setdefault("evaluation", {})["batch_size"]  = args.batch_size
    if overrides:
        cfg.merge(overrides)

    device = cfg.training.device

    # -- Checkpoint ---------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -- Model ---------------------------------------------
    model = build_model(cfg)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    print(f"Loaded checkpoint: {ckpt_path}")

    # -- Data ---------------------------------------------
    videos = (
        cfg.dataset.val_videos
        if args.split == "val"
        else cfg.dataset.test_videos
    )
    loader = build_loader(cfg, videos, batch_size=cfg.evaluation.batch_size)
    print(f"Evaluating on '{args.split}' split  ({len(videos)} videos)")

    # -- Evaluate ---------------------------------------------
    evaluator = Evaluator(model, loader, cfg, device=device)
    evaluator.report()


if __name__ == "__main__":
    main()
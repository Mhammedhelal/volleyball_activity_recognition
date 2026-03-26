"""
scripts/evaluate.py
--------------------
Evaluation entrypoint for both the full hierarchical model and all baselines.

Usage
-----
    # Full hierarchical model
    python scripts/evaluate.py --checkpoint outputs/checkpoints/model_stage2.pt

    # Baseline
    python scripts/evaluate.py --checkpoint outputs/checkpoints/model_B4.pt --baseline B4

    # Specify split
    python scripts/evaluate.py --checkpoint ... --baseline B7 --split val
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
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large
from src.models.baselines import BASELINES
from scripts.common import resolve_videos, get_backbone_fn, build_full_model, build_baseline_model, build_loader, set_seed


# ---------------------------------------------
# Helpers
# ---------------------------------------------
# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model or baseline checkpoint"
    )
    parser.add_argument("--config",     type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument(
        "--baseline", type=str, default=None,
        metavar="KEY",
        help=(
            f"Evaluate a baseline checkpoint. Choices: {list(BASELINES.keys())}. "
            "Omit to evaluate the full hierarchical model."
        ),
    )
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    cfg = Config.from_yaml(config_path)

    overrides: dict = {}
    if args.device     is not None: overrides.setdefault("training",    {})["device"]     = args.device
    if args.batch_size is not None: overrides.setdefault("evaluation",  {})["batch_size"] = args.batch_size
    if overrides:
        cfg.merge(overrides)

    device = cfg.training.device

    # ── checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ── model ─────────────────────────────────────────────────────────────
    if args.baseline is not None:
        model = build_baseline_model(cfg, args.baseline)
        print(f"Evaluating baseline {args.baseline.upper()}  "
              f"(INPUT_TYPE={model.INPUT_TYPE})")
    else:
        model = build_full_model(cfg)
        print("Evaluating full hierarchical model")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)        # handle both wrapped and raw state dicts
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── data ──────────────────────────────────────────────────────────────
    # ── resolve which video IDs actually exist on disk ────────────────────
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent.parent / data_root

    raw_videos = cfg.dataset.val_videos if args.split == "val" else cfg.dataset.test_videos
    videos     = resolve_videos(data_root, raw_videos, args.split.upper())

    loader = build_loader(cfg, videos, cfg.training.stage1.batch_size)

    # ── evaluate ──────────────────────────────────────────────────────────
    evaluator = Evaluator(model, loader, cfg, device=device)
    evaluator.report()


if __name__ == "__main__":
    main()
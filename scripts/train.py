"""
scripts/train.py
----------------
Training entrypoint for both the full hierarchical model and all baselines.

Usage
-----
    # Full hierarchical model (default)
    python scripts/train.py

    # Train a specific baseline
    python scripts/train.py --baseline B1
    python scripts/train.py --baseline B7 --pool avg --lstm_hidden 1024

    # Standard overrides still apply
    python scripts/train.py --baseline B4 --lr 1e-4 --num_epochs 50 --device cuda

    Video Data and Selection
    ------------------------
    The script expects video data organized in folders by video ID, each containing:
    - A folder named with the video ID (e.g., '45', '46')
    - Inside each video folder: an 'annotations.txt' file with frame annotations
    - Frame images stored in subfolders or as individual files

    Video selection:
    - By default, uses video IDs specified in the config file (configs/default.yaml)
    - With --data-root: auto-discovers all video IDs from subfolder names in the data directory
    - Auto-split: 80% train, 10% val, 10% test (when using --data-root)
    - Only videos with existing 'annotations.txt' are included
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate
from src.data.transforms import train_transforms, eval_transforms
from src.engine.trainer import Trainer
from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.cnn_backbones import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large
from src.models.baselines import BASELINES
from src.utils.checkpointing import save_checkpoint
from scripts.common import resolve_videos, get_backbone_fn, build_full_model, build_baseline_model, build_loader, set_seed


# ---------------------------------------------
# Helpers
# ---------------------------------------------
# ── helpers ───────────────────────────────────────────────────────────────────

# Shared helpers imported from src.scripts.common

def build_trainer(
    cfg:    Config,
    model,
    loader: DataLoader,
    stage:  int = 1,
) -> Trainer:
    """
    For the full hierarchical model: stage-aware (1 or 2).
    For baselines: stage=1 always (all params trained together).
    """
    from src.models.baselines.base import BaselineModel

    if isinstance(model, BaselineModel):
        # Baselines: train all parameters in one stage
        return Trainer(
            model         = model,
            params        = model.parameters(),
            train_loader  = loader,
            device        = cfg.training.device,
            learning_rate = cfg.training.stage1.lr,
            momentum      = cfg.training.stage1.momentum,
            num_epochs    = cfg.training.stage1.epochs,
            person_loss_w = cfg.loss.person_weight,
        )

    # Full hierarchical model: stage-aware parameter selection
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    if stage == 1:
        trainable_params = list(model.person_embedder.parameters())
    else:
        trainable_params = (
            list(model.subgroup_pooler.parameters()) +
            list(model.frame_descriptor.parameters())
        )

    return Trainer(
        model         = model,
        params        = trainable_params,
        train_loader  = loader,
        device        = cfg.training.device,
        learning_rate = stage_cfg.lr,
        momentum      = stage_cfg.momentum,
        num_epochs    = stage_cfg.epochs,
        person_loss_w = cfg.loss.person_weight,
    )


# ---------------------------------------------
# Stage runners (full model only)
# ---------------------------------------------

def run_stage1(cfg, model, train_loader, ckpt_dir, model_name="model") -> Path:
    print("\n" + "=" * 70)
    print("STAGE 1  —  CNN + LSTM1  (person-action supervision)")
    print("=" * 70)

    build_trainer(cfg, model, train_loader, stage=1).train()

    ckpt_path = ckpt_dir / f"{model_name}_stage1.pt"
    save_checkpoint({"stage": 1, "model": model.state_dict()}, str(ckpt_path))
    print(f"Stage 1 checkpoint saved to: {ckpt_path}")
    return ckpt_path


def run_stage2(cfg, model, train_loader, ckpt_dir, model_name="model") -> Path:
    print("\n" + "=" * 70)
    print("STAGE 2  —  LSTM2  (group-activity supervision)")
    print("=" * 70)

    build_trainer(cfg, model, train_loader, stage=2).train()

    ckpt_path = ckpt_dir / f"{model_name}_stage2.pt"
    save_checkpoint({"stage": 2, "model": model.state_dict()}, str(ckpt_path))
    print(f"Stage 2 checkpoint saved to: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------
# CLI
# ---------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical group activity model or a baseline"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    # ── model selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--baseline", type=str, default=None,
        metavar="KEY",
        help=(
            "Train a baseline instead of the full hierarchical model. "
            f"Choices: {list(BASELINES.keys())}. "
            "Omit to train the full two-stage model."
        ),
    )

    # ── baseline hyper-params (ignored for full model) ─────────────────────
    parser.add_argument("--pool",        type=str,   default=None,
                        help="Pooling strategy for baselines: max | avg")
    parser.add_argument("--lstm_hidden", type=int,   default=None,
                        help="LSTM hidden size for temporal baselines (B4/B5/B6/B7)")

    # ── full-model stage selection ─────────────────────────────────────────
    parser.add_argument("--stage",             type=int, default=None, choices=[1, 2])
    parser.add_argument("--stage1_checkpoint", type=str, default=None)

    # ── data / output ──────────────────────────────────────────────────────
    parser.add_argument("--data-root",   type=str, default=None)
    parser.add_argument("--model-name",  type=str, default="model")

    # ── shared hyper-params ────────────────────────────────────────────────
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--num_subgroups", type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    # ── config ────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    cfg = Config.from_yaml(config_path)

    overrides: dict = {}
    if args.device        is not None: overrides.setdefault("training",  {})["device"]        = args.device
    if args.batch_size    is not None: overrides.setdefault("training",  {}).setdefault("stage1", {})["batch_size"] = args.batch_size
    if args.num_epochs    is not None: overrides.setdefault("training",  {}).setdefault("stage1", {})["epochs"]     = args.num_epochs
    if args.lr            is not None: overrides.setdefault("training",  {}).setdefault("stage1", {})["lr"]         = args.lr
    if args.num_subgroups is not None: overrides.setdefault("pooling",   {})["num_subgroups"]  = args.num_subgroups
    if args.data_root     is not None: overrides.setdefault("paths",     {})["data_root"]      = args.data_root
    if overrides:
        cfg.merge(overrides)

    set_seed(cfg.training.seed)
    device   = cfg.training.device
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve which video IDs actually exist on disk ────────────────────
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent.parent / data_root

    train_videos = resolve_videos(data_root, cfg.dataset.train_videos, "TRAIN")

    # ── data loaders ──────────────────────────────────────────────────────
    crops_data = True if args.baseline in ('B2', 'B3', 'B5', 'B6', 'B7', None) else False
    train_loader = build_loader(
        cfg, train_videos, train_transforms,
        shuffle=True, batch_size=cfg.training.stage1.batch_size,crops_data=True
    )

    # ── model ─────────────────────────────────────────────────────────────
    if args.baseline is not None:
        # ── BASELINE path ─────────────────────────────────────────────────
        model      = build_baseline_model(cfg, args)
        model_name = f"{args.model_name}_{args.baseline.upper()}"
        print(f"\nTraining baseline {args.baseline.upper()}  "
              f"(INPUT_TYPE={model.INPUT_TYPE})")

        build_trainer(cfg, model, train_loader).train()

        ckpt_path = ckpt_dir / f"{model_name}.pt"
        save_checkpoint({"model": model.state_dict(), "baseline": args.baseline},
                        str(ckpt_path))
        print(f"Checkpoint saved to: {ckpt_path}")

    else:
        # ── FULL MODEL path ───────────────────────────────────────────────
        model = build_full_model(cfg)

        if args.stage == 1:
            run_stage1(cfg, model, train_loader, ckpt_dir, args.model_name)
        elif args.stage == 2:
            if args.stage1_checkpoint:
                ckpt = torch.load(args.stage1_checkpoint, map_location=device)
                model.load_state_dict(ckpt["model"])
                print(f"Loaded stage 1 weights from: {args.stage1_checkpoint}")
            run_stage2(cfg, model, train_loader, ckpt_dir, args.model_name)
        else:
            # Both stages sequentially
            s1_path = run_stage1(cfg, model, train_loader, ckpt_dir, args.model_name)
            run_stage2(cfg, model, train_loader, ckpt_dir, args.model_name)


if __name__ == "__main__":
    main()
"""
scripts/train.py
----------------
Training entrypoint for the hierarchical group activity model.

Usage
-----
    # Default config
    python scripts/train.py

    # Override specific values
    python scripts/train.py --lr 3e-5 --batch_size 16 --device cpu
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate
from src.data.transforms import eval_transforms, train_transforms
from src.engine.trainer import Trainer
from src.models.hierarchical_model import HierarchicalGroupActivityModel


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> HierarchicalGroupActivityModel:
    return HierarchicalGroupActivityModel(
        cnn_output_size = cfg.cnn.feature_dim,             # 4096
        lstm_hidden_p   = cfg.person_lstm.hidden_dim,      # 3000
        lstm_hidden_g   = cfg.group_lstm.hidden_dim,       # 2000
        person_classes  = cfg.labels.num_person_classes,   # 9
        group_classes   = cfg.labels.num_group_classes,    # 8
        n_subgroups     = cfg.pooling.num_subgroups,       # 2
        pool            = cfg.pooling.strategy,            # "max"
        n_layers_p      = cfg.person_lstm.num_layers,      # 1
        n_layers_g      = cfg.group_lstm.num_layers,       # 1
    )


def build_loader(
    cfg:        Config,
    videos:     list[int],
    transform,
    shuffle:    bool,
    batch_size: int,
) -> DataLoader:
    dataset = VolleyballDataset(
        root         = Path(cfg.paths.data_root),
        split_videos = set(videos),
        cfg = cfg,
        transforms   = transform,
        T            = cfg.dataset.num_frames,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = volleyball_collate,
        num_workers = cfg.dataset.num_workers,
        pin_memory  = cfg.dataset.pin_memory,
    )


def build_trainer(
    cfg:    Config,
    model:  HierarchicalGroupActivityModel,
    loader: DataLoader,
    stage:  int,                           # 1 or 2
) -> Trainer:
    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    return Trainer(
        model         = model,
        train_loader  = loader,
        device        = cfg.training.device,
        learning_rate = stage_cfg.lr,
        momentum      = stage_cfg.momentum,
        num_epochs    = stage_cfg.epochs,
        person_loss_w = cfg.loss.person_weight,
    )


# ---------------------------------------------
# CLI
# ---------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical group activity model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--lr",            type=float, default=None, help="Learning rate (both stages)")
    parser.add_argument("--batch_size",    type=int,   default=None, help="Batch size (both stages)")
    parser.add_argument("--device",        type=str,   default=None, help="cuda | cpu")
    parser.add_argument("--num_epochs",    type=int,   default=None, help="Epochs per stage")
    parser.add_argument("--num_subgroups", type=int,   default=None, help="1 | 2 | 4")
    return parser.parse_args()


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Config ---------------------------------------------
    cfg = Config.from_yaml(args.config)

    overrides: dict = {}
    if args.lr is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["lr"] = args.lr
        overrides.setdefault("training", {}).setdefault("stage2", {})["lr"] = args.lr
    if args.batch_size is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["batch_size"] = args.batch_size
        overrides.setdefault("training", {}).setdefault("stage2", {})["batch_size"] = args.batch_size
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.num_epochs is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["epochs"] = args.num_epochs
        overrides.setdefault("training", {}).setdefault("stage2", {})["epochs"] = args.num_epochs
    if args.num_subgroups is not None:
        overrides.setdefault("pooling", {})["num_subgroups"] = args.num_subgroups
    if overrides:
        cfg.merge(overrides)

    # -- Reproducibility ---------------------------------------------
    set_seed(cfg.training.seed)

    # -- Persist run config ---------------------------------------------
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.paths.output_dir) / "run_config.yaml")
    print(cfg)

    # -- Data ---------------------------------------------
    train_loader = build_loader(
        cfg,
        videos     = cfg.dataset.train_videos,
        transform  = train_transforms,
        shuffle    = True,
        batch_size = cfg.training.stage1.batch_size,
    )

    # -- Model ---------------------------------------------
    model = build_model(cfg)

    # -- Stage 1 ---------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1  —  CNN + LSTM1  (person-action supervision)")
    print("=" * 70)
    build_trainer(cfg, model, train_loader, stage=1).train()

    # Freeze person_embedder before stage 2
    for param in model.person_embedder.parameters():
        param.requires_grad = False

    # -- Stage 2 ---------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2  —  LSTM2  (group-activity supervision)")
    print("=" * 70)
    build_trainer(cfg, model, train_loader, stage=2).train()

    # -- Save checkpoint ---------------------------------------------
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.paths.checkpoint_dir) / "final_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")
    print(f"Run evaluate.py with --checkpoint {ckpt_path} to evaluate.")


if __name__ == "__main__":
    main()
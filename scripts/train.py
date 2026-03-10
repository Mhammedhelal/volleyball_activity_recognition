"""
scripts/train.py
----------------
Training entrypoint for the hierarchical group activity model.

Usage
-----
    # Run both stages sequentially (default)
    python scripts/train.py

    # Run Stage 1 only  (CNN + LSTM1, person-action supervision)
    python scripts/train.py --stage 1

    # Run Stage 2 only  (LSTM2, group-activity supervision)
    # Requires a Stage 1 checkpoint to load from
    python scripts/train.py --stage 2 --stage1_checkpoint models/stage1.pt

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import VolleyballDataset, volleyball_collate
from src.data.transforms import eval_transforms, train_transforms
from src.engine.trainer import Trainer
from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.person_embedder import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large
from src.utils.checkpointing import save_checkpoint, load_checkpoint, get_latest_checkpoint


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_video_ids(data_root: Path) -> list[int]:
    """
    Auto-discover video IDs by scanning subdirectories in data_root.
    
    Returns a sorted list of integer video IDs found as folder names.
    Each folder should contain an 'annotations.txt' file.
    
    Example:
        data_root/45/annotations.txt  →  discovers video ID 45
        data_root/46/annotations.txt  →  discovers video ID 46
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    video_ids = []
    for folder in data_root.iterdir():
        if folder.is_dir():
            try:
                video_id = int(folder.name)
                ann_file = folder / "annotations.txt"
                if ann_file.exists():
                    video_ids.append(video_id)
                else:
                    print(f"  Warning: {folder.name}/ has no annotations.txt, skipping")
            except ValueError:
                # Folder name is not an integer, skip
                continue
    
    return sorted(video_ids)


def build_model(cfg: Config) -> HierarchicalGroupActivityModel:
    if cfg.cnn.backbone == "alexnet":
        feature_extractor = build_alexnet_fc7
    elif cfg.cnn.backbone == "resnet50":
        feature_extractor = build_resnet50
    elif cfg.cnn.backbone == "mobilenet_v3_large":
        feature_extractor = build_mobilenet_v3_large
    else:
        raise ValueError(f"Unknown backbone: {cfg.cnn.backbone}")

    return HierarchicalGroupActivityModel(
        feature_extractor = feature_extractor,
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
    # Resolve data_root path relative to project root if it's relative
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / data_root
    
    dataset = VolleyballDataset(
        root         = data_root,
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

    if stage == 1:
        # CNN + LSTM1 only — person-level supervision
        trainable_params = list(model.person_embedder.parameters())
    else:
        # SubGroupPooler + FrameDescriptor (LSTM2 + classifier) — group-level supervision
        # person_embedder weights are intentionally excluded: frozen from Stage 1
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

    # --- Stage selection ---------------------------------------------------
    parser.add_argument(
        "--stage", type=int, default=None, choices=[1, 2],
        help=(
            "Which training stage to run.\n"
            "  1 → CNN + LSTM1  (person-action supervision)\n"
            "  2 → LSTM2        (group-activity supervision)\n"
            "Omit to run both stages sequentially (default)."
        ),
    )
    parser.add_argument(
        "--stage1_checkpoint", type=str, default=None,
        help=(
            "Path to a Stage 1 checkpoint (.pt).  "
            "Required when --stage 2 is used standalone; "
            "ignored when --stage 1 or both stages are run."
        ),
    )

    # --- Data paths -------------------------------------------------------
    parser.add_argument(
        "--data-root", type=str, default=None,
        help=(
            "Path to video data folder (e.g., 'data/videos_g10' or '/absolute/path/to/videos'). "
            "If provided, auto-discovers video IDs from subfolders. "
            "Overrides cfg.paths.data_root."
        ),
    )
    parser.add_argument(
        "--model-name", type=str, default="model",
        help="Name prefix for saved checkpoints (default: 'model'). Useful for tracking different models.",
    )

    # --- Hyper-parameter overrides ----------------------------------------
    parser.add_argument("--lr",            type=float, default=None, help="Learning rate (both stages)")
    parser.add_argument("--batch_size",    type=int,   default=None, help="Batch size (both stages)")
    parser.add_argument("--device",        type=str,   default=None, help="cuda | cpu")
    parser.add_argument("--num_epochs",    type=int,   default=None, help="Epochs per stage")
    parser.add_argument("--num_subgroups", type=int,   default=None, help="1 | 2 | 4")
    return parser.parse_args()


# ---------------------------------------------
# Stage runners
# ---------------------------------------------

def run_stage1(
    cfg:          Config,
    model:        HierarchicalGroupActivityModel,
    train_loader: DataLoader,
    ckpt_dir:     Path,
    model_name:   str = "model",
) -> Path:
    """Train Stage 1 (CNN + LSTM1) and save a checkpoint.  Returns checkpoint path."""
    print("\n" + "=" * 70)
    print("STAGE 1  —  CNN + LSTM1  (person-action supervision)")
    print("=" * 70)

    build_trainer(cfg, model, train_loader, stage=1).train()

    ckpt_path = ckpt_dir / f"{model_name}_stage1.pt"
    save_checkpoint(
        state    = {"stage": 1, "model": model.state_dict()},
        filepath = str(ckpt_path),
    )
    print(f"Stage 1 checkpoint saved to: {ckpt_path}")
    return ckpt_path


def run_stage2(
    cfg:          Config,
    model:        HierarchicalGroupActivityModel,
    train_loader: DataLoader,
    ckpt_dir:     Path,
    model_name:   str = "model",
) -> Path:
    """Freeze person_embedder, train Stage 2 (LSTM2), and save final checkpoint."""
    print("\n" + "=" * 70)
    print("STAGE 2  —  LSTM2  (group-activity supervision)")
    print("=" * 70)

    # person_embedder is excluded from the optimizer in build_trainer — no
    # requires_grad manipulation needed. The optimizer only holds
    # subgroup_pooler + frame_descriptor params, so person_embedder weights
    # are never touched during Stage 2 backprop.
    build_trainer(cfg, model, train_loader, stage=2).train()

    ckpt_path = ckpt_dir / f"{model_name}_final.pt"
    save_checkpoint(
        state    = {"stage": 2, "model": model.state_dict()},
        filepath = str(ckpt_path),
    )
    print(f"Final model checkpoint saved to: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Config - resolve relative paths from project root -----
    # If config path is relative, resolve it from the project root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Get project root (parent of scripts dir)
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / config_path
    
    cfg = Config.from_yaml(config_path)

    # -- Handle --data-root argument -----
    # If provided, update cfg.paths.data_root and auto-discover video IDs
    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            data_root = project_root / data_root
        
        print(f"Using data root: {data_root}")
        cfg.paths.data_root = str(data_root)
        
        # Auto-discover video IDs
        print(f"Discovering video IDs from {data_root}...")
        all_videos = discover_video_ids(data_root)
        print(f"  Found videos: {all_videos}")
        
        if not all_videos:
            raise ValueError(f"No video folders found in {data_root}")
        
        # Use 80% train, 10% val, 10% test split
        num_train = max(1, len(all_videos) * 80 // 100)
        num_val = max(1, len(all_videos) * 10 // 100)
        # remaining are test
        cfg.dataset.train_videos = all_videos[:num_train]
        cfg.dataset.val_videos = all_videos[num_train:num_train+num_val]
        cfg.dataset.test_videos = all_videos[num_train+num_val:]
        print(f"  Train: {cfg.dataset.train_videos}")
        print(f"  Val:   {cfg.dataset.val_videos}")
        print(f"  Test:  {cfg.dataset.test_videos}")

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

    # -- Checkpoint directory (project_root/models) --------------------
    ckpt_dir = Path("models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------
    run = args.stage  # None → both, 1 → stage 1 only, 2 → stage 2 only

    if run == 1:
        # ── Stage 1 only ──────────────────────────────────────────────
        run_stage1(cfg, model, train_loader, ckpt_dir, model_name=args.model_name)
        print(f"\nStage 1 complete.  Resume with:")
        print(f"  python scripts/train.py --stage 2 --stage1_checkpoint {ckpt_dir / f'{args.model_name}_stage1.pt'}")

    elif run == 2:
        # ── Stage 2 only ──────────────────────────────────────────────
        s1_ckpt = Path(args.stage1_checkpoint) if args.stage1_checkpoint else ckpt_dir / f"{args.model_name}_stage1.pt"
        load_checkpoint(filepath=str(s1_ckpt), model=model)

        final_path = run_stage2(cfg, model, train_loader, ckpt_dir, model_name=args.model_name)
        print(f"\nRun evaluate.py with --checkpoint {final_path} to evaluate.")

    else:
        # ── Both stages sequentially (original behaviour) ─────────────
        run_stage1(cfg, model, train_loader, ckpt_dir, model_name=args.model_name)
        final_path = run_stage2(cfg, model, train_loader, ckpt_dir, model_name=args.model_name)
        print(f"\nRun evaluate.py with --checkpoint {final_path} to evaluate.")


if __name__ == "__main__":
    main()
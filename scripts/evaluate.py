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


# ---------------------------------------------
# Helpers
# ---------------------------------------------
# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_videos(data_root: Path, requested: list[int], split_name: str) -> list[int]:
    """
    Return the subset of `requested` video IDs that actually exist on disk
    (i.e. {data_root}/{id}/annotations.txt is present).

    Prints a clear summary so the user knows what was found vs. missing.
    Falls back to ALL discovered videos (with annotations) when none of the
    requested IDs are found — useful when you only have a small subset of the
    full dataset.
    """
    # Discover every numeric subdirectory that has annotations.txt
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

def build_full_model(cfg: Config) -> HierarchicalGroupActivityModel:
    backbone_map = {
        "alexnet":            build_alexnet_fc7,
        "resnet50":           build_resnet50,
        "mobilenet_v3_large": build_mobilenet_v3_large,
    }
    feature_extractor = backbone_map.get(cfg.cnn.backbone, build_alexnet_fc7)
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


def build_baseline_model(cfg: Config, baseline_key: str):
    """Instantiate a baseline with default hyper-params from config."""
    key = baseline_key.upper()
    if key not in BASELINES:
        raise ValueError(
            f"Unknown baseline '{baseline_key}'. "
            f"Choose from: {list(BASELINES.keys())}"
        )

    backbone_map = {
        "alexnet":            build_alexnet_fc7,
        "resnet50":           build_resnet50,
        "mobilenet_v3_large": build_mobilenet_v3_large,
    }
    backbone_fn = backbone_map.get(cfg.cnn.backbone, build_alexnet_fc7)
    num_classes = cfg.labels.num_group_classes
    pool        = cfg.pooling.strategy
    lstm_hidden = cfg.person_lstm.hidden_dim

    cls = BASELINES[key]

    import inspect
    sig    = inspect.signature(cls.__init__)
    params = set(sig.parameters.keys()) - {"self"}

    kwargs: dict = {"num_classes": num_classes}
    if "backbone_fn"  in params: kwargs["backbone_fn"]  = backbone_fn
    if "pool"         in params: kwargs["pool"]          = pool
    if "lstm_hidden"  in params: kwargs["lstm_hidden"]   = lstm_hidden
    if "lstm1_hidden" in params: kwargs["lstm1_hidden"]  = lstm_hidden

    return cls(**kwargs)


def build_loader(cfg: Config, videos: list[int], batch_size: int) -> DataLoader:
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_root    = project_root / data_root

    dataset = VolleyballDataset(
        root         = data_root,
        split_videos = set(videos),
        cfg          = cfg,
        transforms   = eval_transforms,
        T            = cfg.dataset.num_frames,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = volleyball_collate,
        num_workers = cfg.dataset.num_workers,
        pin_memory  = cfg.dataset.pin_memory,
    )


# ---------------------------------------------
# CLI
# ---------------------------------------------

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
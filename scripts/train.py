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
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.transforms import train_transforms, eval_transforms
from src.engine.trainer import Trainer
from src.models.baselines import BASELINES
from src.models.baselines.base import BaselineModel
from src.utils.checkpointing import save_checkpoint
from src.utils.embedding_cache import cache_person_embeddings, build_embedding_loader
from scripts.common import (
    build_baseline_model,
    build_full_model,
    build_loader,
    check_data_integrity,
    get_backbone_fn,
    log_class_distribution,
    resolve_videos,
    set_seed,
    validate_config,
)


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------

def build_trainer(
    cfg:        Config,
    model,
    loader,
    val_loader  = None,
    stage:      int = 1,
    model_name: str = "model",
    use_precomputed_embeddings=False,
) -> Trainer:
    if isinstance(model, BaselineModel):
        return Trainer(
            model          = model,
            params         = model.parameters(),
            train_loader   = loader,
            val_loader     = val_loader,
            cfg            = cfg,
            device         = cfg.training.device,
            learning_rate  = cfg.training.stage1.lr,
            momentum       = cfg.training.stage1.momentum,
            num_epochs     = cfg.training.stage1.epochs,
            person_loss_w  = cfg.loss.person_weight,
            grad_clip      = cfg.training.stage1.grad_clip,
            model_name     = model_name,
            stage          = stage,
        )

    stage_cfg = cfg.training.stage1 if stage == 1 else cfg.training.stage2
    if stage == 1:
        trainable_params = list(model.person_embedder.parameters())
    else:
        # Stage 2: freeze person_embedder completely so no wasted backward pass
        for param in model.person_embedder.parameters():
            param.requires_grad = False
        trainable_params = (
            list(model.subgroup_pooler.parameters()) +
            list(model.frame_descriptor.parameters())
        )

    return Trainer(
        model          = model,
        params         = trainable_params,
        train_loader   = loader,
        val_loader     = val_loader,
        cfg            = cfg,
        device         = cfg.training.device,
        learning_rate  = stage_cfg.lr,
        momentum       = stage_cfg.momentum,
        num_epochs     = stage_cfg.epochs,
        person_loss_w  = cfg.loss.person_weight,
        grad_clip      = stage_cfg.grad_clip,
        model_name     = model_name,
        stage          = stage,
        use_precomputed_embeddings=use_precomputed_embeddings,
    )


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage1(cfg, model, train_loader, val_loader, ckpt_dir, model_name="model") -> Path:
    print("\n" + "=" * 70)
    print("STAGE 1  —  CNN + LSTM1  (person-action supervision)")
    print("=" * 70)
    build_trainer(cfg, model, train_loader, val_loader, stage=1, model_name=model_name).train()
    # The trainer already saves best + final checkpoints.
    # Also save a named stage1 checkpoint for stage2 loading:
    ckpt_path = ckpt_dir / f"{model_name}_stage1.pt"
    save_checkpoint({"stage": 1, "model": model.state_dict()}, str(ckpt_path))
    return ckpt_path


def run_stage2(cfg, model, train_loader, val_loader, ckpt_dir, model_name="model", use_precomputed_embeddings=False) -> Path:
    print("\n" + "=" * 70)
    print("STAGE 2  —  LSTM2  (group-activity supervision)")
    print("=" * 70)
    build_trainer(cfg, model, train_loader, val_loader, stage=2, model_name=model_name, use_precomputed_embeddings=use_precomputed_embeddings).train()
    ckpt_path = ckpt_dir / f"{model_name}_stage2.pt"
    save_checkpoint({"stage": 2, "model": model.state_dict()}, str(ckpt_path))
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical group activity model or a baseline"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    # model selection
    parser.add_argument(
        "--baseline", type=str, default=None, metavar="KEY",
        help=f"Train a baseline. Choices: {list(BASELINES.keys())}.",
    )

    # baseline hyper-params
    parser.add_argument("--pool",        type=str, default=None)
    parser.add_argument("--lstm_hidden", type=int, default=None)

    # full-model stage selection
    parser.add_argument("--stage",             type=int, default=None, choices=[1, 2])
    parser.add_argument("--stage1_checkpoint", type=str, default=None)

    # data / output
    parser.add_argument("--data-root",  type=str, default=None)
    parser.add_argument("--model-name", type=str, default="model")

    # shared hyper-params
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--num_epochs",    type=int,   default=None)
    parser.add_argument("--num_subgroups", type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── config ────────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    cfg = Config.from_yaml(config_path)

    # ── apply CLI overrides ───────────────────────────────────────────────
    overrides: dict = {}
    if args.device        is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.batch_size    is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["batch_size"] = args.batch_size
        overrides.setdefault("training", {}).setdefault("stage2", {})["batch_size"] = args.batch_size
    if args.num_epochs    is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["epochs"] = args.num_epochs
        overrides.setdefault("training", {}).setdefault("stage2", {})["epochs"] = args.num_epochs
    if args.lr            is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["lr"] = args.lr
        overrides.setdefault("training", {}).setdefault("stage2", {})["lr"] = args.lr
    if args.num_subgroups is not None:
        overrides.setdefault("pooling", {})["num_subgroups"] = args.num_subgroups
    if args.data_root     is not None:
        overrides.setdefault("paths", {})["data_root"] = args.data_root
    if overrides:
        cfg.merge(overrides)

    # ── auto-detect device ────────────────────────────────────────────────
    if not hasattr(cfg.training, "device") or cfg.training.device == "cuda":
        if not torch.cuda.is_available():
            cfg.merge({"training": {"device": "cpu"}})
            print("⚠  CUDA not available — falling back to CPU")

    # ── validate config ───────────────────────────────────────────────────
    validate_config(cfg)

    set_seed(cfg.training.seed)
    device   = cfg.training.device
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve & validate data ───────────────────────────────────────────
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = Path(__file__).resolve().parent.parent / data_root

    train_videos = resolve_videos(data_root, cfg.dataset.train_videos, "TRAIN")
    val_videos   = resolve_videos(data_root, cfg.dataset.val_videos,   "VAL")

    check_data_integrity(data_root, train_videos + val_videos)
    log_class_distribution(cfg, train_videos, "TRAIN")

    # ── model ─────────────────────────────────────────────────────────────
    if args.baseline is not None:
        crops_data = args.baseline.upper() not in ("B1", "B4")
        model      = build_baseline_model(
            cfg, baseline_key=args.baseline,
            pool=args.pool, lstm_hidden=args.lstm_hidden,
        )
        model_name = f"{args.model_name}_{args.baseline.upper()}"
        print(f"\nTraining baseline {args.baseline.upper()}  "
              f"(INPUT_TYPE={model.INPUT_TYPE}, crops_data={crops_data})")

        train_loader = build_loader(
            cfg, train_videos, train_transforms,
            shuffle=True, batch_size=cfg.training.stage1.batch_size,
            crops_data=crops_data,
        )
        val_loader = build_loader(
            cfg, val_videos, eval_transforms,
            shuffle=False, batch_size=cfg.training.stage1.batch_size,
            crops_data=crops_data,
        )

        build_trainer(cfg, model, train_loader, val_loader,
                      model_name=model_name).train()

        ckpt_path = ckpt_dir / f"{model_name}.pt"
        save_checkpoint(
            {"model": model.state_dict(), "baseline": args.baseline},
            str(ckpt_path),
        )
        print(f"Checkpoint saved to: {ckpt_path}")

    else:
        # FULL MODEL — always uses crops
        train_loader = build_loader(
            cfg, train_videos, train_transforms,
            shuffle=True, batch_size=cfg.training.stage1.batch_size,
            crops_data=True,
        )
        val_loader = build_loader(
            cfg, val_videos, eval_transforms,
            shuffle=False, batch_size=cfg.training.stage1.batch_size,
            crops_data=True,
        )

        model = build_full_model(cfg)

        if args.stage is None:
            run_stage1(cfg, model, train_loader, val_loader, ckpt_dir, args.model_name)
            run_stage2(cfg, model, train_loader, val_loader, ckpt_dir, args.model_name)

        elif args.stage == 1:
            run_stage1(cfg, model, train_loader, val_loader, ckpt_dir, args.model_name)

            print("\nCaching Stage-1 person embeddings for Stage-2 …")
            cache_train_loader = build_loader(          # eval_transforms: deterministic,
                cfg, train_videos, eval_transforms,     # since person_embedder is now frozen
                shuffle=False, batch_size=cfg.training.stage1.batch_size,
                crops_data=True,
            )
            emb_dir = Path("outputs/embeddings") / args.model_name
            n_train = cache_person_embeddings(model, cache_train_loader, device, emb_dir / "train")
            n_val   = cache_person_embeddings(model, val_loader,          device, emb_dir / "val")
            print(f"  ✔ Cached {n_train} train / {n_val} val samples → {emb_dir}")

        else:
            if args.stage1_checkpoint:
                ckpt = torch.load(args.stage1_checkpoint, map_location=device)
                model.load_state_dict(ckpt.get("model", ckpt))
                print(f"Loaded stage 1 weights from: {args.stage1_checkpoint}")

            emb_dir = Path("outputs/embeddings") / args.model_name
            print(f"\nLoading cached Stage-1 embeddings from {emb_dir} …")
            emb_train_loader = build_embedding_loader(
                emb_dir / "train", batch_size=cfg.training.stage2.batch_size, shuffle=True,
                num_workers=cfg.dataset.num_workers,
                pin_memory=cfg.dataset.pin_memory and torch.cuda.is_available(),
            )
            emb_val_loader = build_embedding_loader(
                emb_dir / "val", batch_size=cfg.training.stage2.batch_size, shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=cfg.dataset.pin_memory and torch.cuda.is_available(),
            )
            run_stage2(cfg, model, emb_train_loader, emb_val_loader, ckpt_dir, args.model_name,
                    use_precomputed_embeddings=True)


if __name__ == "__main__":
    main()
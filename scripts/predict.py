"""
scripts/predict.py
------------------
Run inference on a single annotated frame.

Loads a checkpoint, builds the model, processes one sample, and prints
the predicted group activity and per-player action labels.

Usage
-----
    python scripts/predict.py \
        --checkpoint outputs/checkpoints/final_model.pt \
        --video_id   45 \
        --frame_id   23455

    # Override device
    python scripts/predict.py \
        --checkpoint outputs/checkpoints/final_model.pt \
        --video_id 45 --frame_id 23455 \
        --device cpu
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.dataset import VolleyballDataset
from src.data.transforms import eval_transforms
from src.models.hierarchical_model import HierarchicalGroupActivityModel
from src.models.person_embedder import build_alexnet_fc7, build_resnet50, build_mobilenet_v3_large


# ---------------------------------------------
# Helpers
# ---------------------------------------------

def build_model(cfg: Config, checkpoint: Path, device: str) -> HierarchicalGroupActivityModel:
    if cfg.cnn.backbone == "alexnet":
        feature_extractor = build_alexnet_fc7
    elif cfg.cnn.backbone == "resnet50":
        feature_extractor = build_resnet50
    elif cfg.cnn.backbone == "mobilenet_v3_large":
        feature_extractor = build_mobilenet_v3_large
    else:
        raise ValueError(f"Unknown backbone: {cfg.cnn.backbone}")

    model = HierarchicalGroupActivityModel(
        feature_extractor = feature_extractor,
        lstm_hidden_p   = cfg.person_lstm.hidden_dim,
        lstm_hidden_g   = cfg.group_lstm.hidden_dim,
        person_classes  = cfg.labels.num_person_classes,
        group_classes   = cfg.labels.num_group_classes,
        n_subgroups     = cfg.pooling.num_subgroups,
        pool            = cfg.pooling.strategy,
        n_layers_p      = cfg.person_lstm.num_layers,
        n_layers_g      = cfg.group_lstm.num_layers,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_sample(
    cfg:      Config,
    video_id: int,
    frame_id: int,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Parse the annotation for (video_id, frame_id) and return:
        x       [N, T, C, H, W]   ready for model.forward()
        players list[dict]        sorted left→right, with action/bbox info
    """
    # Resolve data_root path relative to project root if it's relative
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_root = project_root / data_root
    
    ann_file  = data_root / str(video_id) / "annotations.txt"

    # Re-use the dataset parser — no need to duplicate logic
    dummy_dataset = VolleyballDataset(
        root         = data_root,
        split_videos = {video_id},
        cfg = cfg,
        transforms   = eval_transforms,
        T            = cfg.dataset.num_frames,
    )

    # Find the sample matching this frame_id
    match = None
    for vid, ann in dummy_dataset.samples:
        if vid == video_id and ann["frame_id"] == frame_id:
            match = (vid, ann)
            break

    if match is None:
        raise ValueError(
            f"No annotation found for video_id={video_id}, frame_id={frame_id}. "
            f"Check that the frame exists in {ann_file}."
        )

    # __getitem__ returns (x, group_label_tensor, person_labels)
    # We only need x and the sorted players for display
    sample_idx = dummy_dataset.samples.index(match)
    x, _, _ = dummy_dataset[sample_idx]           # [N, T, C, H, W]

    # Re-sort players to match the order used in __getitem__
    players = sorted(match[1]["players"], key=lambda p: p["bbox_center_x"])

    return x, players


def predict(
    model:   HierarchicalGroupActivityModel,
    x:       torch.Tensor,
    device:  str,
    cfg:     Config,
) -> tuple[str, list[str], torch.Tensor, torch.Tensor]:
    """
    Run a single forward pass.

    Returns
    -------
    group_pred      : str           predicted group activity label
    person_preds    : list[str]     predicted action label per player
    group_probs     : [8]           softmax probabilities over group classes
    person_probs    : [N, 9]        softmax probabilities over action classes
    """
    group_activities = cfg.labels.group_activities
    person_actions   = cfg.labels.person_actions

    x = x.unsqueeze(0) if x.dim() == 4 else x    # ensure [N, T, C, H, W]
    x = x.to(device)

    with torch.no_grad():
        group_logits, person_logits = model(x)
        # group_logits  : [8]
        # person_logits : [N, 9]

    group_probs  = group_logits.softmax(dim=-1)         # [8]
    person_probs = person_logits.softmax(dim=-1)        # [N, 9]

    group_pred   = group_activities[group_probs.argmax().item()]
    person_preds = [
        person_actions[p.argmax().item()] for p in person_probs
    ]

    return group_pred, person_preds, group_probs, person_probs


def report(
    video_id:     int,
    frame_id:     int,
    group_pred:   str,
    group_probs:  torch.Tensor,
    person_preds: list[str],
    person_probs: torch.Tensor,
    players:      list[dict],
    cfg:          Config,
) -> None:
    """Print a formatted prediction report to stdout."""
    group_activities = cfg.labels.group_activities
    person_actions   = cfg.labels.person_actions
    W = 20

    print("\n" + "=" * 70)
    print(f"PREDICTION  —  video {video_id}  /  frame {frame_id}")
    print("=" * 70)

    # -- Group activity ---------------------------------------------
    print(f"\n  Group Activity  →  {group_pred.upper()}")
    print(f"\n  {'Class':<{W}} {'Probability':>12}")
    print(f"  {'-'*W} {'-'*12}")
    for label, prob in zip(group_activities, group_probs.tolist()):
        marker = " ◀" if label == group_pred else ""
        print(f"  {label:<{W}} {prob * 100:>11.2f}%{marker}")

    # -- Per-player actions ---------------------------------------------
    print(f"\n  {'Player':>6}  {'BBox (x,y,w,h)':<22} {'Predicted':<12} {'Confidence':>10}")
    print(f"  {'------':>6}  {'-'*22} {'-'*12} {'-'*10}")
    for i, (player, pred, probs) in enumerate(
        zip(players, person_preds, person_probs)
    ):
        bbox       = player["bbox"]
        confidence = probs.max().item()
        bbox_str   = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
        print(f"  {i+1:>6}  {bbox_str:<22} {pred:<12} {confidence * 100:>9.2f}%")

    print("=" * 70 + "\n")


# ---------------------------------------------
# CLI
# ---------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a single annotated frame"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument("--video_id", type=int, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda | cpu  (overrides config)",
    )
    return parser.parse_args()


# ---------------------------------------------
# Main
# ---------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Config - resolve relative paths from project root -----
    config_path = Path(args.config)
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / config_path
    
    cfg = Config.from_yaml(config_path)
    if args.device is not None:
        cfg.merge({"training": {"device": args.device}})
    device = cfg.training.device

    # -- Checkpoint ---------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -- Model ---------------------------------------------
    model = build_model(cfg, ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # -- Sample ---------------------------------------------
    x, players = load_sample(cfg, args.video_id, args.frame_id)
    print(f"Loaded sample: video={args.video_id}  frame={args.frame_id}  players={len(players)}")

    # -- Predict ---------------------------------------------
    group_pred, person_preds, group_probs, person_probs = predict(
        model, x, device, cfg
    )

    # -- Report ---------------------------------------------
    report(
        video_id     = args.video_id,
        frame_id     = args.frame_id,
        group_pred   = group_pred,
        group_probs  = group_probs,
        person_preds = person_preds,
        person_probs = person_probs,
        players      = players,
        cfg          = cfg,
    )


if __name__ == "__main__":
    main()
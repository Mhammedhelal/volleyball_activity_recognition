# Volleyball Activity Recognition

A PyTorch implementation of a **hierarchical two-stage deep temporal model** for
group activity recognition, based on Ibrahim et al., *"Hierarchical Deep Temporal
Models for Group Activity Recognition"* (IEEE TPAMI). The model jointly predicts
a **team-level group activity** (e.g. `r_spike`, `l_set`) and **individual player
actions** (e.g. `spiking`, `blocking`) from short video clips of volleyball rallies.

Also included: all **7 baselines (B1–B7)** from the paper for ablation studies,
a flexible **YAML-based config system**, a **subgroup pooling** module (1/2/4
subgroups), Stage-1 embedding caching for fast Stage-2 fine-tuning, and a
results-visualization script.

---

## Architecture

```
Person crops [N, T, C, H, W]
        │
        ▼
┌───────────────────────┐
│ Stage 1 — PersonEmbedder │   CNN (AlexNet / ResNet50 / MobileNetV3) → LSTM1
│  per-person, per-frame  │   → P[N, T, D+H]  +  person_logits[N, 9]
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 2a — SubGroupPooler│  Max/Avg pool players within each subgroup
│  (1 / 2 / 4 subgroups)   │  → Z[1, T, z_dim]
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Stage 2b — FrameDescriptor│ LSTM2 over frame descriptors
│         (LSTM2)           │ → group_logits[8]
└───────────────────────┘
```

**Outputs:**

- `group_logits [8]` — main task: team activity classification
- `person_logits [N, 9]` — auxiliary task: per-player action classification

| Component | Role |
|---|---|
| CNN backbone | Frozen or fine-tuned spatial feature extractor (fc7 / avgpool output) |
| LSTM1 | Per-person temporal model, shared weights across players |
| SubGroupPooler | Aggregates players into left/right or quadrant sub-groups |
| LSTM2 | Group-level temporal model over pooled frame descriptors |

---

## Baselines

Seven baselines from the paper are implemented for ablation comparisons:

| Key | Baseline | Input | Person LSTM | Group LSTM | Paper Acc |
|-----|----------|-------|:-----------:|:----------:|:---------:|
| B1 | Image Classifier | full frame | ✗ | ✗ | 66.7% |
| B2 | Person Classifier | crops | ✗ | ✗ | 64.6% |
| B3 | Fine-Tuned Person Classifier | crops | ✗ | ✗ | 68.1% |
| B4 | Temporal Image Model (LRCN) | full frame | ✗ | ✓ | 63.1% |
| B5 | Temporal Person Model | crops | ✗ | ✓ | 67.6% |
| B6 | No LSTM1 (person LSTM removed) | crops | ✗ | ✓ | 74.7% |
| B7 | No LSTM2 (group LSTM removed) | crops | ✓ | ✗ | 80.2% |
| **Full** | **Hierarchical (both LSTMs)** | crops | ✓ | ✓ | **81.9%** |

---

## Project Structure

```
volleyball_activity_recognition/
├── README.md
├── requirements.txt
├── cmd.md                        # full CLI command reference
├── .env                          # PYTHONPATH=.
│
├── configs/
│   ├── default.yaml              # base hyperparameters
│   ├── 2group.yaml                # override: 2-subgroup pooling
│   └── 4group.yaml                # override: 4-subgroup pooling
│
├── data/
│   └── info.txt                  # dataset description & video list
│
├── src/
│   ├── config.py                 # Config: YAML I/O, dot/dict access, merging
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # VolleyballDataset + volleyball_collate
│   │   ├── labels.py             # PERSON_ACTIONS, GROUP_ACTIVITIES
│   │   ├── splits.py             # TRAIN/VAL/TEST video ID sets
│   │   └── transforms.py         # train/eval torchvision pipelines
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_backbones.py      # AlexNet / ResNet50 / MobileNetV3 builders
│   │   ├── person_embedder.py    # Stage 1: CNN + LSTM1
│   │   ├── subgroup_pooler.py    # Stage 2a: sub-group max/avg pooling
│   │   ├── frame_descriptor.py   # Stage 2b: LSTM2 + classifier
│   │   ├── hierarchical_model.py # end-to-end model (x or cached P input)
│   │   └── baselines/
│   │       ├── __init__.py       # BASELINES registry {"B1": ..., ..., "B7": ...}
│   │       ├── base.py           # BaselineModel ABC + pool_persons()
│   │       ├── b1_image_classifier.py
│   │       ├── b2_b3_person_classifier.py
│   │       ├── b4_temporal_image.py
│   │       ├── b5_temporal_person.py
│   │       ├── b6_no_lstm1.py
│   │       └── b7_no_lstm2.py
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py            # unified Trainer (hierarchical + baselines)
│   │   └── evaluator.py          # Evaluator + text report generation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── subgroups.py          # make_subgroup_indices()
│       ├── metrics.py            # AverageMeter, MetricsTracker
│       ├── checkpointing.py      # save/load/get_latest_checkpoint
│       ├── embedding_cache.py    # cache Stage-1 P embeddings for Stage 2
│       └── visualization.py      # parse eval logs → dashboard PNGs
│
├── scripts/
│   ├── common.py                 # shared helpers (build_model, build_loader, ...)
│   ├── train.py                  # training entrypoint (full model + baselines)
│   ├── evaluate.py               # evaluation entrypoint
│   ├── predict.py                # single-frame inference
│   └── visualize_results.py      # build dashboards from eval logs
│
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_person_embedder.py
│   ├── test_subgroup_pooler.py
│   ├── test_frame_descriptor.py
│   ├── test_hierarchical.py
│   └── test_b1.py … test_b7.py   # one file per baseline
│
└── outputs/
    ├── checkpoints/               # model_*.pt
    ├── logs/                      # eval_*.txt reports
    ├── figures/                   # dashboard PNGs
    └── embeddings/                # cached Stage-1 person embeddings
```

---

## Installation

```bash
git clone <this-repo>
cd volleyball_activity_recognition
pip install -r requirements.txt
```

Requires Python ≥ 3.10 (uses `list[int]`-style type hints and `from __future__ import annotations`).

### Dataset

Expects the Volleyball dataset laid out as:

```
data/videos_g10/
├── 0/
│   ├── annotations.txt
│   └── 12345/
│       ├── 12341.jpg … 12350.jpg   # temporal window around the annotated frame
├── 1/
│   └── ...
```

Each `annotations.txt` line has the format:

```
<frame>.jpg  <group_label>  x1 y1 w1 h1 action1  x2 y2 w2 h2 action2  ...
```

Update `paths.data_root` in `configs/default.yaml` (or pass `--data-root`) to point at your local copy.

---

## Quick Start

### Train the full hierarchical model

```bash
# Both stages, all defaults
python scripts/train.py

# Custom hyperparameters
python scripts/train.py --lr 3e-5 --batch_size 16 --num_epochs 50

# 2-subgroup pooling (left/right team)
python scripts/train.py --num_subgroups 2 --model-name model_2sg

# Stage 1 only, then Stage 2 with cached embeddings
python scripts/train.py --stage 1 --model-name run1
python scripts/train.py --stage 2 --stage1_checkpoint outputs/checkpoints/run1_stage1.pt --model-name run1
```

### Train a baseline

```bash
python scripts/train.py --baseline B7 --model-name baseline_B7
python scripts/train.py --baseline B6 --pool avg --lstm_hidden 1024
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/model_stage2_best.pt
python scripts/evaluate.py --checkpoint outputs/checkpoints/baseline_B7.pt --baseline B7 --split val
```

### Single-frame prediction

```bash
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_stage2_best.pt \
    --video_id 1 --frame_id 23455
```

### Visualize results

```bash
python scripts/visualize_results.py --model hierarchical
python scripts/visualize_results.py --compare   # cross-model comparison dashboard
```

### Run tests

```bash
pytest tests/ -v
```

See **`cmd.md`** for the full CLI reference, including every flag and example workflow
(reproducing the paper's best result, ablations over subgroups/pooling, etc.).

---

## Configuration

All hyperparameters live in `configs/default.yaml` and are loaded through
`src/config.py`, which supports both dot-notation and dict-notation access:

```python
from src.config import Config

cfg = Config.from_yaml("configs/default.yaml")
cfg.training.stage1.lr            # dot access
cfg["pooling"]["num_subgroups"]    # dict access

cfg.merge({"training": {"stage1": {"lr": 3e-5}}})   # runtime override
cfg.to_yaml("outputs/run_config.yaml")               # persist final config
```

Override files (`configs/2group.yaml`, `configs/4group.yaml`) can be layered on
top of the base config via `Config.from_yaml_with_overrides(...)`, and CLI flags
in `scripts/train.py` / `scripts/evaluate.py` apply on top of that.

---

## Key Design Decisions

1. **CNN backbone** — pretrained AlexNet (paper default), ResNet50, or MobileNetV3-Large; optionally frozen.
2. **Person-level temporal modeling (LSTM1)** — shared weights across all players, fused with CNN features at *every* timestep (not just the last), enabling proper per-frame pooling downstream.
3. **Sub-group pooling** — players are pre-sorted left→right by bounding-box x-coordinate, then split into 1 (whole team), 2 (left/right), or 4 (quadrant) contiguous groups and max/avg-pooled.
4. **Group-level temporal modeling (LSTM2)** — consumes the pooled per-frame descriptors and classifies from the final hidden state.
5. **Multi-task learning** — group-activity loss (primary) + weighted person-action loss (auxiliary), both `CrossEntropyLoss`.
6. **Variable group size** — a custom collate function (`volleyball_collate`) keeps per-sample player crops as a list instead of stacking, since `N` (players per frame) varies.
7. **Two-stage training** — Stage 1 trains the CNN+LSTM1 with person-action supervision; Stage 2 freezes Stage 1 and trains LSTM2 on cached person embeddings for speed.

---

## License / Attribution

Model architecture based on:
> M. S. Ibrahim, S. Muralidharan, Z. Deng, A. Vahdat, G. Mori, *"Hierarchical Deep
> Temporal Models for Group Activity Recognition,"* IEEE Transactions on Pattern
> Analysis and Machine Intelligence.

Dataset: the Volleyball Dataset (55 videos, 4830 annotated frames, 9 player
action classes, 8 team activity classes) — see `data/info.txt` for full details
and source video list.

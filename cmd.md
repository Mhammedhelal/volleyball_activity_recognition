Here is a full, comprehensive guide to all the commands for this repo.

---

# Complete Command Reference

## Setup

```bash
pip install -r requirements.txt
```

---

## 🟢 TRAIN (`scripts/train.py`)

### Full Command Syntax

```
python scripts/train.py [--config CONFIG]
                        [--baseline {B1,B2,B3,B4,B5,B6,B7}]
                        [--stage {1,2}]
                        [--stage1_checkpoint PATH]
                        [--data-root PATH]
                        [--model-name NAME]
                        [--lr FLOAT]
                        [--batch_size INT]
                        [--num_epochs INT]
                        [--num_subgroups {1,2,4}]
                        [--device {cuda,cpu}]
                        [--pool {max,avg}]
                        [--lstm_hidden INT]
```

---

### Full Hierarchical Model (2-Stage)

```bash
# Minimal — uses all defaults from configs/default.yaml
python scripts/train.py

# Both stages, custom data path and named checkpoint
python scripts/train.py \
    --data-root /path/to/volleyball \
    --model-name experiment_run1

# Tune learning rate, batch size, epochs
python scripts/train.py \
    --lr 3e-5 \
    --batch_size 16 \
    --num_epochs 50

# Run only Stage 1 (CNN + LSTM1, person-action supervision)
python scripts/train.py \
    --stage 1 \
    --model-name stage1_only

# Run only Stage 2 (LSTM2, group-activity supervision)
# Loading Stage 1 weights first
python scripts/train.py \
    --stage 2 \
    --stage1_checkpoint outputs/checkpoints/stage1_only_stage1.pt \
    --model-name stage2_finetune

# Use 2-subgroup pooling (left team / right team)
python scripts/train.py \
    --num_subgroups 2 \
    --model-name model_2subgroups

# Use 4-subgroup pooling (quadrants)
python scripts/train.py \
    --num_subgroups 4 \
    --model-name model_4subgroups

# GPU training, custom config
python scripts/train.py \
    --config configs/default.yaml \
    --device cuda \
    --lr 1e-4 \
    --batch_size 32 \
    --num_epochs 100 \
    --num_subgroups 2 \
    --model-name full_model_gpu
```

---

### Baseline Models

There are 7 baselines (from the paper). Use `--baseline KEY` to select one:

| Key  | Baseline                     | Input      | Person LSTM | Group LSTM | Paper Acc       |
| ---- | ---------------------------- | ---------- | ----------- | ---------- | --------------- |
| B1   | Image Classifier             | full frame | ✗          | ✗         | 66.7%           |
| B2   | Person Classifier            | crops      | ✗          | ✗         | 64.6%           |
| B3   | Fine-Tuned Person Classifier | crops      | ✗          | ✗         | 68.1%           |
| B4   | Temporal Image Model         | full frame | ✗          | ✓         | 63.1%           |
| B5   | Temporal Person Model        | crops      | ✗          | ✓         | 67.6%           |
| B6   | No LSTM1 (no person LSTM)    | crops      | ✗          | ✓         | 74.7%           |
| B7   | No LSTM2 (no group LSTM)     | crops      | ✓          | ✗         | 80.2%           |
| Full | Hierarchical (both LSTMs)    | crops      | ✓          | ✓         | **81.9%** |

```bash
# Train B1 — image classification baseline (no persons, no time)
python scripts/train.py --baseline B1 --model-name baseline_B1

# Train B2 — person classification, no temporal modeling
python scripts/train.py --baseline B2 --model-name baseline_B2

# Train B3 — fine-tuned person classifier
python scripts/train.py --baseline B3 --model-name baseline_B3

# Train B4 — temporal model on full frames (LSTM over whole frames)
python scripts/train.py --baseline B4 --model-name baseline_B4

# Train B5 — temporal model with person crops
python scripts/train.py --baseline B5 --model-name baseline_B5

# Train B6 — two-stage without LSTM1 (group LSTM only)
python scripts/train.py --baseline B6 --model-name baseline_B6

# Train B7 — two-stage without LSTM2 (person LSTM only)
python scripts/train.py --baseline B7 --model-name baseline_B7

# Baselines support pool and lstm_hidden overrides
python scripts/train.py \
    --baseline B6 \
    --pool avg \
    --lstm_hidden 1024 \
    --model-name B6_avg_pool

python scripts/train.py \
    --baseline B7 \
    --pool max \
    --lstm_hidden 512 \
    --lr 1e-4 \
    --num_epochs 30 \
    --model-name B7_tuned
```

---

## 🔵 EVALUATE (`scripts/evaluate.py`)

### Full Command Syntax

```
python scripts/evaluate.py --checkpoint PATH
                           [--config CONFIG]
                           [--baseline {B1,...,B7}]
                           [--split {val,test}]
                           [--device {cuda,cpu}]
                           [--batch_size INT]
```

---

### Full Hierarchical Model Evaluation

```bash
# Evaluate on test split (default)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_stage2.pt

# Evaluate on validation split (use during tuning, not final reporting)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --split val

# Evaluate a Stage 1 checkpoint (person-level only, no LSTM2)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_stage1.pt \
    --split test

# Force CPU evaluation
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --device cpu

# Custom batch size for evaluation (useful on low-memory GPUs)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --batch_size 4

# Use a custom config (e.g. with different num_subgroups)
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/model_2subgroups_stage2.pt \
    --config configs/default.yaml \
    --split test
```

---

### Baseline Evaluation

```bash
# Evaluate B1 baseline
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/baseline_B1.pt \
    --baseline B1 \
    --split test

# Evaluate B7 baseline on val split
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/baseline_B7.pt \
    --baseline B7 \
    --split val

# Compare all baselines — run one command per baseline:
for B in B1 B2 B3 B4 B5 B6 B7; do
    python scripts/evaluate.py \
        --checkpoint outputs/checkpoints/baseline_${B}.pt \
        --baseline $B \
        --split test
done
```

Sample output produced:

```
======================================================================
EVALUATION RESULTS
======================================================================
Group Activity Accuracy: 72.34%  (154/213)
----------------------------------------------------------------------
  Class              Accuracy
  r_set              82.50%
  r_spike            65.00%
  ...
  Confusion Matrix (rows=truth, cols=predicted): ...
```

---

## 🟣 PREDICT (`scripts/predict.py`)

### Full Command Syntax

```
python scripts/predict.py --checkpoint PATH
                          --video_id INT
                          --frame_id INT
                          [--config CONFIG]
                          [--device {cuda,cpu}]
```

`--video_id` is the integer folder name (0–55 for the Volleyball dataset). `--frame_id` is the annotated frame number inside that video folder.

```bash
# Basic prediction — video 1, frame 23455
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --video_id 1 \
    --frame_id 23455

# Predict on a different video and frame
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --video_id 4 \
    --frame_id 29885

# Use a baseline checkpoint for inference
python scripts/predict.py \
    --checkpoint outputs/checkpoints/baseline_B7.pt \
    --video_id 0 \
    --frame_id 12345

# Force CPU (useful if CUDA not available)
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_stage2.pt \
    --video_id 5 \
    --frame_id 30000 \
    --device cpu

# Custom config (e.g. if trained with 4 subgroups)
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_4subgroups_stage2.pt \
    --config configs/default.yaml \
    --video_id 9 \
    --frame_id 40123
```

---

## 🔬 TESTS

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_dataset.py -v
pytest tests/test_hierarchical.py -v
pytest tests/test_person_embedder.py -v
pytest tests/test_subgroup_pooler.py -v
pytest tests/test_frame_descriptor.py -v
```

---

## Common Experiment Workflows

### Reproduce the Paper's Best Result (81.9%)

```bash
# Stage 1
python scripts/train.py \
    --num_subgroups 2 \
    --pool max \
    --stage 1 \
    --model-name paper_repro

# Stage 2
python scripts/train.py \
    --num_subgroups 2 \
    --pool max \
    --stage 2 \
    --stage1_checkpoint outputs/checkpoints/paper_repro_stage1.pt \
    --model-name paper_repro

# Evaluate
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/paper_repro_stage2.pt \
    --split test
```

### Ablation: Compare Subgroup Configs

```bash
python scripts/train.py --num_subgroups 1 --model-name ablation_1sg
python scripts/train.py --num_subgroups 2 --model-name ablation_2sg
python scripts/train.py --num_subgroups 4 --model-name ablation_4sg

python scripts/evaluate.py --checkpoint outputs/checkpoints/ablation_1sg_stage2.pt
python scripts/evaluate.py --checkpoint outputs/checkpoints/ablation_2sg_stage2.pt
python scripts/evaluate.py --checkpoint outputs/checkpoints/ablation_4sg_stage2.pt
```

### Ablation: Compare Pooling Strategies

```bash
python scripts/train.py --baseline B6 --pool max --model-name B6_max
python scripts/train.py --baseline B6 --pool avg --model-name B6_avg
python scripts/evaluate.py --checkpoint outputs/checkpoints/B6_max_B6.pt --baseline B6
python scripts/evaluate.py --checkpoint outputs/checkpoints/B6_avg_B6.pt --baseline B6
```

---

## Quick Reference Card

| Goal                           | Key flags                                |
| ------------------------------ | ---------------------------------------- |
| Train full model, both stages  | `train.py`(no extra flags)             |
| Train only stage 1             | `--stage 1`                            |
| Train only stage 2             | `--stage 2 --stage1_checkpoint <path>` |
| Train a baseline               | `--baseline B1`…`--baseline B7`     |
| Use 2-subgroup pooling         | `--num_subgroups 2`                    |
| Use 4-subgroup pooling         | `--num_subgroups 4`                    |
| Custom data path               | `--data-root /path/to/data`            |
| Name your checkpoint           | `--model-name my_run`                  |
| Evaluate on val set            | `evaluate.py --split val`              |
| Evaluate a baseline checkpoint | `--baseline B7`                        |
| Single-frame inference         | `predict.py --video_id X --frame_id Y` |

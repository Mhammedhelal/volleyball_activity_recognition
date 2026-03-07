# Volleyball Activity Recognition - Complete Documentation

## Overview

This project implements a **Hierarchical Deep Temporal Model for Group Activity Recognition** based on the paper by Ibrahim et al. (IEEE TPAMI). The system recognizes both individual player actions and team-level group activities in volleyball matches using a two-stage hierarchical architecture.

**Architecture Summary:**

- **Stage 1 (Person Level):** CNN + LSTM1 → person embeddings P_{t,k}
- **Stage 2a (Sub-group Pooling):** Max/Avg pool → frame descriptors Z_t
- **Stage 2b (Group Level):** LSTM2 → group activity classification

---

## Project Structure

```
volleyball_activity_recognition/
├── src/
│   ├── config.py              # Configuration management system
│   ├── data/
│   │   ├── dataset.py         # PyTorch Dataset implementation
│   │   ├── labels.py          # Label definitions (person actions, group activities)
│   │   ├── splits.py          # Train/Val/Test video splits
│   │   ├── transforms.py      # Image preprocessing pipelines
│   │   └── labels.json        # Label name mappings
│   ├── models/
│   │   ├── person_embedder.py   # Stage 1: PersonEmbedder (CNN + LSTM1)
│   │   ├── subgroup_pooler.py   # Stage 2a: SubGroupPooler
│   │   ├── frame_descriptor.py  # Stage 2b: FrameDescriptor (LSTM2)
│   │   └── hierarchical_model.py # Complete end-to-end model
│   ├── engine/
│   │   ├── trainer.py         # Training loop
│   │   ├── evaluator.py       # Evaluation and metrics computation
│   │   └── losses.py          # Loss functions
│   └── utils/
│       ├── metrics.py         # AverageMeter, MetricsTracker
│       ├── subgroups.py       # Subgroup assignment functions
│       └── checkpointing.py   # Model saving/loading (TODO)
├── scripts/
│   ├── train.py               # Training entrypoint
│   ├── evaluate.py            # Evaluation entrypoint
│   └── predict.py             # Single-sample inference
├── configs/
│   ├── default.yaml           # Base configuration
│   ├── 2group.yaml            # 2-subgroup override
│   └── 4group.yaml            # 4-subgroup override
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_pipeline_smoke_test.ipynb
│   └── 03_results_analysis.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_person_embedder.py
│   ├── test_subgroup_pooler.py
│   ├── test_frame_descriptor.py
│   └── test_hierarchical.py
└── outputs/
    ├── checkpoints/           # Saved model weights
    ├── logs/                  # TensorBoard logs
    └── figures/               # Visualizations
```

---

## Configuration System (`src/config.py`)

### Purpose

Provides a hierarchical configuration management system with YAML I/O and dot-notation access.

### Key Classes

#### `_Namespace`

Internal class providing recursive dot-notation access to configuration dictionaries.

**Key Methods:**

- `__getattr__(key)` / `__setattr__(key, value)`: Dot-notation access (`cfg.training.lr`)
- `__getitem__(key)` / `__setitem__(key, value)`: Dict-notation access (`cfg["training"]["lr"]`)
- `to_dict()`: Convert namespace recursively back to plain Python dict
- `_merge_dict(override)`: Deep-merge override dict into self

#### `Config` (extends `_Namespace`)

Top-level configuration object with YAML support.

**Factory Methods:**

- `from_yaml(path)`: Load configuration from YAML file
- `from_dict(data)`: Construct from plain dict
- `from_yaml_with_overrides(base_path, *override_paths, overrides=None)`: Load base config and apply overrides sequentially

**Merge & Export:**

- `merge(override)`: Deep-merge override into config (in-place), returns self for chaining
- `to_yaml(path)`: Write current config to YAML file
- `copy()`: Return deep copy of config

**Example Usage:**

```python
# Load base config
cfg = Config.from_yaml("configs/default.yaml")

# Access values
lr = cfg.training.stage1.lr      # dot-notation
batch_size = cfg["training"]["stage1"]["batch_size"]  # dict-notation

# Merge overrides at runtime
cfg.merge({"training": {"stage1": {"lr": 3e-5}}})

# Save modified config
cfg.to_yaml("outputs/run_config.yaml")
```

---

## Data Module (`src/data/`)

### `labels.py`

Defines label spaces for the two classification tasks.

**Contents:**

```python
PERSON_ACTIONS = [
    "waiting", "setting", "digging", "falling",
    "spiking", "blocking", "jumping", "moving", "standing"
]  # 9 classes

GROUP_ACTIVITIES = [
    "r_set", "r_spike", "r_pass", "r_winpoint",
    "l_winpoint", "l_pass", "l_spike", "l_set"
]  # 8 classes
```

Loaded from `labels.json` file.

### `splits.py`

Defines which video IDs belong to train/val/test splits.

```python
TRAIN_VIDEOS = {1, 3, 6, 7, 10, 13, ...}  # 24 videos
VAL_VIDEOS   = {0, 2, 8, 12, 17, ...}     # 15 videos
TEST_VIDEOS  = {4, 5, 9, 11, 14, ...}     # 16 videos
```

### `transforms.py`

Image preprocessing pipelines using torchvision.

**Components:**

**Train Transforms** (`train_transforms`):

- `Resize(224, 224)`: Resize to ImageNet canonical size
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`: Photometric augmentation
- `ToTensor()`: Convert PIL Image to tensor [0,1]
- `Normalize(IMAGENET_MEAN, IMAGENET_STD)`: ImageNet normalization

⚠️ **Note:** RandomHorizontalFlip NOT applied — flipping reverses left/right team assignments which would corrupt group labels.

**Eval Transforms** (`eval_transforms`):

- `Resize((224, 224))`
- `ToTensor()`
- `Normalize(IMAGENET_MEAN, IMAGENET_STD)`

### `dataset.py`

PyTorch Dataset for volleyball activity recognition.

#### `VolleyballDataset`

**Constructor Parameters:**

- `root: Path` - Root directory containing extracted dataset
- `split_videos: set[int]` - Video IDs for this split
- `cfg: Config` - Configuration object (for image size, mean/std, etc.)
- `transforms` - Optional preprocessing pipeline
- `T: int` - Temporal window size (must be odd, e.g., 9)

**Data Format Expected:**

```
data/volleyball/
├── 0/
│   ├── annotations.txt
│   └── 12345/
│       ├── 12341.jpg
│       ├── 12342.jpg
│       ...
│       ├── 12349.jpg
│       └── 12350.jpg
├── 1/
│   └── ...
```

Annotation format:

```
<frame>.jpg  <group_label>  x1 y1 w1 h1 action1  x2 y2 w2 h2 action2  ...
```

**Key Methods:**

`__init__()`

1. Parse all annotation files for videos in `split_videos`
2. Store list of `(video_id, annotation_dict)` tuples
3. Validate that T is odd (symmetric temporal window)

Lines 1-60: Setup and validation

**`__getitem__(idx)`** (lines 61-140)
Returns a single sample for training/evaluation.

**Input:** `idx` - index into self.samples

**Process:**

1. Retrieve `(video_id, annotation)` tuple
2. Extract `frame_id`, `group_label`, and sorted player list (sorted by bbox x-coordinate, left→right)
3. Generate symmetric temporal window: `[frame_id - T//2, ..., frame_id, ..., frame_id + T//2]`
4. Pre-load all T frame images from disk (avoiding re-opening same file multiple times)
5. For each player:
   - Extract bounding box crops across all T frames
   - Clamp bbox to image boundaries (safety)
   - Apply transforms (resize, normalize, etc.)
   - Stack temporal crops → `[T, C, H, W]`
6. Stack all players → `x = [N, T, C, H, W]`
7. Create labels:
   - `person_labels = [N]` (action class indices)
   - `group_label_tensor = [1]` (wrapped for batch processing)

**Output:**

```python
(
    x: torch.Tensor [N, T, C, H, W],  # N players, T frames, C channels, H×W spatial
    group_label_tensor: torch.Tensor [1],  # Single group activity label
    person_labels: torch.Tensor [N]   # Individual action labels
)
```

**Fixes Applied:**

- **FIX 1:** Video IDs now correctly paired with annotations (was causing path construction failures)
- **FIX 2:** Person crops correctly accumulated as `[N, T, C, H, W]` (was transposed)
- **FIX 3:** Missing frames substitute nearest available frame instead of crashing
- **FIX 4:** Group label wrapped in tensor `[1]` for clean batch concatenation

**Key Helper Methods:**

`_parse_annotations(ann_file)`

- Parse one annotations.txt file
- Returns list of dicts with `frame_id`, `group_label`, `players`
- Each player dict contains: `bbox`, `bbox_center_x`, `action`, `action_id`

`_nearest_frame(clip_dir, missing_fid, frame_ids)` (static)

- Find and return closest available frame when target is missing
- Fallback: return blank RGB image if no frames exist

#### `volleyball_collate(batch)`

Custom collate function for DataLoader.

**Problem:** Standard collate stacks all tensors along batch dimension (dim=0), but N (number of players) varies across samples. Cannot stack variable-size tensors.

**Solution:** Return lists where needed.

**Input:** `batch` - list of tuples from `__getitem__`

```python
[
    (x_0, group_label_0, person_labels_0),  # sample 0
    (x_1, group_label_1, person_labels_1),  # sample 1
    ...
]
```

**Process & Output:**

```python
(
    frames_list: list[B] of [N_i, T, C, H, W],  # Keep as list (variable N)
    group_labels: torch.Tensor [B],             # Safe to stack (always 1 per sample)
    person_labels_list: list[B] of [N_i]        # Keep as list (variable N)
)
```

---

## Model Architecture (`src/models/`)

### `person_embedder.py` - Stage 1: Person-Level Temporal Encoding

#### `PersonEmbedder`

Combines spatial features (CNN) with temporal dynamics (LSTM) to create rich person embeddings.

**Architecture:**

```
Input: x[N, T, C, H, W]
  ↓
CNN (shared across time)  → x_t,k[D] spatial features
  ↓
LSTM1 (per-person)        → h_t,k[H] temporal context
  ↓
Concatenate              → P_t,k[D+H] fused embedding at each t
  ↓
Output: person_logits[N, 9], P[N, T, D+H]
```

**Constructor Parameters:**

- `cnn_output_size: int = 4096` - AlexNet fc7 layer dimension
- `lstm_hidden: int = 512` - LSTM1 hidden state size (paper: 3000)
- `person_classes: int = 9` - Number of person action classes
- `n_layers: int = 1` - LSTM depth

**Key Attributes:**

- `self.cnn`: Frozen AlexNet up to fc7 (pretrained on ImageNet)
- `self.lstm`: LSTM for temporal sequence modeling
- `self.person_fc`: Fully-connected layer for action classification

**`build_alexnet_fc7()`** (static)

**Purpose:** Extract pretrained AlexNet feature extractor up to fc7.

**AlexNet Structure:**

```
AlexNet.features (conv layers)
  ↓
AlexNet.avgpool (spatial reduction)
  ↓
AlexNet.classifier:
  [0] Dropout
  [1] Linear(9216, 4096)   ← fc6
  [2] ReLU
  [3] Dropout
  [4] Linear(4096, 4096)   ← fc7
  [5] ReLU                 ← STOP HERE (output=4096)
  [6] Linear(4096, 1000)   ← ImageNet head (discarded)
```

**Process:**

1. Load pretrained AlexNet with ImageNet weights
2. Chain features → avgpool → flatten → classifier[:6]
3. Freeze all parameters (`requires_grad = False`)
4. Return (feature_extractor, 4096)

**`forward(x)`**

**Input:** `x [N, T, C, H, W]` - N players, T frames

**Process:**

1. **Reshape to process all frames:** `x.view(N*T, C, H, W)` → `[N*T, C, H, W]`
2. **CNN Forward:** Apply shared CNN to each crop independently

   ```
   cnn_out = self.cnn(x_view)  # [N*T, D] where D=4096
   cnn_out = cnn_out.view(N, T, D)  # [N, T, D]
   ```

   - No temporal awareness in CNN — each frame processed independently
   - Reduces computational load (no Conv3D)

3. **LSTM Forward:** Process temporal sequence for each person

   ```
   h0 = zeros(n_layers, N, hidden_size)  # [1, N, H]
   c0 = zeros(n_layers, N, hidden_size)  # [1, N, H]
   lstm_out, _ = self.lstm(cnn_out, (h0, c0))  # [N, T, H]
   ```

   - Processes one person at a time (batch_first=True)
   - Captures temporal dependencies within each person's sequence
   - Weights shared across all persons

4. **Fusion:** Concatenate spatial + temporal at every timestep

   ```
   P = cat([cnn_out, lstm_out], dim=-1)  # [N, T, D+H]
   ```

   - Allows downstream pooling to produce frame-level Z_t

5. **Person-Action Classification:** Use last timestep embedding

   ```
   person_logits = self.person_fc(P[:, -1, :])  # [N, 9]
   ```

   - Only supervised at T-1 (last timestep)
   - Auxiliary supervision signal for training

**Output:**

- `person_logits [N, 9]`: Logits for individual action classification
- `P [N, T, D+H]`: Person embeddings at every timestep

### `subgroup_pooler.py` - Stage 2a: Sub-group Aggregation

#### `SubGroupPooler`

Aggregates individual person embeddings into frame-level descriptors using sub-group pooling.

**Concept:** Players split into M "sub-groups" (e.g., left/right team, front/back court). Within each subgroup, aggregate via max or average pooling to create a group vector per frame.

**Architecture:**

```
Input: P[N, T, D+H] (person embeddings)
       subgroup_indices = [[0,1,2], [3,4,5], ...] (players per subgroup)
  ↓
For each frame t:
  For each subgroup m:
    G_t,m = MaxPool(P_t,k for k in subgroup_m)  [D+H]
  Z_t = concat(G_t,1, ..., G_t,M)               [M*(D+H)]
  ↓
Output: Z[1, T, z_dim] where z_dim = M * (D+H)
```

**Constructor Parameters:**

- `pool: str = "max"` - Pooling operation ("max" or "avg")

**`forward(P, subgroup_indices)`**

**Inputs:**

- `P [N, T, D+H]`: Person embeddings from PersonEmbedder
- `subgroup_indices: list[list[int]]`: M lists, each containing player indices for that subgroup
  - Example for 2 subgroups with 6 players: `[[0, 1, 2], [3, 4, 5]]`

**Process:**

1. Extract dimensions

   ```
   N, T, embed_dim = P.shape  # embed_dim = D+H
   Z_sequence = []
   ```

2. **For each frame t:**

   ```
   P_t = P[:, t, :]  # [N, D+H] — all persons at frame t
   ```

3. **For each subgroup m:**

   ```
   sub = P_t[subgroup_indices[m]]  # [N_sub, D+H]
   if pool == "max":
       g = sub.max(dim=0).values   # [D+H]  (element-wise max across players)
   else:
       g = sub.mean(dim=0)         # [D+H]  (element-wise mean)
   group_vecs.append(g)
   ```

4. **Concatenate subgroup vectors to form Z_t:**

   ```
   Z_t = cat(group_vecs, dim=-1)  # [M * (D+H)]
   Z_sequence.append(Z_t)
   ```

5. **Stack temporal sequence:**

   ```
   Z = stack(Z_sequence, dim=0).unsqueeze(0)  # [1, T, z_dim]
   ```

   - Unsqueeze adds batch dimension (always 1 for single video)
   - z_dim = M * (D+H)

**Output:** `Z [1, T, z_dim]` - Frame descriptors ready for LSTM2

### `frame_descriptor.py` - Stage 2b: Group-Level Temporal Model

#### `FrameDescriptor` (also called FrameDescriptor or LSTM2 Module)

Models temporal evolution of group activity at the frame level.

**Architecture:**

```
Input: Z[1, T, z_dim] (frame descriptors from SubGroupPooler)
  ↓
LSTM2 forward        → lstm_out[1, T, H_g]
Take last hidden     → h_group[H_g]
Classification head  → group_logits[8]
  ↓
Output: group_logits[8] (one score per group activity class)
```

**Constructor Parameters:**

- `z_dim: int` - Input dimension (frame descriptor size)
- `lstm_hidden: int = 2000` - LSTM2 hidden size (paper: 2000)
- `group_classes: int = 8` - Number of group activity classes
- `n_layers: int = 1` - LSTM depth

**Key Attributes:**

- `self.group_lstm`: LSTM2 for temporal sequence modeling of frames
- `self.group_fc`: Fully-connected layer for group activity classification

**`forward(Z)`**

**Input:** `Z [1, T, z_dim]` - Sequence of T frame descriptors

**Process:**

1. **LSTM2 Forward:** Process frame descriptor sequence

   ```
   lstm_out, _ = self.group_lstm(Z)  # [1, T, lstm_hidden]
   ```

   - batch_first=True, so input shape is [B=1, T, z_dim]
   - Output shape: [B=1, T, H_g]
   - Captures temporal patterns across frames

2. **Extract Last Hidden State:** Use final frame's state for classification

   ```
   h_group = lstm_out[0, -1, :]  # [H_g]
   ```

   - Indexing: `[batch=0, time=T-1, hidden]`
   - Contains aggregated information about entire temporal sequence

3. **Classification:** Linear projection to class logits

   ```
   group_logits = self.group_fc(h_group)  # [8]
   ```

**Output:** `group_logits [8]` - Logits for 8 group activity classes

### `hierarchical_model.py` - Complete End-to-End Model

#### `HierarchicalGroupActivityModel`

Full two-stage hierarchical model combining all components.

**Architecture Diagram:**

```
x[N, T, C, H, W]  (input: N players, T frames, RGB images)
    ↓
[Stage 1: PersonEmbedder]
    CNN → LSTM1 → P[N, T, D+H]  +  person_logits[N, 9]
    ↓
[Stage 2a: SubGroupPooler]
    MaxPool/AvgPool → Z[1, T, z_dim]
    ↓
[Stage 2b: FrameDescriptor]
    LSTM2 → group_logits[8]
    ↓
Output: (group_logits[8], person_logits[N, 9])
```

**Constructor Parameters:**

- `cnn_output_size: int = 4096`
- `lstm_hidden_p: int = 512` - PersonEmbedder LSTM hidden size
- `lstm_hidden_g: int = 512` - FrameDescriptor LSTM hidden size
- `person_classes: int = 9`
- `group_classes: int = 8`
- `n_subgroups: int = 2` - Number of subgroups (1/2/4)
- `pool: str = "max"` - Pooling operation
- `n_layers_p: int = 1` - PersonEmbedder LSTM depth
- `n_layers_g: int = 1` - FrameDescriptor LSTM depth

**`forward(x, subgroup_indices=None)`**

**Inputs:**

- `x [N, T, C, H, W]`: Player crops (players pre-sorted by bbox x-coordinate)
- `subgroup_indices: list[list[int]] | None`: Optional subgroup assignments
  - If None, auto-generate from `n_subgroups` using `make_subgroup_indices(N, n_subgroups)`

**Process:**

1. **Stage 1:** PersonEmbedder

   ```
   person_logits, P = self.person_embedder(x)
   # person_logits: [N, 9]
   # P: [N, T, D+H]
   ```

2. **Auto-generate subgroups if needed:**

   ```
   if subgroup_indices is None:
       subgroup_indices = make_subgroup_indices(N, self.n_subgroups)
   ```

3. **Stage 2a:** SubGroupPooler

   ```
   Z = self.subgroup_pooler(P, subgroup_indices)
   # Z: [1, T, z_dim] where z_dim = (D+H) * n_subgroups
   ```

4. **Stage 2b:** FrameDescriptor

   ```
   group_logits = self.frame_descriptor(Z)
   # group_logits: [8]
   ```

**Output:**

```python
(
    group_logits: torch.Tensor [8],       # Main task: group activity
    person_logits: torch.Tensor [N, 9]    # Auxiliary task: person actions
)
```

**Smoke Test** (if run as `__main__`)

Validates that shapes are correct with default hyperparameters.

---

## Training & Evaluation (`src/engine/`)

### `trainer.py` - Training Loop

#### `Trainer`

Manages the training loop for the hierarchical model with multi-task learning.

**Constructor Parameters:**

- `model: HierarchicalGroupActivityModel`
- `train_loader: DataLoader` - Uses `volleyball_collate`, yields variable-size batches
- `device: str = "cuda"`
- `learning_rate: float = 1e-5` - Paper uses 1e-5
- `momentum: float = 0.9` - SGD momentum
- `num_epochs: int = 100`
- `person_loss_w: float = 1.0` - Weight of auxiliary person-action loss
- `log_every: int = 10` - Logging frequency (epochs)

**Key Attributes:**

- `self.model`: Model on device
- `self.optimizer`: SGD with momentum
- `self.criterion_group`: CrossEntropyLoss for group activity
- `self.criterion_players`: CrossEntropyLoss for person actions
- `self.loss_meter`: AverageMeter for loss tracking
- `self.group_tracker`: MetricsTracker for group predictions
- `self.person_tracker`: MetricsTracker for person predictions

**`train_epoch()`**

Runs one full epoch of training.

**Process:**

1. **Set model to training mode and reset meters:**

   ```
   self.model.train()
   self.loss_meter.reset()
   self.group_tracker.reset()
   self.person_tracker.reset()
   ```

2. **For each batch from train_loader:**

   ```
   frames_list, group_labels, person_labels_list = batch
   # frames_list: list[B] of [N_i, T, C, H, W]
   # group_labels: [B]
   # person_labels_list: list[B] of [N_i]
   ```

3. **Initialize batch loss:**

   ```
   batch_loss = torch.tensor(0.0, device=device)
   ```

4. **For each sample i in batch:**

   ```
   frames = frames_list[i]           # [N_i, T, C, H, W]
   person_labels = person_labels_list[i]  # [N_i]
   group_label = group_labels[i]     # scalar (convert to [1])
   ```

5. **Forward pass:**

   ```
   group_logits, person_logits = model(frames)
   # group_logits: [8]
   # person_logits: [N_i, 9]
   ```

6. **Compute losses:**

   ```
   group_loss = criterion_group(
       group_logits.unsqueeze(0),       # [1, 8] for batch processing
       group_labels[i].unsqueeze(0)     # [1]
   )
   person_loss = criterion_players(
       person_logits,                   # [N_i, 9]
       person_labels                    # [N_i]
   )
   batch_loss += group_loss + person_loss_w * person_loss
   ```

7. **Accumulate metrics (no grad needed):**

   ```
   with torch.no_grad():
       group_tracker.update(
           preds=group_logits.argmax().unsqueeze(0),  # [1]
           targets=group_labels[i].unsqueeze(0)       # [1]
       )
       person_tracker.update(
           preds=person_logits.argmax(dim=-1),        # [N_i]
           targets=person_labels                      # [N_i]
       )
   ```

8. **Average loss over batch and backpropagate:**

   ```
   batch_loss = batch_loss / len(frames_list)
   optimizer.zero_grad()
   batch_loss.backward()
   optimizer.step()
   ```

9. **Update loss meter:**

   ```
   loss_meter.update(batch_loss.item(), n=len(frames_list))
   ```

**Output:** dict with keys:

- `"loss"`: Mean total loss
- `"group_accuracy"`: Group activity accuracy
- `"person_accuracy"`: Person action accuracy

**`train()`**

Runs full training for `num_epochs` epochs.

**Process:**

1. Print training header
2. For each epoch:
   - Call `train_epoch()`
   - Log metrics every `log_every` epochs
3. Print training footer

### `evaluator.py` - Evaluation

#### `Evaluator`

Evaluates trained model on validation or test split.

**Constructor Parameters:**

- `model: HierarchicalGroupActivityModel`
- `val_loader: DataLoader` - Uses `volleyball_collate`, no shuffling
- `device: str = "cuda"`

**Key Attributes:**

- `self.model`: Model on device (set to eval mode)
- `self.val_loader`: Evaluation DataLoader
- `self.group_tracker`: MetricsTracker for group predictions
- `self.person_tracker`: MetricsTracker for person predictions

**`evaluate()`**

Runs full evaluation pass (no gradients).

**Process:** (mirrors `Trainer.train_epoch()` but without backprop)

1. Set model to eval mode
2. Reset trackers
3. For each batch:
   - For each sample in batch:
     - Forward pass
     - Update trackers with predictions

**Output:** dict with keys:

- `"group_accuracy"`: Overall group activity accuracy
- `"person_accuracy"`: Overall person action accuracy
- `"group_per_class"`: dict[class_name, accuracy]
- `"person_per_class"`: dict[class_name, accuracy]
- `"group_correct"`: # correct predictions (group)
- `"group_total"`: # total predictions (group)
- `"person_correct"`: # correct predictions (person)
- `"person_total"`: # total predictions (person)
- `"group_confusion"`: Confusion matrix [8, 8]
- `"person_confusion"`: Confusion matrix [9, 9]

**`report()`**

Prints formatted evaluation report to stdout with:

- Overall and per-class accuracy for group activity
- Overall and per-class accuracy for person actions
- Formatted confusion matrices

### `losses.py`

Note: Based on code inspection, losses are defined inline in `Trainer`:

- `CrossEntropyLoss` for group activity (main task)
- `CrossEntropyLoss` for person actions (auxiliary task)
- Combined loss: `group_loss + person_loss_w * person_loss`

Multi-task learning approach with configurable weights.

---

## Utilities (`src/utils/`)

### `metrics.py` - Metric Tracking

#### `AverageMeter`

Tracks running average of a scalar value (loss, accuracy, etc.).

**Constructor:**

- `name: str = ""` - Human-readable name for logging

**Methods:**

`reset()`: Clear accumulated state

`update(val: float, n: int = 1)`: Add new value

- `val`: Scalar value for this update (e.g., batch loss)
- `n`: Number of samples represented (batch size)
- Updates: `sum += val * n`, `count += n`, `avg = sum / count`

`__repr__()`: String representation

- Returns: `"name: avg (avg)  last: last_val"` format

**Example:**

```python
meter = AverageMeter(name="loss")
meter.update(2.4, n=8)   # 8 samples with loss=2.4
meter.update(1.8, n=8)   # 8 samples with loss=1.8
print(meter.avg)         # → 2.1
```

#### `MetricsTracker`

Accumulates predictions and targets over an epoch/evaluation, computes accuracy and confusion matrix.

**Constructor:**

- `num_classes: int` - Number of output classes
- `class_names: list[str]` - Human-readable name for each class

**Key Attributes:**

- `self._confusion`: Confusion matrix [num_classes, num_classes]

**Methods:**

`reset()`: Clear confusion matrix

`update(preds: torch.Tensor, targets: torch.Tensor)`: Accumulate batch predictions

- `preds [N]`: Predicted class indices
- `targets [N]`: Ground-truth class indices
- Scatters into confusion matrix

`accuracy() → float`: Overall accuracy = correct / total

`per_class_accuracy() → dict[str, float]`: Per-class accuracy for each class

- Classes with zero samples return 0.0

`confusion_matrix() → torch.Tensor [C, C]`: Return raw confusion matrix

- Rows = ground truth, columns = predicted

**Example:**

```python
tracker = MetricsTracker(8, GROUP_ACTIVITIES)
tracker.update(preds=torch.tensor([0, 1]), targets=torch.tensor([0, 1]))
print(tracker.accuracy())         # → 1.0 (100% correct)
print(tracker.per_class_accuracy())  # → {'r_set': 1.0, 'r_spike': 1.0, ...}
```

### `subgroups.py` - Subgroup Assignment

#### `make_subgroup_indices(n_players: int, n_subgroups: int) → list[list[int]]`

Splits N players into M contiguous subgroups.

**Precondition:** Players must be pre-sorted by bounding-box x-coordinate (left → right).

**Examples:**

- `n_subgroups=1`: `[[0, 1, 2, 3, 4, 5]]` (all together)
- `n_subgroups=2`: `[[0, 1, 2], [3, 4, 5]]` (left vs. right team)
- `n_subgroups=4`: `[[0, 1], [2, 3], [4, 5], [6]]` (roughly equal quadrants)

**Algorithm:**

1. Compute base size: `base = n_players // n_subgroups`
2. Compute remainder: `extra = n_players % n_subgroups`
3. First `extra` subgroups get `base + 1` players; rest get `base`
4. Return list of subgroup indices

**Example:**

```python
# 7 players, 2 subgroups
indices = make_subgroup_indices(7, 2)
# base=3, extra=1 → first group gets 4 players
# → [[0, 1, 2, 3], [4, 5, 6]]
```

### `checkpointing.py`

Placeholder module for model checkpoint save/load utilities (TODO).

---

## Data & Configuration (`configs/`)

### `default.yaml` - Base Configuration

Master configuration file with all hyperparameters. Structured as nested YAML keys.

**Top-Level Sections:**

#### `paths:`

- `data_root`: Root directory for extracted dataset
- `output_dir`, `checkpoint_dir`, `log_dir`, `figures_dir`: Output directories

#### `dataset:`

- `train_videos`, `val_videos`, `test_videos`: Video ID lists for splits
- `num_frames`: 9 (temporal window size)
- `image_size`: (224, 224) - AlexNet canonical size
- `mean`, `std`: ImageNet normalization constants
- `num_workers`, `pin_memory`: DataLoader settings

#### `labels:`

- `group_activities`: List of 8 group activity class names
- `person_actions`: List of 9 person action class names
- `num_group_classes`, `num_person_classes`: Class counts

#### `cnn:`

- `backbone`: "alexnet"
- `feature_layer`: "fc7"
- `feature_dim`: 4096
- `pretrained`: true
- `freeze`: false (train jointly with LSTM1)

#### `person_lstm:`

- `input_dim`: 4096 (CNN fc7 output)
- `hidden_dim`: 3000 (paper value)
- `num_layers`: 1
- `dropout`: 0.0

#### `person_embedding:`

- `dim`: 7096 (4096 + 3000, for reference)

#### `group_lstm:` / `pooling:` / `training:` / `loss:` / ... (see file for full details)

### Override Configurations

- `2group.yaml`: Override to use 2 subgroups (left/right teams)
- `4group.yaml`: Override to use 4 subgroups (quadrants)

**Usage:**

```bash
# Load base config with 2-subgroup override
python scripts/train.py --config configs/default.yaml  # base is loaded first internally
# Or using Config.from_yaml_with_overrides()
```

---

## Training Scripts (`scripts/`)

### `train.py` - Training Entrypoint

**Purpose:** Train the hierarchical model with two optional stages (Stage 1: CNN+LSTM1; Stage 2: LSTM2).

**CLI Arguments:**

- `--config`: Path to YAML config (default: "configs/default.yaml")
- `--lr`: Learning rate override (applied to both stages)
- `--batch_size`: Batch size override
- `--device`: Device override ("cuda" or "cpu")
- `--num_epochs`: Number of epochs per stage
- `--num_subgroups`: Number of subgroups (1, 2, or 4)

**Main Flow:**

1. **Parse Arguments & Load Config**

   ```
   args = parse_args()
   cfg = Config.from_yaml(args.config)
   # Apply CLI overrides
   cfg.merge({...})
   ```

2. **Set Reproducibility**

   ```
   set_seed(cfg.training.seed)
   ```

3. **Build Model**

   ```
   model = build_model(cfg)
   # Creates HierarchicalGroupActivityModel with config parameters
   ```

4. **Build Data Loaders**

   ```
   train_loader = build_loader(cfg, TRAIN_VIDEOS, train_transforms, shuffle=True, ...)
   val_loader = build_loader(cfg, VAL_VIDEOS, eval_transforms, shuffle=False, ...)
   ```

5. **Build Trainers**

   ```
   trainer_stage1 = build_trainer(cfg, model, train_loader, stage=1)
   # (Stage 2 training would use different hyperparameters)
   ```

6. **Train**

   ```
   trainer_stage1.train()
   ```

7. **Save Checkpoint** (implicit or explicit at end)

**Helper Functions:**

`set_seed(seed)`: Set seeds for random, numpy, torch (CPU and CUDA)

`build_model(cfg)`: Instantiate HierarchicalGroupActivityModel with config parameters

`build_loader(cfg, videos, transform, shuffle, batch_size)`: Create DataLoader

- Returns DataLoader with `volleyball_collate` as collate function

`build_trainer(cfg, model, loader, stage)`: Create Trainer with stage-specific hyperparameters

- Stage 1: PersonEmbedder + CNN training
- Stage 2: FrameDescriptor training (freezing Stage 1 optional)

### `evaluate.py` - Evaluation Entrypoint

**Purpose:** Evaluate a trained checkpoint on val or test split.

**CLI Arguments:**

- `--config`: Path to YAML config (default: "configs/default.yaml")
- `--checkpoint`: Path to model checkpoint (required)
- `--split`: Dataset split ("val" or "test", default: "test")
- `--device`: Device override

**Main Flow:**

1. Load config and checkpoint
2. Build model with config parameters
3. Load checkpoint weights into model
4. Build evaluation DataLoader
5. Create Evaluator
6. Run evaluation and print report

**Sample Output:**

```
======================================================================
EVALUATION RESULTS
======================================================================

Group Activity Accuracy: 72.34%  (154/213)
----------------------------------------------------------------------
  Class                    Accuracy
  ------                   ---------
  r_set                     82.50%
  r_spike                   65.00%
  ...

  Confusion Matrix (rows=truth, cols=predicted):
  [confused matrix output]
```

### `predict.py` - Single-Sample Inference

**Purpose:** Run inference on a single annotated frame to get predictions.

**CLI Arguments:**

- `--checkpoint`: Path to model checkpoint (required)
- `--video_id`: Video ID
- `--frame_id`: Frame ID within video
- `--config`: Config file (default: "configs/default.yaml")
- `--device`: Device override

**Main Flow:**

1. Load model checkpoint
2. Load sample annotations
3. Parse player list and create tensor
4. Forward pass
5. Display group activity prediction and per-player action predictions

---

## Testing (`tests/`)

Test files use pytest and validate model components in isolation and integration.

### `test_hierarchical.py` - Integration Tests

**Fixtures:**

- `model`: HierarchicalGroupActivityModel instance
- `sample`: Random tensor [N, T, C, H, W]

**Test Classes:**

`TestHierarchicalShapes`: Verify output shapes

- `test_group_logits_shape`: Verify [8] shape
- `test_person_logits_shape`: Verify [N, 9] shape
- `test_variable_N`: Test with different numbers of players
- `test_subgroup_variants`: Test 1/2/4 subgroup configurations
- `test_pool_variants`: Test max/avg pooling

(Other test files exist for individual components: PersonEmbedder, SubGroupPooler, FrameDescriptor, Dataset)

---

## Notebooks (`notebooks/`)

Interactive Jupyter notebooks for exploration and analysis.

1. **01_dataset_exploration.ipynb**: Load and visualize dataset, player bboxes, annotations
2. **02_pipeline_smoke_test.ipynb**: End-to-end forward pass validation
3. **03_results_analysis.ipynb**: Analyze training logs, plot curves, confusion matrices

---

## Data Flow Diagram

```
User Input (Video Frames + Annotations)
    ↓
VolleyballDataset.__getitem__()
    ├─ Parse annotations.txt
    ├─ Load T frames from disk
    ├─ Extract player crops (bbox clipping)
    ├─ Apply transforms (resize, normalize)
    └─ Return (x[N,T,C,H,W], group_label[1], person_labels[N])
    ↓
volleyball_collate() (custom batch collation)
    ├─ Combine multiple samples
    ├─ Return (frames_list[B], group_labels[B], person_labels_list[B])
    └─ frames_list contains variable N per sample
    ↓
HierarchicalGroupActivityModel.forward()
    ├─ Stage 1: PersonEmbedder(x)
    │   ├─ CNN: x[N,T,C,H,W] → x_cnn[N,T,D]
    │   ├─ LSTM1: x_cnn[N,T,D] → h[N,T,H]
    │   └─ Fusion: P[N,T,D+H], person_logits[N,9]
    ├─ Stage 2a: SubGroupPooler(P, subgroup_indices)
    │   └─ Pool per subgroup per frame → Z[1,T,z_dim]
    └─ Stage 2b: FrameDescriptor(Z)
        └─ LSTM2: Z[1,T,z_dim] → group_logits[8]
    ↓
Loss Computation (Multi-task)
    ├─ Group loss: CrossEntropyLoss(group_logits, group_label)
    ├─ Person loss: CrossEntropyLoss(person_logits, person_labels)
    └─ Total: group_loss + λ * person_loss
    ↓
Backward Pass & Parameter Update (SGD with momentum)
    ↓
Metrics Update (AverageMeter, MetricsTracker)
    ↓
Checkpoint Save (optional)
```

---

## Key Design Decisions

### 1. **Spatial Processing (CNN)**

- Use pretrained AlexNet for spatial feature extraction
- Freeze CNN weights during training (option in config)
- No temporal convolutions — reduces computation

### 2. **Person-Level Temporal (LSTM1)**

- Shared LSTM weights across all players
- Input: CNN features at each timestep
- Output: Temporal context vector
- Fusion at every timestep (not just final)

### 3. **Sub-group Pooling**

- Split players into contiguous left-to-right buckets
- Relies on pre-sorting by bbox x-coordinate
- Max/Avg pooling to aggregate within subgroups
- Flexible: 1 subgroup (entire team), 2 (left/right), 4 (quadrants)

### 4. **Group-Level Temporal (LSTM2)**

- Process frame descriptors (pooled subgroup vecs)
- Classify based on final hidden state
- Captures temporal dynamics of group activity

### 5. **Multi-Task Learning**

- Main task: group activity (primary supervision)
- Auxiliary task: person actions (regularization signal)
- Shared backbone ensures both tasks benefit from same representations
- Configurable loss weight (`person_loss_w`)

### 6. **Variable Batch Size Handling**

- Standard DataLoader collate fails with variable N
- Custom `volleyball_collate` returns lists where needed
- Trainer/Evaluator process samples individually within batch

### 7. **Configuration Management**

- Centralized YAML config with Python override
- Supports base config + override configs (2group, 4group)
- CLI argument overrides for quick experiments

---

## Running the Code

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python scripts/train.py

# With custom hyperparameters
python scripts/train.py --lr 3e-5 --batch_size 16 --num_epochs 50

# With 2-subgroup configuration
python scripts/train.py --num_subgroups 2
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/model_epoch_50.pt
```

### Single-Sample Prediction

```bash
python scripts/predict.py \
    --checkpoint outputs/checkpoints/model_epoch_50.pt \
    --video_id 45 \
    --frame_id 23455
```

### Running Tests

```bash
pytest tests/ -v
```

---

## Summary

This volleyball activity recognition system demonstrates a complete hierarchical deep learning pipeline for group activity understanding. The architecture elegantly combines spatial (CNN), person-level temporal (LSTM1), spatial pooling (subgroup aggregation), and group-level temporal (LSTM2) modeling.

**Key Strengths:**
✓ Modular component design (easy to modify/extend)
✓ Flexible configuration system (experiments with minimal code changes)
✓ Multi-task learning improves generalization
✓ Handles variable-size groups gracefully
✓ Comprehensive evaluation metrics and reporting

**Possible Extensions:**

- Implement Stage 2 training (LSTM2-only fine-tuning)
- Temporal data augmentation
- Attention mechanisms for dynamic subgroup assignment
- 3D convolutions for joint spatial-temporal feature learning
- Integration with off-the-shelf pose/detection models

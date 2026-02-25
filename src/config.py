"""Project configuration constants (mirrors configs/default.yaml).

These constants are used as defaults inside model constructors so
that code and the YAML stay in sync. Edit `configs/default.yaml`
and update these values if you change defaults.
"""

SEED = 42

# Dataset
DATA_ROOT = "data/"
NUM_FRAMES = 9
NUM_PLAYERS = 12
INPUT_SIZE = (224, 224)

# Training
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"

# Model defaults
CNN_OUTPUT_SIZE = 4096
LSTM_HIDDEN_P = 512
LSTM_HIDDEN_G = 512
N_SUBGROUPS = 2
POOL = "max"

# Checkpointing
CHECKPOINT_DIR = "outputs/checkpoints"
SAVE_EVERY = 5

"""Utils subpackage exports.

Expose utility helpers and metrics.

Available exports:
  - make_subgroup_indices: Compute subgroup pooling indices
  - Metrics: Loss and evaluation metrics (when implemented)
  - Checkpointing: Model save/load utilities

Usage:
    from src.utils import make_subgroup_indices
    from src.utils.metrics import accuracy_multiclass
    from src.utils.checkpointing import save_checkpoint, load_checkpoint
"""

from .subgroups import make_subgroup_indices

__all__ = [
    "make_subgroup_indices",
]

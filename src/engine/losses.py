"""
src/utils/metrics.py
--------------------
Metric tracking utilities for group activity recognition.

Provides:
  - AverageMeter   : tracks a running mean of a scalar (loss, accuracy, etc.)
  - MetricsTracker : accumulates predictions and targets, computes accuracy
                     and confusion matrix at the end of an epoch / eval pass.
"""

from __future__ import annotations

import torch


# ─────────────────────────────────────────────
# AverageMeter
# ─────────────────────────────────────────────

class AverageMeter:
    """
    Tracks the running mean of a scalar value.

    Typical use: loss or accuracy accumulated over mini-batches.

    Example
    -------
    >>> meter = AverageMeter(name="loss")
    >>> meter.update(2.4, n=8)    # value=2.4, batch_size=8
    >>> meter.update(1.8, n=8)
    >>> meter.avg
    2.1
    >>> str(meter)
    'loss: 2.1000 (avg)  last: 1.8000'
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0   # last value
        self.sum   = 0.0   # cumulative weighted sum
        self.count = 0     # cumulative sample count
        self.avg   = 0.0   # running average

    def update(self, val: float, n: int = 1) -> None:
        """
        Parameters
        ----------
        val : scalar value for this update (e.g. mean loss over the batch)
        n   : number of samples this value represents (e.g. batch size)
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}{self.avg:.4f} (avg)  last: {self.val:.4f}"


# ─────────────────────────────────────────────
# MetricsTracker
# ─────────────────────────────────────────────

class MetricsTracker:
    """
    Accumulates per-sample predictions and ground-truth targets
    over a full epoch or evaluation pass, then computes:

      - overall accuracy
      - per-class accuracy
      - confusion matrix

    One tracker per task (group activity, person action).

    Parameters
    ----------
    num_classes : int   number of output classes
    class_names : list[str]  human-readable label for each class index

    Example
    -------
    >>> tracker = MetricsTracker(8, GROUP_ACTIVITIES)
    >>> tracker.update(preds=torch.tensor([1, 0]), targets=torch.tensor([1, 1]))
    >>> tracker.accuracy()
    0.5
    >>> tracker.per_class_accuracy()
    {'r_set': 0.0, 'r_spike': 1.0, ...}
    """

    def __init__(self, num_classes: int, class_names: list[str]) -> None:
        assert len(class_names) == num_classes, (
            f"class_names length {len(class_names)} != num_classes {num_classes}"
        )
        self.num_classes  = num_classes
        self.class_names  = class_names
        self._confusion   = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def reset(self) -> None:
        """Clear all accumulated state. Call at the start of each epoch."""
        self._confusion.zero_()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate predictions for a batch or a single sample.

        Parameters
        ----------
        preds   : LongTensor [N]  predicted class indices
        targets : LongTensor [N]  ground-truth class indices
        """
        assert preds.shape == targets.shape, (
            f"preds shape {preds.shape} != targets shape {targets.shape}"
        )
        preds   = preds.cpu().view(-1)
        targets = targets.cpu().view(-1)

        # Scatter into confusion matrix
        for t, p in zip(targets.tolist(), preds.tolist()):
            self._confusion[t][p] += 1

    def accuracy(self) -> float:
        """Overall accuracy: correct / total."""
        correct = self._confusion.diagonal().sum().item()
        total   = self._confusion.sum().item()
        return correct / total if total > 0 else 0.0

    def per_class_accuracy(self) -> dict[str, float]:
        """
        Per-class accuracy: for each class c, correct_c / total_c.
        Classes with zero samples return 0.0.
        """
        out = {}
        for c, name in enumerate(self.class_names):
            total_c   = self._confusion[c].sum().item()
            correct_c = self._confusion[c][c].item()
            out[name] = correct_c / total_c if total_c > 0 else 0.0
        return out

    def confusion_matrix(self) -> torch.Tensor:
        """
        Return the raw confusion matrix as a LongTensor [C, C].
        Rows = ground truth, columns = predicted.
        """
        return self._confusion.clone()

    def summary(self) -> dict:
        """
        Return a single dict with all metrics.

        Keys
        ----
        accuracy        : float
        per_class       : dict[str, float]
        confusion_matrix: torch.Tensor [C, C]
        correct         : int
        total           : int
        """
        return {
            "accuracy":         self.accuracy(),
            "per_class":        self.per_class_accuracy(),
            "confusion_matrix": self.confusion_matrix(),
            "correct":          self._confusion.diagonal().sum().item(),
            "total":            self._confusion.sum().item(),
        }

    def pretty_confusion_matrix(self) -> str:
        """
        Return a human-readable confusion matrix string.
        Rows = ground truth, columns = predicted.
        """
        C     = self.num_classes
        names = self.class_names
        W     = max(len(n) for n in names)          # column width
        cm    = self._confusion

        header = " " * (W + 2) + "  ".join(f"{n:>{W}}" for n in names)
        rows   = [header, " " * (W + 2) + ("-" * W + "  ") * C]

        for r, name in enumerate(names):
            row_vals = "  ".join(f"{cm[r][c].item():>{W}}" for c in range(C))
            rows.append(f"{name:>{W}}  {row_vals}")

        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"MetricsTracker("
            f"num_classes={self.num_classes}, "
            f"accuracy={self.accuracy():.4f}, "
            f"total={self._confusion.sum().item()})"
        )
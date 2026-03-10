"""
src/utils/metrics.py
--------------------
Generic metric tracking utilities.

Provides:
  - AverageMeter   : tracks a running mean of any scalar (loss, accuracy, etc.)
  - MetricsTracker : accumulates predictions/targets for any classification task,
                     computes accuracy and confusion matrix at epoch end.
"""

from __future__ import annotations
import torch


class AverageMeter:
    """
    Tracks the running mean of any named scalar value.

    Example
    -------
    >>> loss_meter = AverageMeter("train/loss")
    >>> acc_meter  = AverageMeter("train/group_accuracy")
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}{self.avg:.4f} (avg)  last: {self.val:.4f}"


class MetricsTracker:
    """
    Accumulates per-sample predictions and targets for any named
    classification task, then computes accuracy and confusion matrix.

    Call update(...) for each sample, then call accuracy() and
    confusion_matrix() to retrieve metrics.

    Example
    -------
    >>> tracker = MetricsTracker("group_activity", num_classes=8)
    >>> tracker.update(pred=[7, 3, 1], target=[7, 3, 2])
    >>> tracker.accuracy()
    0.6666...
    >>> tracker.confusion_matrix()
    """

    def __init__(self, name: str, num_classes: int) -> None:
        self.name          = name
        self.num_classes   = num_classes
        self._predictions  = []
        self._targets      = []
        self._confusion    = None
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions and targets."""
        self._predictions = []
        self._targets     = []
        self._confusion   = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(
        self,
        pred: torch.Tensor | list[int],
        target: torch.Tensor | list[int],
    ) -> None:
        """
        Accumulate (prediction, target) pairs.

        Parameters
        ----------
        pred : torch.Tensor or list[int]
            Predicted class labels.  Shape [...] or scalar.
        target : torch.Tensor or list[int]
            Ground-truth class labels. Same shape as pred.
        """
        if isinstance(pred, list):
            pred = torch.tensor(pred, dtype=torch.long)
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.long)

        pred = pred.view(-1).long()
        target = target.view(-1).long()

        self._predictions.append(pred)
        self._targets.append(target)

        # Update confusion matrix
        for p, t in zip(pred, target):
            self._confusion[t, p] += 1

    def predictions(self) -> torch.Tensor:
        """Return all accumulated predictions as a single tensor."""
        if not self._predictions:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(self._predictions, dim=0)

    def targets(self) -> torch.Tensor:
        """Return all accumulated targets as a single tensor."""
        if not self._targets:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(self._targets, dim=0)

    def accuracy(self) -> float:
        """
        Compute classification accuracy.

        Returns
        -------
        float
            Fraction of correct predictions (0 to 1).
        """
        if self._confusion.sum() == 0:
            return 0.0
        correct = self._confusion.trace().item()
        total   = self._confusion.sum().item()
        return correct / total if total > 0 else 0.0

    def confusion_matrix(self) -> torch.Tensor:
        """
        Return the confusion matrix.

        Returns
        -------
        torch.Tensor
            Confusion matrix of shape (num_classes, num_classes).
            Entry [i, j] = count of samples with true class i predicted as j.
        """
        return self._confusion.clone()

    def __repr__(self) -> str:
        return (
            f"MetricsTracker(name='{self.name}', "
            f"num_classes={self.num_classes}, "
            f"accuracy={self.accuracy():.4f}, "
            f"total={self._confusion.sum().item()})"
        )

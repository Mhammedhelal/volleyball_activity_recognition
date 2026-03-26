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
    """Tracks the running mean of any named scalar value."""

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
    Accumulates per-sample predictions and targets for a named classification
    task, then computes accuracy, per-class accuracy, and confusion matrix.
    """

    def __init__(self, name: str, num_classes: int) -> None:
        self.name        = name
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._predictions: list = []
        self._targets:     list = []
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(
        self,
        pred:   "torch.Tensor | list[int]",
        target: "torch.Tensor | list[int]",
    ) -> None:
        if isinstance(pred, list):
            pred = torch.tensor(pred, dtype=torch.long)
        if isinstance(target, list):
            target = torch.tensor(target, dtype=torch.long)

        pred   = pred.view(-1).long()
        target = target.view(-1).long()

        self._predictions.append(pred)
        self._targets.append(target)

        for p, t in zip(pred, target):
            self._confusion[t, p] += 1

    # ── scalar metrics ────────────────────────────────────────────────────

    def accuracy(self) -> float:
        total = self._confusion.sum().item()
        if total == 0:
            return 0.0
        return self._confusion.trace().item() / total

    def per_class_accuracy(self) -> dict:
        cm    = self._confusion.float()
        tp    = cm.diag()
        totals = cm.sum(dim=1)
        acc   = tp / totals.clamp(min=1e-8)
        return {
            str(i): acc[i].item()
            for i in range(self.num_classes)
        }

    def accuracy_per_class(self) -> torch.Tensor:
        cm     = self._confusion.float()
        tp     = cm.diag()
        totals = cm.sum(dim=1)
        return tp / totals.clamp(min=1e-8)

    def precision(self) -> float:
        cm = self._confusion.float()
        if cm.sum() == 0:
            return 0.0
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        return (tp / (tp + fp).clamp(min=1e-8)).mean().item()

    def precision_per_class(self) -> torch.Tensor:
        cm = self._confusion.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        return tp / (tp + fp).clamp(min=1e-8)

    def recall(self) -> float:
        cm = self._confusion.float()
        if cm.sum() == 0:
            return 0.0
        tp = cm.diag()
        fn = cm.sum(dim=1) - tp
        return (tp / (tp + fn).clamp(min=1e-8)).mean().item()

    def recall_per_class(self) -> torch.Tensor:
        cm = self._confusion.float()
        tp = cm.diag()
        fn = cm.sum(dim=1) - tp
        return tp / (tp + fn).clamp(min=1e-8)

    def f1(self) -> float:
        p = self.precision_per_class()
        r = self.recall_per_class()
        return (2 * p * r / (p + r).clamp(min=1e-8)).mean().item()

    def f1_per_class(self) -> torch.Tensor:
        p = self.precision_per_class()
        r = self.recall_per_class()
        return 2 * p * r / (p + r).clamp(min=1e-8)

    def confusion_matrix(self) -> torch.Tensor:
        return self._confusion.clone()

    def predictions(self) -> torch.Tensor:
        if not self._predictions:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(self._predictions, dim=0)

    def targets(self) -> torch.Tensor:
        if not self._targets:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(self._targets, dim=0)

    # ── summary dict (used by Evaluator) ─────────────────────────────────

    def summary(self) -> dict:
        """
        Return a dict with all metrics in one call.

        Keys
        ----
        accuracy        : float
        per_class       : dict[class_name_or_idx, float]
        correct         : int
        total           : int
        confusion_matrix: Tensor [C, C]
        """
        cm      = self._confusion
        total   = cm.sum().item()
        correct = cm.trace().item()

        per_class_acc = {}
        for i in range(self.num_classes):
            row_total = cm[i].sum().item()
            per_class_acc[str(i)] = (
                cm[i, i].item() / row_total if row_total > 0 else 0.0
            )

        return {
            "accuracy":        correct / total if total > 0 else 0.0,
            "per_class":       per_class_acc,
            "correct":         int(correct),
            "total":           int(total),
            "confusion_matrix": cm.clone(),
        }

    def __repr__(self) -> str:
        return (
            f"MetricsTracker(name='{self.name}', "
            f"num_classes={self.num_classes}, "
            f"accuracy={self.accuracy():.4f}, "
            f"total={self._confusion.sum().item()})"
        )
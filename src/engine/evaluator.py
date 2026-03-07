"""
Evaluator for the hierarchical group activity model.

Computes per-class and overall accuracy for both:
  - group activity classification  (main task)
  - individual person action classification (auxiliary task)

Mirrors the evaluation protocol from the paper (Section 4):
accuracy = correct predictions / total predictions
"""


import torch

from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS
from src.utils.metrics import MetricsTracker


class Evaluator:
    """
    Evaluates a trained HierarchicalGroupActivityModel on a dataset split.

    Args:
        model       : HierarchicalGroupActivityModel
        val_loader  : DataLoader using volleyball_collate
                      yields (frames_list, group_labels, person_labels_list)
        device      : "cuda" or "cpu"
    """

    def __init__(self, model, val_loader, device: str = "cuda"):
        self.model      = model.to(device)
        self.val_loader = val_loader
        self.device     = device

        # One tracker per task
        self.group_tracker  = MetricsTracker(len(GROUP_ACTIVITIES), GROUP_ACTIVITIES)
        self.person_tracker = MetricsTracker(len(PERSON_ACTIONS),   PERSON_ACTIONS)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run a full evaluation pass.

        Returns
        -------
        dict with keys:
            group_accuracy   : float
            person_accuracy  : float
            group_per_class  : dict[str, float]
            person_per_class : dict[str, float]
            group_correct    : int
            group_total      : int
            person_correct   : int
            person_total     : int
            group_confusion  : torch.Tensor [8, 8]
            person_confusion : torch.Tensor [9, 9]
        """
        self.model.eval()

        self.group_tracker.reset()
        self.person_tracker.reset()

        for frames_list, group_labels, person_labels_list in self.val_loader:
            # frames_list        : list[B] of [N_i, T, C, H, W]
            # group_labels       : [B]
            # person_labels_list : list[B] of [N_i]

            group_labels = group_labels.to(self.device)

            for i, (frames, person_labels) in enumerate(
                zip(frames_list, person_labels_list)
            ):
                frames        = frames.to(self.device)
                person_labels = person_labels.to(self.device)

                group_logits, person_logits = self.model(frames)
                # group_logits  : [8]
                # person_logits : [N_i, 9]

                self.group_tracker.update(
                    preds   = group_logits.argmax().unsqueeze(0),  # [1]
                    targets = group_labels[i].unsqueeze(0),        # [1]
                )
                self.person_tracker.update(
                    preds   = person_logits.argmax(dim=-1),        # [N_i]
                    targets = person_labels,                       # [N_i]
                )

        group_summary  = self.group_tracker.summary()
        person_summary = self.person_tracker.summary()

        return {
            "group_accuracy":   group_summary["accuracy"],
            "person_accuracy":  person_summary["accuracy"],
            "group_per_class":  group_summary["per_class"],
            "person_per_class": person_summary["per_class"],
            "group_correct":    group_summary["correct"],
            "group_total":      group_summary["total"],
            "person_correct":   person_summary["correct"],
            "person_total":     person_summary["total"],
            "group_confusion":  group_summary["confusion_matrix"],
            "person_confusion": person_summary["confusion_matrix"],
        }

    def report(self) -> None:
        """Print a formatted evaluation report to stdout."""
        results = self.evaluate()
        W = 24

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        # ── Group activity ────────────────────────────────────────────────────
        print(
            f"\nGroup Activity Accuracy: "
            f"{results['group_accuracy'] * 100:.2f}%  "
            f"({results['group_correct']}/{results['group_total']})"
        )
        print("-" * 70)
        print(f"  {'Class':<{W}} {'Accuracy':>10}")
        print(f"  {'-'*W} {'-'*10}")
        for cls, acc in results["group_per_class"].items():
            print(f"  {cls:<{W}} {acc * 100:>9.2f}%")

        print(f"\n  Confusion Matrix (rows=truth, cols=predicted):")
        print(self.group_tracker.pretty_confusion_matrix())

        # ── Person action ─────────────────────────────────────────────────────
        print(
            f"\nPerson Action Accuracy:  "
            f"{results['person_accuracy'] * 100:.2f}%  "
            f"({results['person_correct']}/{results['person_total']})"
        )
        print("-" * 70)
        print(f"  {'Class':<{W}} {'Accuracy':>10}")
        print(f"  {'-'*W} {'-'*10}")
        for cls, acc in results["person_per_class"].items():
            print(f"  {cls:<{W}} {acc * 100:>9.2f}%")

        print(f"\n  Confusion Matrix (rows=truth, cols=predicted):")
        print(self.person_tracker.pretty_confusion_matrix())

        print("=" * 70 + "\n")
"""
src/engine/evaluator.py
------------------------
Evaluator for both the hierarchical model and all baseline models.

Uses the same INPUT_TYPE / HAS_PERSON_LOSS flags as trainer.py —
see trainer.py docstring for routing details.
"""

import torch

from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS
from src.utils.metrics import MetricsTracker

_DEFAULT_INPUT_TYPE      = "crops"
_DEFAULT_HAS_PERSON_LOSS = True


class Evaluator:
    """
    Evaluates a trained model on a dataset split.

    Args:
        model      : HierarchicalGroupActivityModel or any BaselineModel
        val_loader : DataLoader using volleyball_collate (4-tuple)
        cfg        : Config  (passed through for report formatting)
        device     : "cuda" or "cpu"
    """

    def __init__(self, model, val_loader, cfg=None, device: str = "cuda"):
        self.model      = model.to(device)
        self.val_loader = val_loader
        self.cfg        = cfg
        self.device     = device

        self.input_type      = getattr(model, "INPUT_TYPE",      _DEFAULT_INPUT_TYPE)
        self.has_person_loss = getattr(model, "HAS_PERSON_LOSS", _DEFAULT_HAS_PERSON_LOSS)

        self.group_tracker  = MetricsTracker(len(GROUP_ACTIVITIES), GROUP_ACTIVITIES)
        self.person_tracker = MetricsTracker(len(PERSON_ACTIONS),   PERSON_ACTIONS)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run a full evaluation pass. Returns metrics dict."""
        self.model.eval()
        self.group_tracker.reset()
        self.person_tracker.reset()

        for batch in self.val_loader:
            crops_list, full_frames, group_labels, person_labels_list = batch

            full_frames  = full_frames.to(self.device)
            group_labels = group_labels.to(self.device)

            for i, (crops, person_labels) in enumerate(
                zip(crops_list, person_labels_list)
            ):
                crops         = crops.to(self.device)
                person_labels = person_labels.to(self.device)

                # ── select input ─────────────────────────────────────────────
                x = full_frames[i] if self.input_type == "frame" else crops

                # ── forward ──────────────────────────────────────────────────
                if self.has_person_loss:
                    group_logits, person_logits = self.model(x)
                else:
                    group_logits  = self.model(x)
                    person_logits = None

                # ── update trackers ──────────────────────────────────────────
                g_label = group_labels[i].view(1)
                self.group_tracker.update(
                    preds   = group_logits.argmax().unsqueeze(0),
                    targets = g_label,
                )
                if self.has_person_loss and person_logits is not None:
                    self.person_tracker.update(
                        preds   = person_logits.argmax(dim=-1),
                        targets = person_labels,
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

        width = 70
        print("\n" + "=" * width)
        print("EVALUATION RESULTS")
        print("=" * width)

        g_acc     = results["group_accuracy"]
        g_correct = results["group_correct"]
        g_total   = results["group_total"]
        print(f"\nGroup Activity Accuracy: {g_acc*100:.2f}%  ({g_correct}/{g_total})")
        print("-" * width)
        print(f"  {'Class':<28}{'Accuracy':>10}")
        print(f"  {'------':<28}{'--------':>10}")
        for cls, acc in results["group_per_class"].items():
            print(f"  {cls:<28}{acc*100:>9.2f}%")

        if self.has_person_loss:
            p_acc     = results["person_accuracy"]
            p_correct = results["person_correct"]
            p_total   = results["person_total"]
            print(f"\nPerson Action Accuracy: {p_acc*100:.2f}%  ({p_correct}/{p_total})")
            print("-" * width)
            print(f"  {'Class':<28}{'Accuracy':>10}")
            print(f"  {'------':<28}{'--------':>10}")
            for cls, acc in results["person_per_class"].items():
                print(f"  {cls:<28}{acc*100:>9.2f}%")

        print("\n" + "=" * width + "\n")
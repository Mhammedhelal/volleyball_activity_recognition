"""
src/engine/evaluator.py
------------------------
Evaluator for both the hierarchical model and all baseline models.

Uses the same INPUT_TYPE / HAS_PERSON_LOSS flags as trainer.py.

DataLoader collate format (3-tuple from volleyball_collate):

    crops_data=True  (INPUT_TYPE == "crops")
        x_batch            : list[B] of Tensor [N_i, T, C, H, W]
        group_labels       : LongTensor [B]
        person_labels_list : list[B] of LongTensor [N_i]

    crops_data=False  (INPUT_TYPE == "frame")
        x_batch            : Tensor [B, T, C, H, W]
        group_labels       : LongTensor [B]
        person_labels_list : list[B] of LongTensor [N_i]
"""

import torch

from src.utils.metrics import MetricsTracker

_DEFAULT_INPUT_TYPE      = "crops"
_DEFAULT_HAS_PERSON_LOSS = True


class Evaluator:
    """
    Evaluates a trained model on a dataset split.

    Args:
        model      : HierarchicalGroupActivityModel or any BaselineModel
        val_loader : DataLoader using make_collate_fn(crops_data=...)
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

        # crops_data mirrors input_type
        self.crops_data = (self.input_type != "frame")

        # ── labels from config ────────────────────────────────────────────
        if cfg is not None:
            self.group_activities = cfg.labels.group_activities
            self.person_actions   = cfg.labels.person_actions
        else:
            # Fallback to default (for backward compatibility)
            from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS
            self.group_activities = GROUP_ACTIVITIES
            self.person_actions   = PERSON_ACTIONS

        self.group_tracker  = MetricsTracker(self.group_activities, len(self.group_activities))
        self.person_tracker = MetricsTracker(self.person_actions,   len(self.person_actions))

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run a full evaluation pass. Returns metrics dict."""
        self.model.eval()
        self.group_tracker.reset()
        self.person_tracker.reset()

        for batch in self.val_loader:
            # 3-tuple from volleyball_collate
            x_batch, group_labels, person_labels_list = batch

            group_labels = group_labels.to(self.device)   # [B]

            if self.crops_data:
                # x_batch is list[B] of [N_i, T, C, H, W]
                for i, (crops, person_labels) in enumerate(
                    zip(x_batch, person_labels_list)
                ):
                    crops         = crops.to(self.device)
                    person_labels = person_labels.to(self.device)
                    self._eval_sample(crops, group_labels[i], person_labels)
            else:
                # x_batch is Tensor [B, T, C, H, W]
                x_batch = x_batch.to(self.device)
                for i, person_labels in enumerate(person_labels_list):
                    person_labels = person_labels.to(self.device)
                    self._eval_sample(x_batch[i], group_labels[i], person_labels)

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

    def _eval_sample(
        self,
        x:             torch.Tensor,   # [N, T, C, H, W] or [T, C, H, W]
        group_label:   torch.Tensor,   # scalar
        person_labels: torch.Tensor,   # [N]
    ) -> None:
        if self.has_person_loss:
            group_logits, person_logits = self.model(x)
        else:
            group_logits  = self.model(x)
            person_logits = None

        g_label = group_label.view(1)
        self.group_tracker.update(
            preds   = group_logits.argmax().unsqueeze(0),
            targets = g_label,
        )
        if self.has_person_loss and person_logits is not None:
            self.person_tracker.update(
                preds   = person_logits.argmax(dim=-1),
                targets = person_labels,
            )

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
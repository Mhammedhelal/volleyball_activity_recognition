"""
Evaluator for the hierarchical group activity model.

Computes per-class and overall accuracy for both:
  - group activity classification  (main task)
  - individual person action classification (auxiliary task)

Mirrors the evaluation protocol from the paper (Section 4):
accuracy = correct predictions / total predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labels import GROUP_ACTIVITIES, PERSON_ACTIONS


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

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run a full evaluation pass.

        Returns a dict with:
            group_accuracy          : float   overall group activity accuracy
            person_accuracy         : float   overall person action accuracy
            group_per_class         : dict    {class_name: accuracy}
            person_per_class        : dict    {class_name: accuracy}
            group_correct           : int
            group_total             : int
            person_correct          : int
            person_total            : int
        """
        self.model.eval()

        # Per-class counters
        n_group  = len(GROUP_ACTIVITIES)
        n_person = len(PERSON_ACTIONS)

        group_correct_per_class  = torch.zeros(n_group,  dtype=torch.long)
        group_total_per_class    = torch.zeros(n_group,  dtype=torch.long)
        person_correct_per_class = torch.zeros(n_person, dtype=torch.long)
        person_total_per_class   = torch.zeros(n_person, dtype=torch.long)

        for frames_list, group_labels, person_labels_list in self.val_loader:
            # frames_list        : list[B] of [N_i, T, C, H, W]
            # group_labels       : [B]
            # person_labels_list : list[B] of [N_i]

            group_labels = group_labels.to(self.device)   # [B]

            for i, (frames, person_labels) in enumerate(
                zip(frames_list, person_labels_list)
            ):
                frames        = frames.to(self.device)         # [N_i, T, C, H, W]
                person_labels = person_labels.to(self.device)  # [N_i]

                group_logits, person_logits = self.model(frames)
                # group_logits  : [8]
                # person_logits : [N_i, 9]

                # ── Group prediction ─────────────────────────────────────────
                group_pred  = group_logits.argmax()            # scalar
                group_truth = group_labels[i]                  # scalar

                group_total_per_class[group_truth] += 1
                if group_pred == group_truth:
                    group_correct_per_class[group_truth] += 1

                # ── Person predictions ───────────────────────────────────────
                person_preds = person_logits.argmax(dim=-1)    # [N_i]

                for pred, truth in zip(person_preds, person_labels):
                    person_total_per_class[truth] += 1
                    if pred == truth:
                        person_correct_per_class[truth] += 1

        # ── Aggregate ────────────────────────────────────────────────────────
        group_correct  = group_correct_per_class.sum().item()
        group_total    = group_total_per_class.sum().item()
        person_correct = person_correct_per_class.sum().item()
        person_total   = person_total_per_class.sum().item()

        group_accuracy  = group_correct  / group_total  if group_total  > 0 else 0.0
        person_accuracy = person_correct / person_total if person_total > 0 else 0.0

        # ── Per-class accuracy ────────────────────────────────────────────────
        group_per_class = {
            GROUP_ACTIVITIES[c]: (
                group_correct_per_class[c].item() /
                group_total_per_class[c].item()
            ) if group_total_per_class[c] > 0 else 0.0
            for c in range(n_group)
        }

        person_per_class = {
            PERSON_ACTIONS[c]: (
                person_correct_per_class[c].item() /
                person_total_per_class[c].item()
            ) if person_total_per_class[c] > 0 else 0.0
            for c in range(n_person)
        }

        return {
            "group_accuracy":   group_accuracy,
            "person_accuracy":  person_accuracy,
            "group_per_class":  group_per_class,
            "person_per_class": person_per_class,
            "group_correct":    group_correct,
            "group_total":      group_total,
            "person_correct":   person_correct,
            "person_total":     person_total,
        }

    def report(self) -> None:
        """Print a formatted evaluation report to stdout."""
        results = self.evaluate()

        w = 24   # column width

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        # ── Group activity ────────────────────────────────────────────────────
        print(f"\nGroup Activity Accuracy: "
              f"{results['group_accuracy'] * 100:.2f}%  "
              f"({results['group_correct']}/{results['group_total']})")
        print("-" * 70)
        print(f"  {'Class':<{w}} {'Accuracy':>10}")
        print(f"  {'-'*w} {'-'*10}")
        for cls, acc in results["group_per_class"].items():
            print(f"  {cls:<{w}} {acc * 100:>9.2f}%")

        # ── Person action ─────────────────────────────────────────────────────
        print(f"\nPerson Action Accuracy:  "
              f"{results['person_accuracy'] * 100:.2f}%  "
              f"({results['person_correct']}/{results['person_total']})")
        print("-" * 70)
        print(f"  {'Class':<{w}} {'Accuracy':>10}")
        print(f"  {'-'*w} {'-'*10}")
        for cls, acc in results["person_per_class"].items():
            print(f"  {cls:<{w}} {acc * 100:>9.2f}%")

        print("=" * 70 + "\n")
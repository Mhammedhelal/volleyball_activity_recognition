"""
src/engine/trainer.py
----------------------
Training loop for both the hierarchical model and all baseline models.

Routing logic
-------------
The trainer inspects two class-level flags on the model:

    model.INPUT_TYPE      "frame"  → feed full_frames  [T, C, H, W]  per sample
                          "crops"  → feed crops         [N, T, C, H, W]  per sample

    model.HAS_PERSON_LOSS True  → model returns (group_logits, person_logits)
                                  and person_labels are used in the aux loss
                          False → model returns group_logits only
                                  (person_labels are ignored)

The full HierarchicalGroupActivityModel always has:
    INPUT_TYPE      = "crops"
    HAS_PERSON_LOSS = True

All baselines have HAS_PERSON_LOSS = False.
"""

import torch
import torch.nn as nn
from typing import Iterable

from src.utils.metrics import AverageMeter, MetricsTracker
from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS

# Sentinel values for models that don't declare the flags
# (keeps backwards-compatibility with the full hierarchical model)
_DEFAULT_INPUT_TYPE      = "crops"
_DEFAULT_HAS_PERSON_LOSS = True


class Trainer:
    """
    Unified trainer for hierarchical model and all baseline models.

    DataLoader must use volleyball_collate, which now yields 4-tuples:
        crops_list         : list[B] of [N_i, T, C, H, W]
        full_frames        : [B, T, C, H, W]
        group_labels       : [B]
        person_labels_list : list[B] of [N_i]

    Args:
        model          : HierarchicalGroupActivityModel or any BaselineModel
        params         : parameters the optimizer should update
        train_loader   : DataLoader using volleyball_collate
        device         : "cuda" or "cpu"
        learning_rate  : default 1e-5 (paper value)
        momentum       : default 0.9  (paper value)
        num_epochs     : total training epochs
        person_loss_w  : weight of auxiliary person-action loss (ignored for baselines)
        log_every      : print summary every N epochs
    """

    def __init__(
        self,
        model,
        params:         Iterable[nn.Parameter],
        train_loader,
        device:         str   = "cuda",
        learning_rate:  float = 1e-5,
        momentum:       float = 0.9,
        num_epochs:     int   = 100,
        person_loss_w:  float = 1.0,
        log_every:      int   = 10,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.device         = device
        self.num_epochs     = num_epochs
        self.person_loss_w  = person_loss_w
        self.log_every      = log_every

        # Read routing flags with safe defaults for the full hierarchical model
        self.input_type      = getattr(model, "INPUT_TYPE",      _DEFAULT_INPUT_TYPE)
        self.has_person_loss = getattr(model, "HAS_PERSON_LOSS", _DEFAULT_HAS_PERSON_LOSS)

        self.optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
        )

        self.criterion_group   = nn.CrossEntropyLoss()
        self.criterion_players = nn.CrossEntropyLoss()

        self.loss_meter     = AverageMeter(name="loss")
        self.group_tracker  = MetricsTracker(len(GROUP_ACTIVITIES), GROUP_ACTIVITIES)
        self.person_tracker = MetricsTracker(len(PERSON_ACTIONS),   PERSON_ACTIONS)

    # ─────────────────────────────────────────────────────────────────────────

    def _forward_sample(
        self,
        crops:         torch.Tensor,    # [N, T, C, H, W]
        full_frame:    torch.Tensor,    # [T, C, H, W]
        group_label:   torch.Tensor,    # scalar or [1]
        person_labels: torch.Tensor,    # [N]
    ) -> torch.Tensor:
        """
        Run one sample forward, compute loss, update trackers.
        Returns the scalar loss for this sample (not yet divided by batch size).
        """
        # ── select input tensor ───────────────────────────────────────────────
        if self.input_type == "frame":
            x = full_frame                      # [T, C, H, W]
        else:
            x = crops                           # [N, T, C, H, W]

        # ── forward pass ──────────────────────────────────────────────────────
        if self.has_person_loss:
            group_logits, person_logits = self.model(x)
        else:
            group_logits = self.model(x)
            person_logits = None

        # ── group loss ────────────────────────────────────────────────────────
        g_label = group_label.view(1) if group_label.dim() == 0 else group_label
        loss = self.criterion_group(
            group_logits.unsqueeze(0),  # [1, C]
            g_label,                    # [1]
        )

        # ── optional person loss ──────────────────────────────────────────────
        if self.has_person_loss and person_logits is not None:
            loss = loss + self.person_loss_w * self.criterion_players(
                person_logits,   # [N, P]
                person_labels,   # [N]
            )

        # ── update trackers (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            self.group_tracker.update(
                preds   = group_logits.argmax().unsqueeze(0),
                targets = g_label,
            )
            if self.has_person_loss and person_logits is not None:
                self.person_tracker.update(
                    preds   = person_logits.argmax(dim=-1),
                    targets = person_labels,
                )

        return loss

    # ─────────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> dict:
        """Run one full pass over the training set."""
        self.model.train()
        self.loss_meter.reset()
        self.group_tracker.reset()
        self.person_tracker.reset()

        for batch in self.train_loader:
            crops_list, full_frames, group_labels, person_labels_list = batch

            full_frames  = full_frames.to(self.device)   # [B, T, C, H, W]
            group_labels = group_labels.to(self.device)  # [B]

            batch_loss = torch.tensor(0.0, device=self.device)

            for i, (crops, person_labels) in enumerate(
                zip(crops_list, person_labels_list)
            ):
                crops         = crops.to(self.device)           # [N_i, T, C, H, W]
                person_labels = person_labels.to(self.device)   # [N_i]
                frame_i       = full_frames[i]                  # [T, C, H, W]

                batch_loss = batch_loss + self._forward_sample(
                    crops         = crops,
                    full_frame    = frame_i,
                    group_label   = group_labels[i],
                    person_labels = person_labels,
                )

            batch_loss = batch_loss / len(crops_list)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            self.loss_meter.update(batch_loss.item(), n=len(crops_list))

        return {
            "loss":            self.loss_meter.avg,
            "group_accuracy":  self.group_tracker.accuracy(),
            "person_accuracy": self.person_tracker.accuracy(),
        }

    def train(self) -> None:
        """Train for num_epochs epochs."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch()

            if (epoch + 1) % self.log_every == 0:
                msg = (
                    f"Epoch [{epoch+1:>4}/{self.num_epochs}]  "
                    f"Loss: {metrics['loss']:.4f}  "
                    f"Group Acc: {metrics['group_accuracy']*100:.2f}%"
                )
                if self.has_person_loss:
                    msg += f"  Person Acc: {metrics['person_accuracy']*100:.2f}%"
                print(msg)

        print("\n✅ Training completed!")
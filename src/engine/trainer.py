"""
src/engine/trainer.py
----------------------
Training loop for both the hierarchical model and all baseline models.

Routing logic
-------------
The trainer inspects two class-level flags on the model:

    model.INPUT_TYPE      "frame"  → feed full frames  [T, C, H, W]  per sample
                          "crops"  → feed crops         [N, T, C, H, W]  per sample

    model.HAS_PERSON_LOSS True  → model returns (group_logits, person_logits)
                                  and person_labels are used in the aux loss
                          False → model returns group_logits only
                                  (person_labels are ignored)

The full HierarchicalGroupActivityModel always has:
    INPUT_TYPE      = "crops"
    HAS_PERSON_LOSS = True

All baselines have HAS_PERSON_LOSS = False.

DataLoader collate format
-------------------------
volleyball_collate now returns a **3-tuple**:

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
import torch.nn as nn
from typing import Iterable

from src.utils.metrics import AverageMeter, MetricsTracker
from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS

# Sentinel defaults (backwards-compat with full hierarchical model)
_DEFAULT_INPUT_TYPE      = "crops"
_DEFAULT_HAS_PERSON_LOSS = True


class Trainer:
    """
    Unified trainer for hierarchical model and all baseline models.

    Args:
        model          : HierarchicalGroupActivityModel or any BaselineModel
        params         : parameters the optimizer should update
        train_loader   : DataLoader using make_collate_fn(crops_data=...)
        device         : "cuda" or "cpu"
        learning_rate  : default 1e-5 (paper value)
        momentum       : default 0.9  (paper value)
        num_epochs     : total training epochs
        person_loss_w  : weight of auxiliary person-action loss
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

        # Read routing flags with safe defaults
        self.input_type      = getattr(model, "INPUT_TYPE",      _DEFAULT_INPUT_TYPE)
        self.has_person_loss = getattr(model, "HAS_PERSON_LOSS", _DEFAULT_HAS_PERSON_LOSS)

        # crops_data mirrors input_type: frame-level models use stacked full frames
        self.crops_data = (self.input_type != "frame")

        self.optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
        )

        self.criterion_group   = nn.CrossEntropyLoss()
        self.criterion_players = nn.CrossEntropyLoss()

        self.loss_meter     = AverageMeter(name="loss")
        self.group_tracker  = MetricsTracker(GROUP_ACTIVITIES, len(GROUP_ACTIVITIES))
        self.person_tracker = MetricsTracker(PERSON_ACTIONS, len(PERSON_ACTIONS))

    # ------------------------------------------------------------------

    def _forward_sample(
        self,
        x:             torch.Tensor,   # [N, T, C, H, W]  OR  [T, C, H, W]
        group_label:   torch.Tensor,   # scalar or [1]
        person_labels: torch.Tensor,   # [N]
    ) -> torch.Tensor:
        """
        Run one sample forward, compute loss, update trackers.
        Returns the scalar loss for this sample.
        """
        # ── forward pass ──────────────────────────────────────────────────
        if self.has_person_loss:
            group_logits, person_logits = self.model(x)
        else:
            group_logits  = self.model(x)
            person_logits = None

        # ── group loss ────────────────────────────────────────────────────
        g_label = group_label.view(1) if group_label.dim() == 0 else group_label
        loss = self.criterion_group(
            group_logits.unsqueeze(0),   # [1, C]
            g_label,                     # [1]
        )

        # ── optional person loss ──────────────────────────────────────────
        if self.has_person_loss and person_logits is not None:
            loss = loss + self.person_loss_w * self.criterion_players(
                person_logits,   # [N, P]
                person_labels,   # [N]
            )

        # ── update trackers (no grad) ─────────────────────────────────────
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

    # ------------------------------------------------------------------

    def train_epoch(self) -> dict:
        """Run one full pass over the training set."""
        self.model.train()
        self.loss_meter.reset()
        self.group_tracker.reset()
        self.person_tracker.reset()

        for batch in self.train_loader:
            # 3-tuple from volleyball_collate
            x_batch, group_labels, person_labels_list = batch

            group_labels = group_labels.to(self.device)   # [B]
            batch_loss   = torch.tensor(0.0, device=self.device)

            if self.crops_data:
                # x_batch is list[B] of [N_i, T, C, H, W]
                batch_size = len(x_batch)
                for i, (crops, person_labels) in enumerate(
                    zip(x_batch, person_labels_list)
                ):
                    crops         = crops.to(self.device)
                    person_labels = person_labels.to(self.device)
                    batch_loss    = batch_loss + self._forward_sample(
                        x             = crops,
                        group_label   = group_labels[i],
                        person_labels = person_labels,
                    )
            else:
                # x_batch is Tensor [B, T, C, H, W]
                x_batch    = x_batch.to(self.device)
                batch_size = x_batch.shape[0]
                for i, person_labels in enumerate(person_labels_list):
                    person_labels = person_labels.to(self.device)
                    batch_loss    = batch_loss + self._forward_sample(
                        x             = x_batch[i],   # [T, C, H, W]
                        group_label   = group_labels[i],
                        person_labels = person_labels,
                    )

            batch_loss = batch_loss / batch_size

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            self.loss_meter.update(batch_loss.item(), n=batch_size)

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
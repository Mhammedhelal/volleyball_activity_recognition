"""
Training loop and trainer utilities for the hierarchical group activity model.
"""

import torch
import torch.nn as nn
from typing import Iterable

from src.utils.metrics import AverageMeter, MetricsTracker
from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS


class Trainer:
    """
    Trainer for the two-stage hierarchical group activity model.

    Because N (number of players) varies per sample, volleyball_collate
    returns lists rather than a single stacked tensor. This trainer
    processes each sample in the batch individually, accumulates the
    multi-task loss, and updates once per batch.

    Args:
        model          : HierarchicalGroupActivityModel
        params         : iterable of parameters the optimizer should update.
                         Stage 1 → person_embedder params only
                         Stage 2 → subgroup_pooler + frame_descriptor params only
        train_loader   : DataLoader using volleyball_collate
                         yields (frames_list, group_labels, person_labels_list)
                           frames_list        list[B] of [N_i, T, C, H, W]
                           group_labels       [B]
                           person_labels_list list[B] of [N_i]
        device         : "cuda" or "cpu"
        learning_rate  : paper uses 1e-5
        momentum       : paper uses 0.9
        num_epochs     : total training epochs
        person_loss_w  : weight of the auxiliary person-action loss
        log_every      : print a summary every N epochs
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

        # Only the params for the active stage are passed in — the optimizer
        # never sees (and never updates) the frozen stage's parameters.
        self.optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
        )

        self.criterion_group   = nn.CrossEntropyLoss()
        self.criterion_players = nn.CrossEntropyLoss()

        # Metric trackers — reset each epoch inside train_epoch()
        self.loss_meter     = AverageMeter(name="loss")
        self.group_tracker  = MetricsTracker(name="group_activity", num_classes=len(GROUP_ACTIVITIES))
        self.person_tracker = MetricsTracker(name="person_action", num_classes=len(PERSON_ACTIONS))

    def train_epoch(self) -> dict:
        """
        Run one full pass over the training set.

        Returns
        -------
        dict with keys:
            loss            : float   mean total loss over the epoch
            group_accuracy  : float   group activity training accuracy
            person_accuracy : float   person action training accuracy
        """
        self.model.train()

        # Reset all meters at the start of each epoch
        self.loss_meter.reset()
        self.group_tracker.reset()
        self.person_tracker.reset()

        for frames_list, group_labels, person_labels_list in self.train_loader:
            group_labels = group_labels.to(self.device)   # [B]

            batch_loss = torch.tensor(0.0, device=self.device)

            for i, (frames, person_labels) in enumerate(
                zip(frames_list, person_labels_list)
            ):
                frames        = frames.to(self.device)         # [N_i, T, C, H, W]
                person_labels = person_labels.to(self.device)  # [N_i]

                group_logits, person_logits = self.model(frames)
                # group_logits  : [8]
                # person_logits : [N_i, 9]

                group_loss = self.criterion_group(
                    group_logits.unsqueeze(0),     # [1, 8]
                    group_labels[i].unsqueeze(0),  # [1]
                )
                person_loss = self.criterion_players(
                    person_logits,                 # [N_i, 9]
                    person_labels,                 # [N_i]
                )

                batch_loss = batch_loss + group_loss + self.person_loss_w * person_loss

                # Accumulate predictions into trackers (detached — no grad needed)
                with torch.no_grad():
                    self.group_tracker.update(
                        pred   = group_logits.argmax().unsqueeze(0),  # [1]
                        target = group_labels[i].unsqueeze(0),        # [1]
                    )
                    self.person_tracker.update(
                        pred   = person_logits.argmax(dim=-1),        # [N_i]
                        target = person_labels,                       # [N_i]
                    )

            # Average loss over the batch then step
            batch_loss = batch_loss / len(frames_list)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            self.loss_meter.update(batch_loss.item(), n=len(frames_list))

        return {
            "loss":            self.loss_meter.avg,
            "group_accuracy":  self.group_tracker.accuracy(),
            "person_accuracy": self.person_tracker.accuracy(),
        }

    def train(self) -> None:
        """Train for num_epochs."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            metrics = self.train_epoch()

            if (epoch + 1) % self.log_every == 0:
                print(
                    f"Epoch [{epoch+1:>4}/{self.num_epochs}]  "
                    f"Loss: {metrics['loss']:.4f}  "
                    f"Group Acc: {metrics['group_accuracy']*100:.2f}%  "
                    f"Person Acc: {metrics['person_accuracy']*100:.2f}%"
                )

        print("\n✅ Training completed!")
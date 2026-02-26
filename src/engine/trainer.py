"""
Training loop and trainer utilities for the hierarchical group activity model.
"""

import torch
import torch.nn as nn


class Trainer:
    """
    Trainer for the two-stage hierarchical group activity model.

    Because N (number of players) varies per sample, volleyball_collate
    returns lists rather than a single stacked tensor. This trainer
    processes each sample in the batch individually, accumulates the
    multi-task loss, and updates once per batch.

    Args:
        model          : HierarchicalGroupActivityModel
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
    """

    def __init__(
        self,
        model,
        train_loader,
        device:         str   = "cuda",
        learning_rate:  float = 1e-5,
        momentum:       float = 0.9,
        num_epochs:     int   = 100,
        person_loss_w:  float = 1.0,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.device         = device
        self.num_epochs     = num_epochs
        self.person_loss_w  = person_loss_w

        # Paper: SGD with fixed LR=1e-5 and momentum=0.9
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )

        self.criterion_group   = nn.CrossEntropyLoss()
        self.criterion_players = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        """Run one full pass over the training set.

        Returns
          mean total loss over all batches.
        """
        self.model.train()
        epoch_loss = 0.0

        for frames_list, group_labels, person_labels_list in self.train_loader:
            # frames_list        : list[B] of [N_i, T, C, H, W]
            # group_labels       : [B]       — one group label per sample
            # person_labels_list : list[B] of [N_i]

            group_labels = group_labels.to(self.device)

            batch_loss = torch.tensor(0.0, device=self.device)

            for i, (frames, person_labels) in enumerate(
                zip(frames_list, person_labels_list)
            ):
                frames        = frames.to(self.device)         # [N_i, T, C, H, W]
                person_labels = person_labels.to(self.device)  # [N_i]

                group_logits, person_logits = self.model(frames)
                # group_logits  : [8]
                # person_logits : [N_i, 9]

                # group_logits unsqueezed to [1, 8] to give CrossEntropyLoss
                # a batch dimension; group_labels[i] is a scalar → [1]
                group_loss = self.criterion_group(
                    group_logits.unsqueeze(0),          # [1, 8]
                    group_labels[i].unsqueeze(0),       # [1]
                )
                person_loss = self.criterion_players(
                    person_logits,                      # [N_i, 9]
                    person_labels,                      # [N_i]
                )

                batch_loss = batch_loss + group_loss + self.person_loss_w * person_loss

            # Average over batch before stepping
            batch_loss = batch_loss / len(frames_list)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

        return epoch_loss / len(self.train_loader)

    def train(self):
        """Train for num_epochs and return the trained model."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            epoch_loss = self.train_epoch()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}]  Loss: {epoch_loss:.4f}")

        print("\n✅ Training completed!")
        return self.model
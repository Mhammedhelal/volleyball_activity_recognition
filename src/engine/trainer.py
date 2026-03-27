"""
src/engine/trainer.py
----------------------
Training loop for both the hierarchical model and all baseline models.

Routing logic
-------------
    model.INPUT_TYPE      "frame"  → feed full frames  [T, C, H, W]  per sample
                          "crops"  → feed crops         [N, T, C, H, W]  per sample

    model.HAS_PERSON_LOSS True  → model returns (group_logits, person_logits)
                          False → model returns group_logits only

DataLoader collate format (3-tuple from volleyball_collate):

    crops_data=True
        x_batch            : list[B] of Tensor [N_i, T, C, H, W]
        group_labels       : LongTensor [B]
        person_labels_list : list[B] of LongTensor [N_i]

    crops_data=False
        x_batch            : Tensor [B, T, C, H, W]
        group_labels       : LongTensor [B]
        person_labels_list : list[B] of LongTensor [N_i]
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Iterable

from src.utils.metrics import AverageMeter, MetricsTracker
from src.data.labels import GROUP_ACTIVITIES, PERSON_ACTIONS

_DEFAULT_INPUT_TYPE      = "crops"
_DEFAULT_HAS_PERSON_LOSS = True


class Trainer:
    """
    Unified trainer for hierarchical model and all baseline models.

    Args:
        model          : HierarchicalGroupActivityModel or any BaselineModel
        params         : parameters the optimizer should update
        train_loader   : DataLoader using make_collate_fn(crops_data=...)
        val_loader     : optional DataLoader for per-epoch validation
        device         : "cuda" or "cpu"  (auto-detected if not provided)
        learning_rate  : default 1e-5 (paper value)
        momentum       : default 0.9  (paper value)
        num_epochs     : total training epochs
        person_loss_w  : weight of auxiliary person-action loss
        grad_clip      : max gradient norm (0 = disabled)
        log_every      : print summary every N epochs
        checkpoint_dir : directory to save best model checkpoint
        model_name     : prefix for checkpoint filenames
        stage          : training stage (1 or 2), used in checkpoint name
    """

    def __init__(
        self,
        model,
        params:         Iterable[nn.Parameter],
        train_loader,
        val_loader      = None,
        device:         str   = None,
        learning_rate:  float = 1e-5,
        momentum:       float = 0.9,
        num_epochs:     int   = 100,
        person_loss_w:  float = 1.0,
        grad_clip:      float = 5.0,
        log_every:      int   = 10,
        checkpoint_dir: str   = "outputs/checkpoints",
        model_name:     str   = "model",
        stage:          int   = 1,
    ):
        # ── device (auto-detect if not provided) ─────────────────────────
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.num_epochs     = num_epochs
        self.person_loss_w  = person_loss_w
        self.grad_clip      = grad_clip
        self.log_every      = log_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name     = model_name
        self.stage          = stage

        # ── model routing flags ───────────────────────────────────────────
        self.input_type      = getattr(model, "INPUT_TYPE",      _DEFAULT_INPUT_TYPE)
        self.has_person_loss = getattr(model, "HAS_PERSON_LOSS", _DEFAULT_HAS_PERSON_LOSS)
        self.crops_data      = (self.input_type != "frame")

        # ── optimiser ────────────────────────────────────────────────────
        self.optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
        )

        self.criterion_group   = nn.CrossEntropyLoss()
        self.criterion_players = nn.CrossEntropyLoss()

        # ── meters & trackers ─────────────────────────────────────────────
        self.loss_meter     = AverageMeter(name="loss")
        self.group_tracker  = MetricsTracker(GROUP_ACTIVITIES, len(GROUP_ACTIVITIES))
        self.person_tracker = MetricsTracker(PERSON_ACTIONS,   len(PERSON_ACTIONS))

        # ── best-model tracking ───────────────────────────────────────────
        self.best_val_group_acc = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_sample(
        self,
        x:             torch.Tensor,
        group_label:   torch.Tensor,
        person_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one sample, compute combined loss, update trackers."""
        if self.has_person_loss:
            group_logits, person_logits = self.model(x)
        else:
            group_logits  = self.model(x)
            person_logits = None

        g_label = group_label.view(1) if group_label.dim() == 0 else group_label
        loss = self.criterion_group(
            group_logits.unsqueeze(0),
            g_label,
        )

        if self.has_person_loss and person_logits is not None:
            loss = loss + self.person_loss_w * self.criterion_players(
                person_logits,
                person_labels,
            )

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

    def _run_loader(self, loader, train: bool) -> dict:
        """Run one full pass over *loader*. Backprop only when train=True."""
        if train:
            self.model.train()
        else:
            self.model.eval()

        self.loss_meter.reset()
        self.group_tracker.reset()
        self.person_tracker.reset()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                x_batch, group_labels, person_labels_list = batch
                group_labels = group_labels.to(self.device)
                batch_loss   = torch.tensor(0.0, device=self.device)

                if self.crops_data:
                    batch_size = len(x_batch)
                    for i, (crops, person_labels) in enumerate(
                        zip(x_batch, person_labels_list)
                    ):
                        crops         = crops.to(self.device)
                        person_labels = person_labels.to(self.device)
                        batch_loss   += self._forward_sample(crops, group_labels[i], person_labels)
                else:
                    x_batch    = x_batch.to(self.device)
                    batch_size = x_batch.shape[0]
                    for i, person_labels in enumerate(person_labels_list):
                        person_labels = person_labels.to(self.device)
                        batch_loss   += self._forward_sample(x_batch[i], group_labels[i], person_labels)

                batch_loss = batch_loss / batch_size

                if train:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

                self.loss_meter.update(batch_loss.item(), n=batch_size)

        return {
            "loss":            self.loss_meter.avg,
            "group_accuracy":  self.group_tracker.accuracy(),
            "person_accuracy": self.person_tracker.accuracy(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict:
        """Run one full training epoch."""
        return self._run_loader(self.train_loader, train=True)

    def validate(self) -> dict:
        """Run one full validation pass (no gradients)."""
        if self.val_loader is None:
            return {}
        return self._run_loader(self.val_loader, train=False)

    def _save_checkpoint(self, epoch: int, tag: str) -> Path:
        """Save full checkpoint (model + optimizer state + metadata)."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"{self.model_name}_stage{self.stage}_{tag}.pt"
        torch.save(
            {
                "epoch":     epoch,
                "stage":     self.stage,
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        return path

    def train(self) -> None:
        """Train for num_epochs epochs, validate each epoch, save best model."""
        print("\n" + "=" * 70)
        print(f"STARTING TRAINING  —  stage {self.stage}  —  device: {self.device}")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch()
            val_metrics   = self.validate()

            # ── save best model based on val group accuracy ───────────────
            val_acc = val_metrics.get("group_accuracy", 0.0)
            if val_acc > self.best_val_group_acc:
                self.best_val_group_acc = val_acc
                best_path = self._save_checkpoint(epoch + 1, "best")
                print(f"  ✔ New best val accuracy: {val_acc*100:.2f}%  → {best_path.name}")

            # ── periodic console logging ──────────────────────────────────
            if (epoch + 1) % self.log_every == 0:
                msg = (
                    f"Epoch [{epoch+1:>4}/{self.num_epochs}]  "
                    f"Loss: {train_metrics['loss']:.4f}  "
                    f"Train Grp: {train_metrics['group_accuracy']*100:.2f}%"
                )
                if self.has_person_loss:
                    msg += f"  Train Prs: {train_metrics['person_accuracy']*100:.2f}%"
                if val_metrics:
                    msg += f"  Val Grp: {val_metrics['group_accuracy']*100:.2f}%"
                print(msg)

        # ── save final checkpoint ─────────────────────────────────────────
        final_path = self._save_checkpoint(self.num_epochs, "final")
        print(f"\n✅ Training completed!  Final checkpoint → {final_path.name}")
        if self.val_loader is not None:
            print(f"   Best val group accuracy: {self.best_val_group_acc*100:.2f}%")
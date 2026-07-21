"""
src/utils/embedding_cache.py
-----------------------------
Cache Stage-1 person embeddings (P) to disk so a standalone Stage-2 run can
skip the frozen CNN + LSTM1 forward pass entirely.

Each cached sample is one .pt file:
    P             : [N, T, D+H]   person_embedder output
    group_label   : [1]
    person_labels : [N]
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import make_collate_fn


@torch.no_grad()
def cache_person_embeddings(model, loader, device: str, cache_dir: Path) -> int:
    """
    Run model.person_embedder over every sample in *loader* and write P (+
    labels) to *cache_dir* as one .pt file per sample. Wipes *cache_dir*
    first so a stale cache from a previous config never leaks into a new run.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model.eval()   # deterministic (dropout off) — cached once, reused every epoch

    idx = 0
    for x_batch, group_labels, person_labels_list in loader:
        for i, (crops, person_labels) in enumerate(zip(x_batch, person_labels_list)):
            crops = crops.to(device)
            _, P  = model.person_embedder(crops)              # [N, T, D+H]

            torch.save(
                {
                    "P":             P.cpu(),
                    "group_label":   group_labels[i].view(1).cpu(),
                    "person_labels": person_labels.cpu(),
                },
                cache_dir / f"{idx}.pt",
            )
            idx += 1

    return idx


class PersonEmbeddingDataset(Dataset):
    """Reads (P, group_label, person_labels) tuples written by cache_person_embeddings()."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"), key=lambda p: int(p.stem))
        if not self.files:
            raise FileNotFoundError(
                f"No cached embeddings found in {self.cache_dir}. "
                "Run Stage 1 first (it caches embeddings automatically) before Stage 2."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx])
        return data["P"], data["group_label"], data["person_labels"]


def build_embedding_loader(
    cache_dir:   Path,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int = 0,
    pin_memory:  bool = False,
) -> DataLoader:
    dataset = PersonEmbeddingDataset(cache_dir)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = make_collate_fn(crops_data=True),   # P varies in N, same as crops
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
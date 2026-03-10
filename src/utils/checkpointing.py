"""
src/utils/checkpointing.py
--------------------------
Utilities for saving, loading, and discovering model checkpoints.

Public API
----------
    save_checkpoint(state, filepath)
    load_checkpoint(filepath, model, optimizer=None) -> dict
    get_latest_checkpoint(checkpoint_dir)             -> str | None
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str) -> None:
    """Serialize *state* to *filepath*, creating parent directories as needed.

    The caller decides what goes into *state*.  A typical call looks like:

        save_checkpoint(
            state={
                "epoch":      epoch,
                "stage":      1,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            },
            filepath="models/stage1.pt",
        )

    Parameters
    ----------
    state : dict
        Arbitrary dictionary to persist (must be torch-serialisable).
    filepath : str
        Destination path, e.g. ``"models/stage1.pt"``.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"✔  Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    filepath:  str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a checkpoint from *filepath* into *model* (and optionally *optimizer*).

    Parameters
    ----------
    filepath : str
        Path to the ``.pt`` checkpoint file.
    model : nn.Module
        Model whose weights will be restored from ``state["model"]``.
    optimizer : torch.optim.Optimizer, optional
        If provided, its state is restored from ``state["optimizer"]``
        when that key is present in the checkpoint.

    Returns
    -------
    dict
        The full checkpoint dictionary (contains at least ``"model"``; may
        also contain ``"epoch"``, ``"stage"``, ``"optimizer"``, etc.).

    Raises
    ------
    FileNotFoundError
        When *filepath* does not exist.
    KeyError
        When the checkpoint dictionary has no ``"model"`` key.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: '{path}'.\n"
            "Run Stage 1 first:  python scripts/train.py --stage 1"
        )

    # Always map to CPU first so the caller can move tensors as needed.
    state: dict = torch.load(path, map_location="cpu")

    if "model" not in state:
        raise KeyError(
            f"Checkpoint at '{path}' has no 'model' key. "
            f"Available keys: {list(state.keys())}"
        )

    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    stage_info = f"  stage={state['stage']}" if "stage" in state else ""
    epoch_info = f"  epoch={state['epoch']}" if "epoch" in state else ""
    print(f"✔  Loaded checkpoint ← {path}{stage_info}{epoch_info}")

    return state


# ---------------------------------------------------------------------------
# Discover latest checkpoint
# ---------------------------------------------------------------------------

def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path of the most recent ``.pt`` file in *checkpoint_dir*.

    "Most recent" is determined by the integer epoch number embedded in the
    filename (e.g. ``epoch_12.pt`` → 12).  Files without a recognisable epoch
    number are ranked below those that have one, and ties are broken by
    modification time (newest wins).

    Parameters
    ----------
    checkpoint_dir : str
        Directory to search (non-recursively).

    Returns
    -------
    str | None
        Absolute path of the latest checkpoint, or ``None`` if the directory
        contains no ``.pt`` files.

    Examples
    --------
    >>> ckpt = get_latest_checkpoint("models/")
    >>> if ckpt:
    ...     state = load_checkpoint(ckpt, model)
    """
    ckpt_dir = Path(checkpoint_dir)
    candidates = list(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None

    _epoch_re = re.compile(r"epoch[_\-]?(\d+)", re.IGNORECASE)

    def _sort_key(p: Path) -> tuple[int, float]:
        match = _epoch_re.search(p.stem)
        epoch = int(match.group(1)) if match else -1
        return (epoch, p.stat().st_mtime)

    latest = max(candidates, key=_sort_key)
    return str(latest)
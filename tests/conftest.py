"""
tests/conftest.py
-----------------
Shared pytest fixtures available to test files that need them.

Provides:
  - device: CUDA availability detection (used by integration tests)
"""

import sys
from pathlib import Path
import pytest
import torch

# Ensure src can be imported from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────
# Session-level fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope='session')
def device():
    """Detect CUDA availability and return appropriate device string."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
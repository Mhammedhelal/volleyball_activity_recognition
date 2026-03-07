# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────
import json

from pathlib import Path

_LABELS_FILE = Path(__file__).parent / "labels.json"

with open(_LABELS_FILE) as f:
    _labels = json.load(f)

PERSON_ACTIONS     : list[str] = _labels["person_actions"]
GROUP_ACTIVITIES   : list[str] = _labels["group_activities"]

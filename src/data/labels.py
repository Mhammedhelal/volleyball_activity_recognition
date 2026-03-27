# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────
import json
from pathlib import Path

_LABELS_FILE = Path(__file__).parent / "labels.json"

with open(_LABELS_FILE) as f:
    _labels = json.load(f)

# Sort by integer value so index position == class index
# labels.json stores {"name": index, ...} dicts
PERSON_ACTIONS: list[str] = [
    k for k, v in sorted(_labels["person_actions"].items(), key=lambda x: x[1])
]
GROUP_ACTIVITIES: list[str] = [
    k for k, v in sorted(_labels["group_activities"].items(), key=lambda x: x[1])
]

# Sanity check — must match default.yaml expectations
assert len(PERSON_ACTIONS)   == 9,  f"Expected 9 person actions,   got {len(PERSON_ACTIONS)}"
assert len(GROUP_ACTIVITIES) == 8,  f"Expected 8 group activities, got {len(GROUP_ACTIVITIES)}"
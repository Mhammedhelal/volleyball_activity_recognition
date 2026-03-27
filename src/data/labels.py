# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────
#
# Labels are loaded exclusively from configs/default.yaml so there is a
# single source of truth.  labels.json is no longer used and can be deleted.
#
from pathlib import Path

import yaml

_DEFAULT_YAML = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"

with _DEFAULT_YAML.open("r") as _f:
    _cfg = yaml.safe_load(_f)

# Preserve the list order defined in default.yaml (index == class index)
PERSON_ACTIONS: list[str] = list(_cfg["labels"]["person_actions"])
GROUP_ACTIVITIES: list[str] = list(_cfg["labels"]["group_activities"])

# Sanity check — must match default.yaml expectations
assert len(PERSON_ACTIONS) == _cfg["labels"]["num_person_classes"], (
    f"Expected {_cfg['labels']['num_person_classes']} person actions, "
    f"got {len(PERSON_ACTIONS)}"
)
assert len(GROUP_ACTIVITIES) == _cfg["labels"]["num_group_classes"], (
    f"Expected {_cfg['labels']['num_group_classes']} group activities, "
    f"got {len(GROUP_ACTIVITIES)}"
)
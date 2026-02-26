# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────
import json

with open("labels.json") as f:
    LABELS = json.load(f)

PERSON_ACTIONS = LABELS["person_actions"]
GROUP_ACTIONS  = LABELS["group_activities"]

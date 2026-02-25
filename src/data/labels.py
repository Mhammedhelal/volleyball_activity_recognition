# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────

PERSON_ACTIONS = [
    "waiting", "setting", "digging", "falling",
    "spiking", "blocking", "jumping", "moving", "standing"
]  # 9 classes

GROUP_ACTIVITIES = [
    "left_set", "left_spike", "left_pass", "left_winpoint",
    "right_set", "right_spike", "right_pass", "right_winpoint",
]  # 8 classes

# Sub-group activity classes (same 4 per team, used by GroupEmbedder)
SUBGROUP_ACTIVITIES = ["set", "spike", "pass", "winpoint"]  # 4 classes
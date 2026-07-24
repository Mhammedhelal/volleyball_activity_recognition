"""
src/utils/visualization.py
---------------------------
Parses the plain-text evaluation reports written by
`src.engine.evaluator.Evaluator.report()` (see src/engine/evaluator.py) and
renders a results dashboard (PNG) summarizing group-activity and
person-action performance.

Log format expected (produced by Evaluator.report())
-----------------------------------------------------
    ======================================================================
    EVALUATION RESULTS
    Model: <model_name>   |   Timestamp: <YYYY-MM-DD HH:MM:SS>
    ======================================================================

    Group Activity Accuracy: 72.34%  (154/213)
    ----------------------------------------------------------------------
      Class                       Accuracy
      ------                       --------
      r_set                          82.50%
      r_spike                        65.00%
      ...

    Person Action Accuracy: 65.00%  (500/750)
    ----------------------------------------------------------------------
      Class                       Accuracy
      ------                       --------
      waiting                        70.00%
      ...

    ======================================================================

The person-action section is optional (frame-based baselines B1/B4 have
HAS_PERSON_LOSS=False and never write it — see Evaluator.report()).

Public API
----------
    parse_eval_log(path)                       -> dict
    find_eval_logs(log_dir, pattern="eval_*.txt") -> list[Path]
    load_latest_per_model(log_dir, pattern=...)   -> dict[model_name, dict]
    build_dashboard(parsed, save_path)          -> Path   (single run)
    build_comparison_dashboard(parsed_list, save_path) -> Path  (multi-run)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_HEADER_RE      = re.compile(r"Model:\s*(?P<model>\S+)\s*\|\s*Timestamp:\s*(?P<ts>.+)")
_GROUP_ACC_RE   = re.compile(
    r"Group Activity Accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)"
)
_PERSON_ACC_RE  = re.compile(
    r"Person Action Accuracy:\s*([\d.]+)%\s*\((\d+)/(\d+)\)"
)
_CLASS_ROW_RE   = re.compile(r"^\s{2}(\S.*?)\s{2,}([\d.]+)%\s*$")
_SEPARATOR_RE   = re.compile(r"^\s*-{5,}\s*$")
_TABLE_HEADER_RE = re.compile(r"^\s*(Class|-+)\s+(Accuracy|-+)\s*$")


def parse_eval_log(path: str | Path) -> dict:
    """
    Parse a single Evaluator.report() log file.

    Returns
    -------
    dict with keys:
        model_name        : str
        timestamp         : str
        group_accuracy     : float   (0-1)
        group_correct      : int
        group_total        : int
        group_per_class     : dict[str, float]   (0-1 values)
        person_accuracy     : float | None
        person_correct       : int | None
        person_total          : int | None
        person_per_class      : dict[str, float] | None
        source_path        : str
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    model_name = path.stem
    timestamp  = ""
    header_match = _HEADER_RE.search(text)
    if header_match:
        model_name = header_match.group("model")
        timestamp  = header_match.group("ts").strip()

    group_match = _GROUP_ACC_RE.search(text)
    if group_match is None:
        raise ValueError(f"Could not find Group Activity Accuracy line in {path}")
    group_accuracy = float(group_match.group(1)) / 100.0
    group_correct  = int(group_match.group(2))
    group_total    = int(group_match.group(3))

    person_match = _PERSON_ACC_RE.search(text)
    person_accuracy = person_correct = person_total = None
    if person_match:
        person_accuracy = float(person_match.group(1)) / 100.0
        person_correct  = int(person_match.group(2))
        person_total    = int(person_match.group(3))

    # ── walk lines to pull the two per-class tables ────────────────────────
    group_per_class:  dict[str, float] = {}
    person_per_class: dict[str, float] = {}

    section: Optional[str] = None  # "group" | "person" | None
    for line in lines:
        if "Group Activity Accuracy" in line:
            section = "group"
            continue
        if "Person Action Accuracy" in line:
            section = "person"
            continue
        if line.strip().startswith("=" * 5):
            section = None
            continue
        if _SEPARATOR_RE.match(line) or _TABLE_HEADER_RE.match(line):
            continue

        row = _CLASS_ROW_RE.match(line)
        if row and section is not None:
            cls_name = row.group(1).strip()
            acc      = float(row.group(2)) / 100.0
            if section == "group":
                group_per_class[cls_name] = acc
            elif section == "person":
                person_per_class[cls_name] = acc

    return {
        "model_name":        model_name,
        "timestamp":         timestamp,
        "group_accuracy":    group_accuracy,
        "group_correct":     group_correct,
        "group_total":       group_total,
        "group_per_class":   group_per_class,
        "person_accuracy":   person_accuracy,
        "person_correct":    person_correct,
        "person_total":      person_total,
        "person_per_class":  person_per_class or None,
        "source_path":       str(path),
    }


def find_eval_logs(log_dir: str | Path, pattern: str = "eval_*.txt") -> list[Path]:
    """Return all evaluation log files under *log_dir* matching *pattern*."""
    log_dir = Path(log_dir)
    return sorted(log_dir.glob(pattern))


def load_latest_per_model(
    log_dir: str | Path, pattern: str = "eval_*.txt"
) -> dict[str, dict]:
    """
    Parse all matching logs and keep only the most recently modified log
    per model_name (useful for a fair cross-model comparison dashboard).
    """
    latest: dict[str, dict] = {}
    latest_mtime: dict[str, float] = {}

    for log_path in find_eval_logs(log_dir, pattern):
        parsed = parse_eval_log(log_path)
        mtime  = log_path.stat().st_mtime
        name   = parsed["model_name"]
        if name not in latest_mtime or mtime > latest_mtime[name]:
            latest[name]       = parsed
            latest_mtime[name] = mtime

    return latest


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_subplot(ax, per_class: dict[str, float], overall: float, title: str) -> None:
    if not per_class:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    names  = list(per_class.keys())
    values = [per_class[n] * 100 for n in names]

    # Sort ascending so weakest classes are easiest to spot on the left
    order  = sorted(range(len(names)), key=lambda i: values[i])
    names  = [names[i] for i in order]
    values = [values[i] for i in order]

    colors = ["#d62728" if v < overall * 100 else "#2ca02c" for v in values]

    ax.barh(names, values, color=colors)
    ax.axvline(overall * 100, color="black", linestyle="--", linewidth=1,
               label=f"Overall: {overall*100:.1f}%")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)


def _summary_text(ax, parsed: dict) -> None:
    ax.axis("off")
    lines = [
        f"Model:      {parsed['model_name']}",
        f"Timestamp:  {parsed['timestamp'] or 'n/a'}",
        "",
        f"Group Activity Accuracy:  {parsed['group_accuracy']*100:.2f}%  "
        f"({parsed['group_correct']}/{parsed['group_total']})",
    ]
    if parsed.get("person_accuracy") is not None:
        lines.append(
            f"Person Action Accuracy:   {parsed['person_accuracy']*100:.2f}%  "
            f"({parsed['person_correct']}/{parsed['person_total']})"
        )
    else:
        lines.append("Person Action Accuracy:   n/a (model has no person supervision)")

    ax.text(
        0.02, 0.95, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=11, family="monospace",
    )


# ---------------------------------------------------------------------------
# Public dashboard builders
# ---------------------------------------------------------------------------

def build_dashboard(parsed: dict, save_path: str | Path) -> Path:
    """
    Build a single-run dashboard: overall summary + per-class accuracy bars
    for group activity and (if present) person action.

    Parameters
    ----------
    parsed    : dict returned by parse_eval_log()
    save_path : destination .png path (parent dirs created if needed)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_person = parsed.get("person_per_class") is not None
    n_cols = 3 if has_person else 2

    fig = plt.figure(figsize=(6 * n_cols, 6))
    gs  = fig.add_gridspec(1, n_cols, width_ratios=[1] + [1.4] * (n_cols - 1))

    ax_summary = fig.add_subplot(gs[0, 0])
    _summary_text(ax_summary, parsed)

    ax_group = fig.add_subplot(gs[0, 1])
    _bar_subplot(
        ax_group, parsed["group_per_class"], parsed["group_accuracy"],
        "Group Activity — Per-Class Accuracy",
    )

    if has_person:
        ax_person = fig.add_subplot(gs[0, 2])
        _bar_subplot(
            ax_person, parsed["person_per_class"], parsed["person_accuracy"],
            "Person Action — Per-Class Accuracy",
        )

    fig.suptitle(f"Evaluation Dashboard — {parsed['model_name']}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def build_comparison_dashboard(parsed_list: list[dict], save_path: str | Path) -> Path:
    """
    Build a cross-model comparison dashboard: grouped bar chart of overall
    group/person accuracy for each parsed run (e.g. across baselines B1-B7,
    or across n_subgroups=1/2/4 ablations).

    Parameters
    ----------
    parsed_list : list of dicts returned by parse_eval_log()
    save_path   : destination .png path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not parsed_list:
        raise ValueError("parsed_list is empty — nothing to compare")

    # Sort by group accuracy descending for readability
    parsed_list = sorted(parsed_list, key=lambda p: p["group_accuracy"], reverse=True)

    names           = [p["model_name"] for p in parsed_list]
    group_accs      = [p["group_accuracy"] * 100 for p in parsed_list]
    person_accs     = [
        p["person_accuracy"] * 100 if p.get("person_accuracy") is not None else 0.0
        for p in parsed_list
    ]
    has_person_flags = [p.get("person_accuracy") is not None for p in parsed_list]

    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))

    ax.bar([i - width / 2 for i in x], group_accs, width, label="Group Activity", color="#1f77b4")
    person_bar_x = [i + width / 2 for i in x]
    person_bar_vals = [v if flag else 0 for v, flag in zip(person_accs, has_person_flags)]
    ax.bar(person_bar_x, person_bar_vals, width, label="Person Action", color="#ff7f0e")

    # Annotate bars missing person accuracy
    for i, flag in enumerate(has_person_flags):
        if not flag:
            ax.text(i + width / 2, 1.5, "n/a", ha="center", va="bottom",
                     fontsize=8, rotation=90, color="gray")

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model Comparison — Overall Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path
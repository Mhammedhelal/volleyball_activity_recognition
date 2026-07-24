"""
scripts/visualize_results.py
-----------------------------
Build a results dashboard from Evaluator.report() log files and save it
under the outputs/figures directory.

Two modes
---------
1. Single-run dashboard (default): parses ONE log file — either the most
   recent one for a given --model name, or an explicit --log-file — and
   renders a per-class accuracy dashboard.

2. Comparison dashboard (--compare): parses the latest log for every model
   found under --log-dir and renders a grouped bar chart comparing overall
   group/person accuracy across all of them (e.g. baselines B1-B7, or
   n_subgroups=1/2/4 ablations).

Usage
-----
    # Dashboard for the most recent log of a specific model
    python scripts/visualize_results.py --model hierarchical

    # Dashboard for one specific log file
    python scripts/visualize_results.py --log-file outputs/logs/eval_B7_20260724_101500.txt

    # Comparison dashboard across every model found in outputs/logs
    python scripts/visualize_results.py --compare

    # Custom config (for paths.log_dir / paths.figures_dir) and output name
    python scripts/visualize_results.py --config configs/default.yaml --compare \
        --output outputs/figures/baseline_comparison.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils.visualization import (
    build_comparison_dashboard,
    build_dashboard,
    find_eval_logs,
    load_latest_per_model,
    parse_eval_log,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a results dashboard from Evaluator.report() logs"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config (used for default log/figures dirs)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override log directory (default: cfg.paths.log_dir)")
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Override figures output directory (default: cfg.paths.figures_dir)")
    parser.add_argument("--pattern", type=str, default="eval_*.txt",
                        help="Glob pattern for log files")

    parser.add_argument("--log-file", type=str, default=None,
                        help="Parse this exact log file instead of searching by --model")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to find the most recent log for (single-run mode)")

    parser.add_argument("--compare", action="store_true",
                        help="Build a comparison dashboard across all models found in --log-dir")

    parser.add_argument("--output", type=str, default=None,
                        help="Output .png path (default: auto-named under figures dir)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── config (optional — only used for default dir resolution) ──────────
    cfg = None
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path
    if config_path.exists():
        cfg = Config.from_yaml(config_path)

    # ── resolve directories ────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent

    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    elif cfg is not None and hasattr(cfg, "paths") and hasattr(cfg.paths, "log_dir"):
        log_dir = Path(cfg.paths.log_dir)
    else:
        log_dir = Path("outputs/logs")
    if not log_dir.is_absolute():
        log_dir = project_root / log_dir

    if args.figures_dir is not None:
        figures_dir = Path(args.figures_dir)
    elif cfg is not None and hasattr(cfg, "paths") and hasattr(cfg.paths, "figures_dir"):
        figures_dir = Path(cfg.paths.figures_dir)
    else:
        figures_dir = Path("outputs/figures")
    if not figures_dir.is_absolute():
        figures_dir = project_root / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── comparison mode ─────────────────────────────────────────────────────
    if args.compare:
        latest_per_model = load_latest_per_model(log_dir, pattern=args.pattern)
        if not latest_per_model:
            raise FileNotFoundError(
                f"No log files matching '{args.pattern}' found under {log_dir}"
            )
        print(f"Found {len(latest_per_model)} model(s) to compare: "
              f"{list(latest_per_model.keys())}")

        output_path = Path(args.output) if args.output else figures_dir / "comparison_dashboard.png"
        saved = build_comparison_dashboard(list(latest_per_model.values()), output_path)
        print(f"✔  Comparison dashboard saved → {saved}")
        return

    # ── single-run mode ──────────────────────────────────────────────────────
    if args.log_file is not None:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = project_root / log_path
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
    elif args.model is not None:
        latest_per_model = load_latest_per_model(log_dir, pattern=args.pattern)
        if args.model not in latest_per_model:
            raise FileNotFoundError(
                f"No log found for model '{args.model}' under {log_dir}. "
                f"Available: {list(latest_per_model.keys())}"
            )
        log_path = Path(latest_per_model[args.model]["source_path"])
    else:
        # No --log-file / --model given: fall back to the most recently
        # modified log file in log_dir.
        candidates = find_eval_logs(log_dir, pattern=args.pattern)
        if not candidates:
            raise FileNotFoundError(
                f"No log files matching '{args.pattern}' found under {log_dir}. "
                "Run scripts/evaluate.py first, or pass --log-file / --model / --compare."
            )
        log_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"No --log-file/--model/--compare given — using most recent log: {log_path.name}")

    parsed = parse_eval_log(log_path)
    print(f"Parsed log for model '{parsed['model_name']}'  "
          f"(group_acc={parsed['group_accuracy']*100:.2f}%)")

    output_path = (
        Path(args.output) if args.output
        else figures_dir / f"dashboard_{parsed['model_name']}.png"
    )
    saved = build_dashboard(parsed, output_path)
    print(f"✔  Dashboard saved → {saved}")


if __name__ == "__main__":
    main()
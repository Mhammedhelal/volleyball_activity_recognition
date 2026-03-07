"""
scripts/run_tests.py
--------------------
Generic test runner. Reads test configuration from a YAML config file
via src/config.py so it works in any project without editing this file.

The config must contain a `testing` section:

    testing:
      tests_dir: "tests"
      module_under_test:
        test_dataset.py:   "VolleyballDataset / volleyball_collate"
        test_model.py:     "MyModel"
        ...

For every test:
  - Print PASS / FAIL
  - On failure: show the test name, the module under test, and the exception

Usage
-----
    python scripts/run_tests.py
    python scripts/run_tests.py --config configs/default.yaml
    python scripts/run_tests.py --verbose          # show PASS lines too
    python scripts/run_tests.py --stop-first       # stop after first failure
    python scripts/run_tests.py --file test_dataset.py
"""

import argparse
import importlib
import importlib.util
import inspect
import sys
import traceback
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
sys.path.insert(0, str(Path(__file__).resolve().parent))



# ─────────────────────────────────────────────
# Colours (disabled on Windows or when piped)
# ─────────────────────────────────────────────

USE_COLOUR = sys.stdout.isatty() and sys.platform != "win32"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text


GREEN  = lambda t: _c("32", t)
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ─────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────

def _load_testing_config(
    config_path: Path,
    project_root: Path,
) -> tuple[Path, dict[str, str]]:
    """
    Load the `testing` section from a YAML config file.

    Tries to use src/config.py if it exists in the project root;
    otherwise falls back to plain yaml.safe_load so the runner
    works in projects that don't have src/config.py.

    Returns
    -------
    tests_dir          : Path   absolute path to the tests directory
    module_under_test  : dict   {filename: human-readable module description}
    """
    src_config = project_root / "src" / "config.py"

    if src_config.exists():
        # Use the project's own Config class
        sys.path.insert(0, str(project_root))
        spec   = importlib.util.spec_from_file_location("src.config", src_config)
        mod    = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg    = mod.Config.from_yaml(config_path)
        testing = cfg.testing
        tests_dir         = project_root / testing.tests_dir
        module_under_test = dict(testing.module_under_test.to_dict()
                                 if hasattr(testing.module_under_test, "to_dict")
                                 else testing.module_under_test)
    else:
        # Plain YAML fallback — no src/config.py dependency
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        testing           = raw.get("testing", {})
        tests_dir         = project_root / testing.get("tests_dir", "tests")
        module_under_test = testing.get("module_under_test", {})

    return tests_dir, module_under_test


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class TestResult:
    file:       str
    class_name: str
    test_name:  str
    module:     str
    passed:     bool
    exc_type:   str = ""
    exc_msg:    str = ""
    tb:         str = ""


@dataclass
class RunSummary:
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self)  -> list[TestResult]: return [r for r in self.results if r.passed]
    @property
    def failed(self)  -> list[TestResult]: return [r for r in self.results if not r.passed]
    @property
    def n_pass(self)  -> int: return len(self.passed)
    @property
    def n_fail(self)  -> int: return len(self.failed)
    @property
    def n_total(self) -> int: return len(self.results)


# ─────────────────────────────────────────────
# Test discovery and execution
# ─────────────────────────────────────────────

def _collect_test_methods(cls) -> list[tuple[str, Callable]]:
    """Return all test_* methods on a class, in definition order."""
    return [
        (name, fn)
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name.startswith("test")
    ]


def _collect_fixtures(cls) -> dict[str, Callable]:
    """
    Collect module-scoped fixture methods (non-test, non-private methods).
    Called once per class; results are cached and injected by parameter name.
    """
    return {
        name: fn
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("test") and not name.startswith("_")
    }


def _run_test(
    cls,
    method_name:    str,
    fixtures:       dict,
    cached_fixtures: dict,
) -> tuple[bool, str, str, str]:
    """
    Run one test method, injecting fixtures by parameter name.
    Returns (passed, exc_type, exc_msg, traceback_str).
    """
    fn  = getattr(cls, method_name)
    sig = inspect.signature(fn)

    kwargs: dict = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if param_name in cached_fixtures:
            kwargs[param_name] = cached_fixtures[param_name]
        elif param_name in fixtures:
            cached_fixtures[param_name] = fixtures[param_name](cls())
            kwargs[param_name] = cached_fixtures[param_name]

    try:
        fn(cls(), **kwargs)
        return True, "", "", ""
    except Exception as exc:
        return False, type(exc).__name__, str(exc), traceback.format_exc()


def run_test_file(
    test_path:         Path,
    module_under_test: dict[str, str],
    verbose:           bool,
    stop_first:        bool,
    summary:           RunSummary,
) -> bool:
    """
    Dynamically import one test file, run all Test* classes and test_*
    methods, and append results to summary.

    Returns False if stop_first is True and a failure occurred.
    """
    file_name  = test_path.name
    module_tag = module_under_test.get(file_name, file_name)

    print(f"\n{BOLD(f'── {file_name}')}")
    print(DIM(f"   testing: {module_tag}"))
    print()

    # ── Dynamic import ────────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location(test_path.stem, test_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(RED(f"  [IMPORT ERROR] {exc}"))
        traceback.print_exc()
        return True   # import errors aren't test failures — keep going

    # ── Discover Test* classes ────────────────────────────────────────────────
    test_classes = [
        obj for _, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__name__.startswith("Test") and obj.__module__ == mod.__name__
    ]

    if not test_classes:
        print(YELLOW("  No test classes found."))
        return True

    for cls in test_classes:
        methods         = _collect_test_methods(cls)
        fixtures        = _collect_fixtures(cls)
        cached_fixtures: dict = {}

        # Eagerly call module-scoped fixtures so failures surface immediately
        for fname, ffn in fixtures.items():
            try:
                cached_fixtures[fname] = ffn(cls())
            except Exception:
                pass   # will fail again at call time with a clear traceback

        for method_name, _ in methods:
            passed, exc_type, exc_msg, tb = _run_test(
                cls, method_name, fixtures, cached_fixtures
            )
            result = TestResult(
                file       = file_name,
                class_name = cls.__name__,
                test_name  = method_name,
                module     = module_tag,
                passed     = passed,
                exc_type   = exc_type,
                exc_msg    = exc_msg,
                tb         = tb,
            )
            summary.results.append(result)

            label  = f"{cls.__name__}.{method_name}"
            status = GREEN("PASS") if passed else RED("FAIL")

            if passed:
                if verbose:
                    print(f"  [{status}]  {label}")
            else:
                print(f"  [{status}]  {label}")
                print(f"         {BOLD('Module under test:')} {module_tag}")
                print(f"         {BOLD('Exception:        ')} {exc_type}: {exc_msg}")
                tb_lines = [l for l in tb.strip().splitlines() if l.strip()]
                for line in (tb_lines[-3:] if len(tb_lines) > 3 else tb_lines):
                    print(DIM(f"         {line}"))
                print()

                if stop_first:
                    return False

    return True


# ─────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────

def print_summary(summary: RunSummary) -> None:
    print("\n" + "=" * 70)
    print(BOLD("TEST SUMMARY"))
    print("=" * 70)

    if summary.n_fail == 0:
        print(GREEN(f"\n  All {summary.n_total} tests passed.\n"))
        return

    print(RED(f"\n  {summary.n_fail} / {summary.n_total} tests FAILED\n"))

    current_file = None
    for r in summary.failed:
        if r.file != current_file:
            current_file = r.file
            print(f"  {BOLD(r.file)}  {DIM(f'({r.module})')}")
        print(RED(f"    ✗  {r.class_name}.{r.test_name}"))
        print(DIM( f"       {r.exc_type}: {r.exc_msg}"))

    bar_pass = "█" * summary.n_pass
    bar_fail = "░" * summary.n_fail
    print(f"\n  {GREEN(bar_pass)}{RED(bar_fail)}"
          f"  {summary.n_pass} passed / {summary.n_fail} failed\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic test runner. Reads test configuration from a YAML config "
            "file so it works in any project without editing this script."
        )
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help=(
            "Path to YAML config file containing a 'testing' section. "
            "Defaults to configs/default.yaml relative to the project root."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show PASS lines in addition to FAIL lines",
    )
    parser.add_argument(
        "--stop-first", "-x", action="store_true",
        help="Stop after the first test failure",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Run a single test file (e.g. test_dataset.py)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Project root = two levels up from this script (scripts/run_tests.py)
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    config_path = Path(args.config) if args.config else (
        project_root / "configs" / "default.yaml"
    )

    if not config_path.exists():
        print(RED(f"Config file not found: {config_path}"))
        print(YELLOW(
            "Provide --config or create configs/default.yaml with a "
            "'testing.module_under_test' section."
        ))
        sys.exit(1)

    # ── Load testing config ───────────────────────────────────────────────────
    try:
        tests_dir, module_under_test = _load_testing_config(config_path, project_root)
    except Exception as exc:
        print(RED(f"Failed to load testing config from {config_path}: {exc}"))
        traceback.print_exc()
        sys.exit(1)

    if not tests_dir.exists():
        print(RED(f"Tests directory not found: {tests_dir}"))
        sys.exit(1)

    print(DIM(f"Config:    {config_path}"))
    print(DIM(f"Tests dir: {tests_dir}"))

    # ── Discover test files ───────────────────────────────────────────────────
    # Preserve the order from module_under_test, then append any extra test_*.py
    if args.file:
        test_files = [tests_dir / args.file]
    else:
        ordered = [tests_dir / name for name in module_under_test if
                   (tests_dir / name).exists()]
        extras  = [p for p in sorted(tests_dir.glob("test_*.py"))
                   if p not in ordered]
        test_files = ordered + extras

    if not test_files:
        print(YELLOW("No test files found."))
        sys.exit(0)

    print(BOLD(f"\nRunning {len(test_files)} test file(s)\n"))

    summary = RunSummary()

    for test_path in test_files:
        keep_going = run_test_file(
            test_path         = test_path,
            module_under_test = module_under_test,
            verbose           = args.verbose,
            stop_first        = args.stop_first,
            summary           = summary,
        )
        if not keep_going:
            print(YELLOW("\n  Stopped after first failure (--stop-first)."))
            break

    print_summary(summary)
    sys.exit(0 if summary.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
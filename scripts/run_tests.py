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

Supports:
  - Class-level fixture methods (non-test, non-private methods on Test* classes)
  - Module-level @pytest.fixture functions (scope="module" honoured via shared cache)
  - @pytest.mark.parametrize (single decorator, stacked decorators → cartesian product)
  - setup_method / teardown_method lifecycle hooks

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
import itertools
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
# Parametrize expansion
# ─────────────────────────────────────────────

def _expand_parametrize(name: str, fn: Callable) -> list[tuple[str, Callable, dict]]:
    """
    Expand @pytest.mark.parametrize decorators into individual test variants.

    Supports:
      - Single decorator   → one argname, list of values
      - Stacked decorators → cartesian product of all param lists

    Returns a list of (display_name, fn, extra_kwargs) tuples.
    If the function has no parametrize marks, returns [(name, fn, {})].
    """
    marks = getattr(fn, "pytestmark", [])
    param_marks = [m for m in marks if getattr(m, "name", None) == "parametrize"]

    if not param_marks:
        return [(name, fn, {})]

    # Each mark contributes one axis: (argname, [values])
    axes = []
    for mark in param_marks:
        argname = mark.args[0]
        values  = mark.args[1]
        axes.append((argname, values))

    # Cartesian product across all axes
    variants = []
    argnames = [a[0] for a in axes]
    valuelists = [a[1] for a in axes]

    for combo in itertools.product(*valuelists):
        extra_kwargs = dict(zip(argnames, combo))
        suffix       = ",".join(f"{k}={v}" for k, v in extra_kwargs.items())
        display_name = f"{name}[{suffix}]"
        variants.append((display_name, fn, extra_kwargs))

    return variants


# ─────────────────────────────────────────────
# Test discovery
# ─────────────────────────────────────────────

def _collect_test_methods(cls) -> list[tuple[str, Callable, dict]]:
    """
    Return all test_* methods on a class, expanded for @pytest.mark.parametrize.
    Each entry is (display_name, fn, extra_kwargs).
    """
    results = []
    for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction):
        if not name.startswith("test"):
            continue
        results.extend(_expand_parametrize(name, fn))
    return results


def _collect_class_fixtures(cls) -> dict[str, Callable]:
    """
    Collect class-level fixture methods (non-test, non-private methods).
    Called once per class; results are cached and injected by parameter name.
    """
    return {
        name: fn
        for name, fn in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("test") and not name.startswith("_")
    }


def _collect_module_fixtures(mod) -> dict[str, Callable]:
    """
    Collect module-level @pytest.fixture functions.

    Detection strategy (tried in order — first match wins):

    1. _pytestfixturefunction  — present when pytest is fully initialised
    2. _fixture_function_marker — older pytest internals
    3. __wrapped__             — @pytest.fixture wraps the function; the
                                 wrapper callable is not a plain function so
                                 inspect.isfunction returns False for it, but
                                 its __wrapped__ attribute points to the original
    4. Callable but not a plain function — @pytest.fixture returns a
                                 FixtureFunctionMarker / SubRequest object;
                                 detect by checking the type name contains
                                 "fixture" (case-insensitive)

    Falls back gracefully: if pytest is not installed, none of the test files
    that use @pytest.fixture will even import successfully, so this is a
    non-issue in that case.
    """
    result = {}

    # Collect plain functions (strategy 1 & 2)
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ != mod.__name__:
            continue
        if (hasattr(obj, "_pytestfixturefunction") or
                hasattr(obj, "_fixture_function_marker")):
            result[name] = obj

    # Collect callables that are NOT plain functions (strategy 3 & 4)
    # @pytest.fixture wraps the function in a FixtureFunctionMarker which is
    # callable but not a function — inspect.isfunction returns False for it.
    for name in dir(mod):
        if name.startswith("_") or name in result:
            continue
        obj = getattr(mod, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            continue
        if not callable(obj):
            continue
        # Check for __wrapped__ (the original function is stored here)
        if hasattr(obj, "__wrapped__") and inspect.isfunction(obj.__wrapped__):
            if getattr(obj.__wrapped__, "__module__", None) == mod.__name__:
                result[name] = obj.__wrapped__
            continue
        # Check type name as last resort
        type_name = type(obj).__name__.lower()
        if "fixture" in type_name:
            # Try to get the underlying callable
            underlying = getattr(obj, "__func__", None) or getattr(obj, "func", None)
            if underlying and inspect.isfunction(underlying):
                result[name] = underlying
            else:
                # Store the callable itself — it will be called with no args
                result[name] = obj

    return result


# ─────────────────────────────────────────────
# Test execution
# ─────────────────────────────────────────────

def _run_test(
    cls,
    method_name:          str,
    fn:                   Callable,
    fixtures:             dict,
    cached_fixtures:      dict,
    extra_kwargs:         dict,
    module_fixtures:      dict,
    module_fixture_cache: dict,
) -> tuple[bool, str, str, str]:
    """
    Run one test method, injecting fixtures by parameter name.

    Resolution order for each parameter:
      1. extra_kwargs        — values from @pytest.mark.parametrize
      2. cached_fixtures     — class-level fixtures already built
      3. fixtures            — class-level fixture factories (build + cache)
      4. module_fixture_cache — module-level fixtures already built
      5. module_fixtures     — module-level @pytest.fixture factories (build + cache)

    Returns (passed, exc_type, exc_msg, traceback_str).
    """
    sig    = inspect.signature(fn)
    kwargs = dict(extra_kwargs)   # start with parametrize values

    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if param_name in kwargs:
            continue
        if param_name in cached_fixtures:
            kwargs[param_name] = cached_fixtures[param_name]
        elif param_name in fixtures:
            cached_fixtures[param_name] = fixtures[param_name](cls())
            kwargs[param_name] = cached_fixtures[param_name]
        elif param_name in module_fixture_cache:
            kwargs[param_name] = module_fixture_cache[param_name]
        elif param_name in module_fixtures:
            try:
                module_fixture_cache[param_name] = module_fixtures[param_name]()
            except TypeError:
                module_fixture_cache[param_name] = module_fixtures[param_name].__call__()
            kwargs[param_name] = module_fixture_cache[param_name]

    try:
        instance = cls()
        if hasattr(instance, "setup_method"):
            instance.setup_method()
        fn(instance, **kwargs)
        if hasattr(instance, "teardown_method"):
            instance.teardown_method()
        return True, "", "", ""
    except Exception as exc:
        return False, type(exc).__name__, str(exc), traceback.format_exc()


# ─────────────────────────────────────────────
# File runner
# ─────────────────────────────────────────────

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

    # ── Module-level fixtures (shared across all classes in this file) ────────
    module_fixtures      = _collect_module_fixtures(mod)
    module_fixture_cache: dict = {}

    # ── Discover Test* classes ────────────────────────────────────────────────
    test_classes = [
        obj for _, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__name__.startswith("Test") and obj.__module__ == mod.__name__
    ]

    if not test_classes:
        print(YELLOW("  No test classes found."))
        return True

    for cls in test_classes:
        methods         = _collect_test_methods(cls)   # (display_name, fn, extra_kwargs)
        fixtures        = _collect_class_fixtures(cls)
        cached_fixtures: dict = {}

        # Eagerly call class-level fixtures so failures surface immediately
        for fname, ffn in fixtures.items():
            try:
                cached_fixtures[fname] = ffn(cls())
            except Exception:
                pass   # will fail again at call time with a clear traceback

        for display_name, fn, extra_kwargs in methods:
            passed, exc_type, exc_msg, tb = _run_test(
                cls            = cls,
                method_name    = display_name,
                fn             = fn,
                fixtures       = fixtures,
                cached_fixtures = cached_fixtures,
                extra_kwargs   = extra_kwargs,
                module_fixtures      = module_fixtures,
                module_fixture_cache = module_fixture_cache,
            )
            result = TestResult(
                file       = file_name,
                class_name = cls.__name__,
                test_name  = display_name,
                module     = module_tag,
                passed     = passed,
                exc_type   = exc_type,
                exc_msg    = exc_msg,
                tb         = tb,
            )
            summary.results.append(result)

            label  = f"{cls.__name__}.{display_name}"
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
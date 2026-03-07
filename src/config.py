"""
src/config.py
-------------
Configuration class that loads from YAML and provides dict-like access.

Usage
-----
    from src.config import Config

    # Load from file
    cfg = Config.from_yaml("configs/default.yaml")

    # Attribute access (nested)
    cfg.training.stage1.lr          # → 1e-5
    cfg.cnn.feature_dim             # → 4096

    # Dict-like access
    cfg["pooling"]["num_subgroups"] # → 2

    # Override a value at runtime
    cfg.training.stage1.lr = 3e-5

    # Merge an override dict (e.g. from argparse or a variant yaml)
    cfg.merge({"training": {"stage1": {"batch_size": 16}}})

    # Convert back to a plain dict
    plain = cfg.to_dict()

    # Save the (possibly modified) config
    cfg.to_yaml("outputs/run_config.yaml")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import yaml


# ---------------------------------------------------------------------------
# Internal recursive namespace
# ---------------------------------------------------------------------------

class _Namespace:
    """Recursive dot-access namespace backed by a plain dict."""

    def __init__(self, data: dict) -> None:
        # Store raw dict so we can serialise without extra logic
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._set(key, value)

    #  construction helpers 

    def _set(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            self._data[key] = _Namespace(value)
        else:
            self._data[key] = value

    #  attribute access 

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'. "
                f"Available keys: {list(self._data.keys())}"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            object.__setattr__(self, "_data", value)
        else:
            self._set(key, value)

    def __delattr__(self, key: str) -> None:
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    #  dict-like access 

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._set(key, value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return (v for v in self._data.values())

    def items(self):
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    #  serialisation 

    def to_dict(self) -> dict:
        """Recursively convert to a plain Python dict."""
        out = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, _Namespace) else value
        return out

    #  merging 

    def _merge_dict(self, override: dict) -> None:
        """Deep-merge *override* into self, in-place."""
        for key, value in override.items():
            if (
                key in self._data
                and isinstance(self._data[key], _Namespace)
                and isinstance(value, dict)
            ):
                self._data[key]._merge_dict(value)
            else:
                self._set(key, value)

    #  repr 

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Namespace({inner})"


# ---------------------------------------------------------------------------
# Public Config class
# ---------------------------------------------------------------------------

class Config(_Namespace):
    """
    Top-level configuration object.

    Loads a YAML file and wraps it in a recursive namespace that supports
    both attribute-style (``cfg.training.stage1.lr``) and dict-style
    (``cfg["training"]["stage1"]["lr"]``) access.

    Parameters
    ----------
    data : dict
        Raw configuration dict (usually parsed from YAML).
    """

    def __init__(self, data: dict) -> None:
        super().__init__(data)

    #  factories 

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        Config

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Construct a Config directly from a plain dict."""
        return cls(copy.deepcopy(data))

    #  merging 

    def merge(self, override: dict | "Config") -> "Config":
        """Deep-merge *override* into this config (in-place).

        Nested keys that exist in both objects are merged recursively;
        new keys are added; existing leaf values are replaced.

        Parameters
        ----------
        override : dict | Config
            Values to merge in.  Can be a plain dict or another Config.

        Returns
        -------
        Config
            ``self``, to allow chaining: ``cfg.merge({...}).merge({...})``

        Example
        -------
        >>> cfg = Config.from_yaml("configs/default.yaml")
        >>> cfg.merge({"training": {"stage1": {"lr": 3e-5}}})
        >>> cfg.training.stage1.lr
        3e-05
        """
        if isinstance(override, Config):
            override = override.to_dict()
        self._merge_dict(override)
        return self

    @classmethod
    def from_yaml_with_overrides(
        cls,
        base_path: str | Path,
        *override_paths: str | Path,
        overrides: dict | None = None,
    ) -> "Config":
        """Load a base YAML and apply zero or more override YAMLs and/or a
        plain-dict override on top.

        Parameters
        ----------
        base_path : str | Path
            Path to the base config (e.g. ``configs/default.yaml``).
        *override_paths : str | Path
            Paths to override config files applied in order
            (e.g. ``configs/2group.yaml``).
        overrides : dict, optional
            Final plain-dict overrides applied last (e.g. from argparse).

        Example
        -------
        >>> cfg = Config.from_yaml_with_overrides(
        ...     "configs/default.yaml",
        ...     "configs/2group.yaml",
        ...     overrides={"training": {"stage1": {"batch_size": 32}}},
        ... )
        """
        cfg = cls.from_yaml(base_path)
        for p in override_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"Override config not found: {p}")
            with p.open("r") as fh:
                data = yaml.safe_load(fh) or {}
            cfg.merge(data)
        if overrides:
            cfg.merge(overrides)
        return cfg

    #  serialisation 

    def to_yaml(self, path: str | Path) -> None:
        """Write the current config to a YAML file.

        Parameters
        ----------
        path : str | Path
            Destination path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.dump(
                self.to_dict(),
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    #  convenience helpers 

    def copy(self) -> "Config":
        """Return a deep copy of this config."""
        return Config.from_dict(self.to_dict())

    #  repr 

    def __repr__(self) -> str:
        top_keys = list(self._data.keys())
        return f"Config(sections={top_keys})"

    def __str__(self) -> str:
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
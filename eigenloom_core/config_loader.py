"""Declarative configuration loader for the EigenLoom cybernetic manifold grid."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any


class ManifoldConfigSpec:
    """Thread-safe accessor around config/eigenloom_manifold_grid.json."""

    def __init__(self, spec_path: Path | None = None) -> None:
        default_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "eigenloom_manifold_grid.json"
        )
        self._path = Path(spec_path) if spec_path is not None else default_path
        self._raw: dict[str, Any] = json.loads(
            self._path.read_text(encoding="utf-8")
        )
        self._constants: dict[str, Any] = dict(self._raw["numeric_constants"])

    @property
    def path(self) -> Path:
        return self._path

    def raw_copy(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)

    def section(self, key: str) -> Any:
        return self._raw[key]

    def c_int(self, name: str) -> int:
        return int(self._constants[name])

    def c_float(self, name: str) -> float:
        return float(self._constants[name])


_DEFAULT_LOCK = threading.RLock()
_DEFAULT_SPEC: ManifoldConfigSpec | None = None


def get_manifold_spec(spec_path: Path | None = None) -> ManifoldConfigSpec:
    """Return cached default ManifoldConfigSpec or load an explicit override."""
    global _DEFAULT_SPEC
    if spec_path is not None:
        return ManifoldConfigSpec(spec_path)
    with _DEFAULT_LOCK:
        if _DEFAULT_SPEC is None:
            _DEFAULT_SPEC = ManifoldConfigSpec()
        return _DEFAULT_SPEC

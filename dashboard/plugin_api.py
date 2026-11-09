"""EigenLoom Dashboard FastAPI Router — Read-only SQLite WAL telemetry endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from eigenloom_core.config_loader import get_manifold_spec  # noqa: E402
from eigenloom_core.storage import (  # noqa: E402
    EigenLoomStore,
    resolve_eigenloom_db_path,
)
from eigenloom_core.weight_bus import CyberneticWeightBus  # noqa: E402

try:
    from fastapi import APIRouter
except ImportError:

    class APIRouter:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.routes: list[tuple[str, Any]] = []

        def get(self, path: str) -> Any:
            def _decorator(fn: Any) -> Any:
                self.routes.append((path, fn))
                return fn

            return _decorator


router = APIRouter()
_BASELINE_BUS: CyberneticWeightBus | None = None


def _get_baseline_bus() -> CyberneticWeightBus:
    global _BASELINE_BUS
    if _BASELINE_BUS is None:
        _BASELINE_BUS = CyberneticWeightBus(get_manifold_spec())
    return _BASELINE_BUS


@router.get("/status")
def get_eigenloom_status() -> dict[str, Any]:
    """Return read-only SQLite WAL status merged with 20-engine HyperGrid metadata."""
    db_path = resolve_eigenloom_db_path()
    ro = EigenLoomStore.read_readonly_snapshot(db_path)
    base = ro["snapshot"] if isinstance(ro.get("snapshot"), dict) else _get_baseline_bus().manifold_status()
    merged = dict(base)
    merged["wal_persisted"] = bool(ro["persisted"])
    merged["persisted_tasks"] = int(ro["tasks"])
    merged["persisted_verifications"] = int(ro["verifications"])
    merged["persisted_attractors"] = int(ro["attractors"])
    return merged


@router.get("/engines")
def get_eigenloom_engines() -> dict[str, Any]:
    """Return all 20 Functional Engines, 72 Sub-Engines, and 5 Feedback Timescales."""
    spec = get_manifold_spec()
    status = get_eigenloom_status()
    return {
        "engine_count": status["engine_count"],
        "sub_engine_count": status["sub_engine_count"],
        "hypergrid_total_dimensions": status["hypergrid_total_dimensions"],
        "strata": spec.section("strata"),
        "feedback_timescales": spec.section("feedback_timescales"),
        "engines": status["engines"],
    }

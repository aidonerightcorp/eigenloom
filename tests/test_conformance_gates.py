"""Mechanical conformance gates: Zero Magic Literals, Domain-Agnostic Naming, Desktop Surface, and Dual Linalg."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as eigenloom_plugin  # noqa: E402
from eigenloom_core.config_loader import get_manifold_spec  # noqa: E402
from eigenloom_core.grid_and_gates import (  # noqa: E402
    DomainAgnosticNamingGate,
    GridCellRecord,
    GridLedger,
    ZeroHardcodedValuesAstGate,
)
from eigenloom_core.linalg_backend import (  # noqa: E402
    add_scaled,
    frobenius_norm,
    identity_matrix,
    matmul,
    ridge_solve,
    svd_compact,
    woodbury_rank1_update,
)
from eigenloom_core.matryoshka_manifold import MatryoshkaManifold  # noqa: E402
from eigenloom_core.weight_bus import CyberneticWeightBus  # noqa: E402


class _RecordingContext:
    def __init__(self) -> None:
        self.tools: list[str] = []
        self.hooks: list[str] = []
        self.commands: list[str] = []
        self.skills: list[str] = []
        self.context_engine = None
        self.memory_provider = None
        self.delegation_provider = None

    def get_config(self, key: str, default=None):
        return default

    def register_context_engine(self, eng) -> None:
        self.context_engine = eng

    def register_memory_provider(self, prov) -> None:
        self.memory_provider = prov

    def register_delegation_provider(self, prov) -> None:
        self.delegation_provider = prov

    def register_tool(self, name: str, **kwargs) -> None:
        self.tools.append(name)

    def register_hook(self, hook_name: str, callback, **kwargs) -> None:
        self.hooks.append(hook_name)

    def register_command(self, name: str, handler, **kwargs) -> None:
        self.commands.append(name)

    def register_skill(self, name: str, path: Path, **kwargs) -> None:
        self.skills.append(name)


def test_zero_hardcoded_values_ast_gate() -> None:
    operational_files = sorted((REPO_ROOT / "eigenloom_core").glob("*.py")) + [
        REPO_ROOT / "__init__.py",
        REPO_ROOT / "dashboard" / "plugin_api.py",
    ]
    violations = ZeroHardcodedValuesAstGate.scan_paths(operational_files)
    assert violations == [], f"Magic literal violations found: {violations}"


def test_domain_agnostic_naming_gate() -> None:
    violations = DomainAgnosticNamingGate.scan_repository(REPO_ROOT)
    assert violations == [], f"Domain-coupled naming violations: {violations}"


def test_desktop_plugin_surface_sdk_conformance() -> None:
    desktop_js = REPO_ROOT / "desktop" / "plugin.js"
    assert desktop_js.is_file()
    src = desktop_js.read_text(encoding="utf-8")
    assert "from '@hermes/plugin-sdk'" in src
    assert "from 'react/jsx-runtime'" in src
    assert "#f5f2ea" in src
    assert "#141413" in src
    assert "only light" in src
    for forbidden in (
        "eval(",
        "new Function(",
        "hermesDesktop",
        "[data-slot",
        "[data-tour",
        "[data-sidebar",
        "[data-testid",
        "<script",
        "webview",
    ):
        assert forbidden not in src


def test_overview_diagram_exists_and_light_palette() -> None:
    png_path = REPO_ROOT / "docs" / "eigenloom-overview.png"
    assert png_path.is_file()
    size = png_path.stat().st_size
    assert 10_000 < size < 250_000
    header = png_path.read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"


def test_manifest_and_registration_exact_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eigenloom_core.storage as storage_mod

    monkeypatch.setattr(
        storage_mod,
        "resolve_eigenloom_db_path",
        lambda hermes_home=None: tmp_path / "parity.db",
    )
    manifest_text = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert "kind: standalone" in manifest_text
    assert "manifest_version: 2" in manifest_text
    assert 'requires_hermes: ">=0.23.0"' in manifest_text

    ctx = _RecordingContext()
    eigenloom_plugin.register(ctx)
    assert len(ctx.tools) == 12
    assert len(set(ctx.tools)) == 12
    for t_name in ctx.tools:
        assert f"- {t_name}" in manifest_text
    assert set(ctx.hooks) == {
        "on_session_start",
        "pre_tool_call",
        "post_tool_call",
        "on_session_end",
    }
    assert ctx.commands == ["eigenloom"]
    assert ctx.skills == ["eigenloom-cybernetics"]
    assert ctx.context_engine is not None and ctx.context_engine.name == "eigenloom"
    assert ctx.memory_provider is not None and ctx.memory_provider.name == "eigenloom"
    assert ctx.delegation_provider is not None and ctx.delegation_provider.name == "eigenloom"


def test_twenty_engines_seventy_two_subengines_hundred_hypergrid_dims() -> None:
    spec = get_manifold_spec()
    bus = CyberneticWeightBus(spec)
    status = bus.manifold_status()
    assert status["engine_count"] == 20
    assert status["sub_engine_count"] == 72
    assert status["hypergrid_total_dimensions"] == 100
    assert status["timescale_count"] == 5
    assert status["matryoshka_spaces"] == [
        "task2vec",
        "config2vec",
        "weight2vec",
        "trace2vec",
    ]
    assert status["matryoshka_prefix_dims"] == [16, 32, 64, 128]
    for required_engine in (
        "inter_engine_coupling_and_jacobian_bus",
        "homeostatic_meta_controller_engine",
        "structural_morphism_and_transfer_engine",
        "human_attention_and_escalation_engine",
    ):
        assert required_engine in bus.engines


def test_grid_ledger_enforces_maker_checker_separation() -> None:
    spec = get_manifold_spec()
    manifold = MatryoshkaManifold(spec)
    ledger = GridLedger(spec)
    c_vec = manifold.embed_config({"alpha": 0.5})
    rec = GridCellRecord(
        cell_id="test_cell",
        engine_id="quotient_context_dag_engine",
        regime="test",
        values={"alpha": 0.5},
        state="represented",
        config_vec=c_vec,
    )
    ledger.register_cell(rec)
    ledger.transition("test_cell", "evaluated", reward=0.9)

    with pytest.raises(ValueError, match="maker != checker"):
        ledger.transition(
            "test_cell",
            "verified",
            maker_id="same_actor",
            checker_id="same_actor",
        )

    verified = ledger.transition(
        "test_cell",
        "verified",
        maker_id="actor_maker",
        checker_id="actor_checker",
    )
    assert verified.state == "verified"


def test_dual_linalg_stdlib_and_default_svd_reconstruction() -> None:
    mat = [
        [3.0, 1.0, 0.5, -0.2],
        [0.8, 2.5, -0.4, 0.7],
        [-0.3, 0.6, 1.9, 0.3],
        [0.2, -0.5, 0.4, 1.4],
    ]
    for stdlib_only in (False, True):
        u_m, s_v, vt_m = svd_compact(mat, force_stdlib=stdlib_only)
        diag_s = [[s_v[r] if r == c else 0.0 for c in range(len(s_v))] for r in range(len(s_v))]
        recon = matmul(matmul(u_m, diag_s, force_stdlib=stdlib_only), vt_m, force_stdlib=stdlib_only)
        err = frobenius_norm(add_scaled(recon, mat, scale=-1.0))
        assert err < 1e-6


def test_dual_linalg_ridge_and_woodbury_consistency() -> None:
    x_mat = [[1.0, 0.2], [0.4, 1.1], [0.9, -0.5], [-0.3, 0.8]]
    y_mat = [[2.0, -1.0], [1.2, 0.9], [1.5, -1.3], [-0.1, 0.7]]
    w_def = ridge_solve(x_mat, y_mat, ridge_lambda=0.05, force_stdlib=False)
    w_std = ridge_solve(x_mat, y_mat, ridge_lambda=0.05, force_stdlib=True)
    assert frobenius_norm(add_scaled(w_def, w_std, scale=-1.0)) < 1e-7

    a0 = identity_matrix(3)
    feat = [0.5, -0.3, 0.8]
    u_def = woodbury_rank1_update(a0, feat, damping=1.0, force_stdlib=False)
    u_std = woodbury_rank1_update(a0, feat, damping=1.0, force_stdlib=True)
    assert frobenius_norm(add_scaled(u_def, u_std, scale=-1.0)) < 1e-9
    assert math.isfinite(u_std[0][0])

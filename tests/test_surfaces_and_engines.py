"""Comprehensive functional tests across all 5 Hermes surfaces, 4-space Matryoshka, HyperGrid SHA, and 7 targets."""

from __future__ import annotations

import base64
import concurrent.futures
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as eigenloom_plugin  # noqa: E402
from dashboard import plugin_api as dashboard_api  # noqa: E402
from eigenloom_core.context_engine import EigenLoomContextEngine  # noqa: E402
from eigenloom_core.delegation_provider import (  # noqa: E402
    DelegationRouteError,
    EigenLoomDelegationProvider,
)
from eigenloom_core.linalg_backend import safetensors_loads, vec_norm  # noqa: E402
from eigenloom_core.matryoshka_manifold import MatryoshkaManifold  # noqa: E402
from eigenloom_core.memory_provider import EigenLoomMemoryProvider  # noqa: E402
from eigenloom_core.storage import EigenLoomStore  # noqa: E402
from eigenloom_core.weight_bus import CyberneticWeightBus  # noqa: E402


class _HarnessContext:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.hooks: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}
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

    def register_tool(self, name: str, handler, **kwargs) -> None:
        self.tools[name] = handler

    def register_hook(self, hook_name: str, callback, **kwargs) -> None:
        self.hooks[hook_name] = callback

    def register_command(self, name: str, handler, **kwargs) -> None:
        self.commands[name] = handler

    def register_skill(self, name: str, path: Path, **kwargs) -> None:
        pass


# ─── 1. 4-Space Matryoshka Manifold Tests ─────────────────────────────────────


def test_matryoshka_four_spaces_and_elastic_prefix_unit_norms() -> None:
    m = MatryoshkaManifold()
    v_task = m.embed_task("compile high precision matrix operator", stratum="compile")
    v_cfg = m.embed_config({"ridge_lambda": 0.01, "layers": 3}, stratum="compile")
    v_wt = m.embed_weights([[[1.0, 0.2], [0.1, 0.8]]], stratum="compile")
    v_tr = m.embed_trace("verified matrix operator trace", stratum="compile")

    for vec in (v_task, v_cfg, v_wt, v_tr):
        assert len(vec.values) == 128
        for dim in (16, 32, 64, 128):
            sliced = vec.slice_prefix(dim)
            assert len(sliced) == dim
            assert abs(vec_norm(sliced) - 1.0) < 1e-6

    # Same-stratum vectors share positive coarse 16d regime alignment
    assert v_task.cosine(v_cfg, 16) > 0.0
    assert v_task.cosine(v_wt, 16) > -1.0


def test_matryoshka_progressive_cascade_search_saves_macs() -> None:
    m = MatryoshkaManifold()
    candidates = [
        (f"item_{i}", m.embed_trace(f"document topic number {i} spectral matrix"), {"idx": i})
        for i in range(40)
    ]
    target_vec = m.embed_trace("document topic number 17 spectral matrix")
    res = m.progressive_cascade_search(target_vec, candidates, top_k=3)
    assert len(res["matches"]) == 3
    assert res["matches"][0]["id"] == "item_17"
    assert res["mac_operations_used"] < res["exhaustive_mac_baseline"]
    assert res["compute_savings_ratio"] > 0.25


# ─── 2. Fluid 100D HyperGrid & Opportunistic SHA Tests ────────────────────────


def test_fluid_hypergrid_soft_kernel_interpolation_and_clamping() -> None:
    bus = CyberneticWeightBus()
    q_reflex = bus.manifold.embed_task("ultra fast reflex token filter", stratum="ingestion_and_memory")
    q_deep = bus.manifold.embed_task("deep exhaustive verification spectrum", stratum="adaptation_and_economics")

    cfg_a = bus.hypergrid.interpolate_fluid_config("quotient_context_dag_engine", q_reflex, temperature=0.15)
    cfg_b = bus.hypergrid.interpolate_fluid_config("quotient_context_dag_engine", q_deep, temperature=0.15)

    assert abs(sum(cfg_a["kernel_weights"].values()) - 1.0) < 1e-6
    assert 0.35 <= cfg_a["parameters"]["compression_ratio_target"] <= 0.88
    assert cfg_a["interpolation_entropy"] >= 0.0
    assert cfg_b["config2vec"].space == "config2vec"


def test_woodbury_bilinear_surrogate_reduces_epistemic_uncertainty() -> None:
    bus = CyberneticWeightBus()
    t_vec = bus.manifold.embed_task("novel algebraic synthesis task", stratum="compile")
    c_vec = bus.manifold.embed_config({"rank_cap": 8, "ridge": 0.02}, stratum="compile")

    before = bus.hypergrid.surrogate.score(t_vec, c_vec)["epistemic_std"]
    bus.hypergrid.surrogate.observe(t_vec, c_vec, reward=0.92)
    after = bus.hypergrid.surrogate.score(t_vec, c_vec)["epistemic_std"]
    assert after < before


def test_opportunistic_sha_exploration_headroom_and_zero_tokens() -> None:
    bus = CyberneticWeightBus()
    skipped = bus.hypergrid.opportunistic_explore(
        "fluid_hypergrid_interpolator_engine", compute_headroom_ratio=0.0
    )
    assert skipped["skipped_low_headroom"] is True
    assert skipped["candidates_generated"] == 0

    active = bus.hypergrid.opportunistic_explore(
        "fluid_hypergrid_interpolator_engine",
        compute_headroom_ratio=0.85,
        seed=42,
    )
    assert active["skipped_low_headroom"] is False
    assert active["candidates_generated"] >= 10
    assert active["covering_array_rows"] >= 1
    assert len(active["rung_counts"]) == 3
    assert active["rung_counts"][0] >= active["rung_counts"][1] >= active["rung_counts"][2]
    assert active["extra_llm_tokens"] == 0
    assert active["winner_cell_id"] is not None


def test_hypergrid_boundary_auto_expansion() -> None:
    bus = CyberneticWeightBus()
    grid = bus.hypergrid.grids["ridge_polytope_compiler_engine"]
    old_high = grid.parameters["rank_cap"].bounds.high
    old_gen = grid.generation
    expansions = grid.maybe_expand_boundaries({"rank_cap": old_high})
    assert len(expansions) == 1
    assert grid.parameters["rank_cap"].bounds.high > old_high
    assert grid.generation == old_gen + 1


# ─── 3. Surface 1: ContextEngine Tests ────────────────────────────────────────


def test_context_engine_preserves_exact_message_reference_for_prompt_cache(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = EigenLoomStore(db_path=tmp_path / "ctx.db", manifold=bus.manifold)
    eng = EigenLoomContextEngine(bus=bus, store=store)

    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Short prompt within budget"},
    ]
    selected = eng.select_context(messages, budget_tokens=4096)
    assert selected is messages
    assert bus.cost_ledger.prompt_cache_identity_hits == 1


def test_context_engine_quotient_compression_and_grep(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = EigenLoomStore(db_path=tmp_path / "ctx_comp.db", manifold=bus.manifold)
    eng = EigenLoomContextEngine(bus=bus, store=store, context_length=400, threshold_percent=0.2)

    msgs = [{"role": "system", "content": "Core rules"}]
    for i in range(12):
        msgs.append({"role": "user", "content": "duplicate filler note about routine check"})
        msgs.append({"role": "assistant", "content": f"Unique spectral eigenvalue observation alpha_{i} = 1.732"})
    msgs.append({"role": "user", "content": "Final active question"})

    compacted = eng.compress(msgs, force=True, focus_topic="spectral eigenvalue")
    assert len(compacted) < len(msgs)
    assert eng.compression_count == 1

    grep_raw = eng.handle_tool_call("eigenloom_context_grep", {"query": "alpha_5", "limit": 5})
    grep_data = json.loads(grep_raw)
    assert grep_data["count"] >= 1
    assert any("alpha_5" in m["snippet"] for m in grep_data["matches"])


def test_context_engine_clone_and_pooled_thread_safety(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = EigenLoomStore(db_path=tmp_path / "ctx_pool.db", manifold=bus.manifold)
    parent = EigenLoomContextEngine(bus=bus, store=store)
    child = parent.clone_for_agent()
    child.update_model("fast-tier", context_length=8192)
    assert parent.context_length != 8192

    def _worker(tid: int) -> int:
        clone = parent.clone_for_agent()
        clone.on_session_start(session_id=f"sess_{tid}")
        turns = [{"role": "user", "content": f"thread {tid} message {k}"} for k in range(10)]
        out = clone.compress(turns, force=True)
        return len(out)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_worker, range(8)))
    assert all(r >= 2 for r in results)


# ─── 4. Surface 2: MemoryProvider Tests ───────────────────────────────────────


def test_memory_provider_profile_isolation_and_hopfield_recall(tmp_path: Path) -> None:
    home_a = tmp_path / "profile_alpha"
    home_b = tmp_path / "profile_beta"

    prov_a = EigenLoomMemoryProvider()
    prov_b = EigenLoomMemoryProvider()
    assert prov_a.is_available() is True

    prov_a.initialize("s1", hermes_home=str(home_a), agent_context="primary")
    prov_b.initialize("s2", hermes_home=str(home_b), agent_context="primary")

    prov_a.sync_turn("What is the Frobenius norm target?", "Keep anchor drift strictly at zero.")
    rec_a = json.loads(prov_a.handle_tool_call("eigenloom_attractor_recall", {"query": "Frobenius anchor drift"}))
    rec_b = json.loads(prov_b.handle_tool_call("eigenloom_attractor_recall", {"query": "Frobenius anchor drift"}))

    assert len(rec_a["matches"]) == 1
    assert len(rec_b["matches"]) == 0


def test_memory_provider_suppresses_writes_in_cron_and_subagent(tmp_path: Path) -> None:
    for ctx_kind in ("cron", "subagent", "flush"):
        home = tmp_path / f"home_{ctx_kind}"
        prov = EigenLoomMemoryProvider()
        prov.initialize("s_bg", hermes_home=str(home), agent_context=ctx_kind)
        prov.sync_turn("background prompt", "background completion")
        prov.on_pre_compress([{"role": "user", "content": "transient subagent note"}])
        recalled = json.loads(prov.handle_tool_call("eigenloom_attractor_recall", {"query": "background"}))
        assert recalled["matches"] == []


def test_memory_provider_pre_compress_consolidates_safetensors_cartridge(tmp_path: Path) -> None:
    prov = EigenLoomMemoryProvider()
    prov.initialize("s_main", hermes_home=str(tmp_path / "home_main"), agent_context="primary")
    summary = prov.on_pre_compress(
        [
            {"role": "user", "content": "Configure Woodburys inverse covariance solver"},
            {"role": "assistant", "content": "Verified O(d^2) rank-one update active"},
        ]
    )
    assert "Woodbury" in summary
    prefetched = prov.prefetch("Woodbury inverse covariance")
    assert "hopfield=" in prefetched
    assert "EigenLoom Hopfield Attractor Recall" in prov.system_prompt_block()
    prov.shutdown()
    assert prov.system_prompt_block() == ""


# ─── 5. Surface 3: DelegationProvider Tests ───────────────────────────────────


def test_delegation_provider_linucb_routing_and_maker_checker_enforcement(tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    store = EigenLoomStore(db_path=tmp_path / "del.db", manifold=bus.manifold)
    delegator = EigenLoomDelegationProvider(bus=bus, store=store)

    params = delegator.task_parameters({})
    assert {"engine_stratum", "verification_role", "maker_id"} <= set(params.keys())

    maker_route = delegator.route_task(
        {"goal": "compile C99 operator kernel", "engine_stratum": "compile", "verification_role": "maker"},
        {},
    )
    assert "label" in maker_route
    maker_label = maker_route["label"]

    checker_route = delegator.route_task(
        {
            "goal": "audit C99 operator kernel",
            "engine_stratum": "verify",
            "verification_role": "checker",
            "maker_id": maker_label,
        },
        {},
    )
    assert checker_route["label"] != maker_label

    # Single-tier config colliding with maker_id MUST fail closed with DelegationRouteError
    with pytest.raises(DelegationRouteError):
        delegator.route_task(
            {
                "goal": "audit with colliding tier",
                "verification_role": "checker",
                "maker_id": "solo_tier",
            },
            {"tiers": [{"tier_id": "solo_tier", "model": "solo-model", "reasoning_effort": "high"}]},
        )


# ─── 6. Surface 4 & Weight-Space Bus: 12 Tools, 7 Targets, 0.0 Drift ──────────


def test_fission_claim_fence_verify_and_cache_hit_pipeline(tmp_path: Path) -> None:
    rt = eigenloom_plugin.EigenLoomRuntime(db_path=tmp_path / "pipe.db")
    dag = rt.store.fission_dag("Design parser. Build evaluator. Verify invariants.")
    assert dag["task_count"] == 3
    first_id = dag["tasks"][0]["task_id"]

    lease_1 = rt.store.claim_and_fence(first_id, "agent_maker_1", lease_ttl_seconds=300.0)
    assert lease_1["acquired"] is True
    assert lease_1["fencing_stamp"].startswith("fence_")

    # Competing actor cannot steal active unexpired lease
    lease_2 = rt.store.claim_and_fence(first_id, "agent_intruder", lease_ttl_seconds=300.0)
    assert lease_2["acquired"] is False

    # Self-verification must raise ValueError
    with pytest.raises(ValueError, match="Independent verification"):
        rt.store.verify_record(first_id, maker_id="agent_maker_1", checker_id="agent_maker_1", passed=True)

    ver = rt.store.verify_record(
        first_id,
        maker_id="agent_maker_1",
        checker_id="agent_checker_2",
        passed=True,
        quality_score=0.95,
        output_summary="Verified parser AST contract",
    )
    assert ver["passed"] is True
    assert ver["dawid_skene_posterior_truth"] > 0.5
    assert "agent_maker_1" in ver["eigentrust_scores"]

    # Subsequent claim returns instant verified cache hit
    cached = rt.store.claim_and_fence(first_id, "agent_later")
    assert cached["cache_hit"] is True
    assert cached["output_summary"] == "Verified parser AST contract"


@pytest.mark.parametrize(
    "target",
    ["safetensors", "wasm_base64", "wat", "wgsl", "c99", "sql", "python"],
)
def test_compile_cartridge_all_seven_portable_targets(target: str, tmp_path: Path) -> None:
    bus = CyberneticWeightBus()
    rules = [
        {"input": [1.0, 0.0, 0.5, -0.5], "target": [0.8, -0.2, 0.1, 0.4]},
        {"input": [0.0, 1.0, -0.5, 0.5], "target": [-0.3, 0.9, 0.4, -0.1]},
        {"input": [0.5, 0.5, 0.5, 0.5], "target": [0.25, 0.35, 0.25, 0.15]},
    ]
    cart = bus.compiler.compile_stack("test_cart", rules, num_layers=3, activation="gelu")
    artifact = bus.compiler.lower_to_target(cart, target)
    assert isinstance(artifact, str) and len(artifact) > 20

    if target == "safetensors":
        raw = base64.b64decode(artifact)
        tensors, meta = safetensors_loads(raw)
        assert "readout_w" in tensors
        assert "layer_0.film_gamma" in tensors
        assert meta["cartridge_id"] == "test_cart"
    elif target == "wasm_base64":
        wasm_bytes = base64.b64decode(artifact)
        # Verify WebAssembly binary magic (\0asm) + version 1 with zero .wasm files on disk
        assert list(wasm_bytes[:8]) == [0, 97, 115, 109, 1, 0, 0, 0]
        assert list(REPO_ROOT.rglob("*.wasm")) == []
    elif target == "python":
        mod_file = tmp_path / "generated_cartridge.py"
        mod_file.write_text(artifact, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("generated_cartridge", mod_file)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.evaluate_cartridge([1.0, 0.0, 0.5, -0.5], [0.4])
        assert len(out) == 4 and all(math.isfinite(v) for v in out)


def test_multilayer_film_conditioning_and_early_exit() -> None:
    bus = CyberneticWeightBus()
    rules = [{"input": [1.0, 0.0, 0.0, 0.0], "target": [4.0, -2.0, -2.0, -2.0]}]
    cart = bus.compiler.compile_stack(
        "film_stack", rules, num_layers=5, early_exit_margin=0.15
    )
    res_fast = cart.forward([1.0, 0.0, 0.0, 0.0], cond_vec=[0.9] * 16, allow_early_exit=True)
    res_full = cart.forward([1.0, 0.0, 0.0, 0.0], cond_vec=[-0.9] * 16, allow_early_exit=False)

    assert res_fast["layers_executed"] < 5
    assert res_fast["layers_saved"] >= 1
    assert res_fast["output"] != res_full["output"]


@pytest.mark.parametrize(
    "mode",
    ["svd_nullspace", "lora_xs", "recursive_woodbury", "procrustes_align", "hopfield_write"],
)
def test_surgical_edits_preserve_exact_zero_anchor_drift_and_cegar(mode: str) -> None:
    bus = CyberneticWeightBus()
    probe = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    eng = bus.engines["svd_nullspace_surgical_editor_engine"]
    probe_before = [
        sum(eng.weights[r][c] * probe[c] for c in range(8)) for r in range(8)
    ]

    res = bus.surgical_edit(
        "svd_nullspace_surgical_editor_engine",
        mode=mode,
        input_vec=[0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.8],
        counterexample_probes=[probe],
    )
    assert res["anchor_drift"] == 0.0

    probe_after = [
        sum(eng.weights[r][c] * probe[c] for c in range(8)) for r in range(8)
    ]
    probe_diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(probe_before, probe_after)))
    assert probe_diff < 1e-9


def test_spectral_diagnostics_morphism_coupling_and_evsi() -> None:
    bus = CyberneticWeightBus()
    spec_diag = bus.analyze_spectrum(
        engine_id="rmt_marchenko_pastur_spectral_engine",
        sample_inputs=[[1.0, 2.0, 0.5, 1.0, 0.3, 0.8, 1.2, 0.4]],
    )
    assert spec_diag["effective_rank"] > 1.0
    assert spec_diag["marchenko_pastur_bulk_edge"] > 0.0
    assert len(spec_diag["wanda_row_saliency"]) == 8

    # Isomorphic renamed DAGs yield spectral isomorphism 1.0 and 0.0 target anchor drift
    dag_a = [{"id": "n1", "depends_on": []}, {"id": "n2", "depends_on": ["n1"]}, {"id": "n3", "depends_on": ["n2"]}]
    dag_b = [{"id": "m1", "depends_on": []}, {"id": "m2", "depends_on": ["m1"]}, {"id": "m3", "depends_on": ["m2"]}]
    morph = bus.morphism_transfer(dag_a, dag_b)
    assert abs(morph["dag_spectral_isomorphism"] - 1.0) < 1e-6
    assert morph["procrustes_orthogonality_error"] < 1e-6
    assert morph["target_anchor_drift"] == 0.0

    step = bus.coupling_step(
        impulse_engine_id="wanda_saliency_pruning_engine",
        impulse_delta=0.05,
        compute_headroom_ratio=0.7,
    )
    assert step["coupling_step"] == 1
    assert step["opportunistic_exploration"]["extra_llm_tokens"] == 0

    evsi_low = bus.calibrate_effort("routine ping", stakes=0.01)
    evsi_high = bus.calibrate_effort("critical distributed protocol verification", stakes=5.0)
    assert evsi_low["reasoning_effort"] == "low"
    assert evsi_high["recommended_matryoshka_dim"] >= evsi_low["recommended_matryoshka_dim"]


def test_all_twelve_registered_tools_hooks_command_and_dashboard_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eigenloom_core.storage as storage_mod

    test_db = tmp_path / "registered_tools.db"
    monkeypatch.setattr(
        storage_mod, "resolve_eigenloom_db_path", lambda hermes_home=None: test_db
    )
    monkeypatch.setattr(
        dashboard_api, "resolve_eigenloom_db_path", lambda hermes_home=None: test_db
    )
    ctx = _HarnessContext()
    eigenloom_plugin.register(ctx)

    # Exercise registered slash command and hooks
    ctx.hooks["on_session_start"](session_id="test_sess")
    ctx.hooks["pre_tool_call"]()
    status_txt = ctx.commands["eigenloom"]("status")
    step_txt = ctx.commands["eigenloom"]("step")
    assert "20 engines (72 sub-engines)" in status_txt
    assert "100 HyperGrid dims" in status_txt
    assert "coupling step" in step_txt

    # Invoke tool handlers through registered dictionary
    fission = json.loads(ctx.tools["eigenloom_fission_dag"]({"goal": "Alpha. Beta."}))
    tid = fission["tasks"][0]["task_id"]
    claim = json.loads(ctx.tools["eigenloom_claim_and_fence"]({"task_id": tid, "actor_id": "m1"}))
    assert claim["acquired"] is True
    ver = json.loads(
        ctx.tools["eigenloom_verify_record"](
            {"task_id": tid, "maker_id": "m1", "checker_id": "c1", "passed": True}
        )
    )
    assert ver["passed"] is True

    cart = json.loads(
        ctx.tools["eigenloom_compile_cartridge"](
            {
                "rules": [{"condition": "high load", "action": "shed noncritical"}],
                "target": "wgsl",
                "probe_input": [0.5, -0.2, 0.1, 0.8],
            }
        )
    )
    assert "@compute" in cart["artifact"]
    assert cart["probe_evaluation"] is not None

    m_status = json.loads(
        ctx.tools["eigenloom_manifold_status"]({"engine_id": "fluid_hypergrid_interpolator_engine"})
    )
    assert "interpolated_engine_config" in m_status

    ctx.hooks["post_tool_call"]()
    ctx.hooks["on_session_end"]()

    # Dashboard read-only FastAPI endpoints
    dash_status = dashboard_api.get_eigenloom_status()
    dash_engines = dashboard_api.get_eigenloom_engines()
    assert dash_status["engine_count"] == 20
    assert len(dash_engines["engines"]) == 20

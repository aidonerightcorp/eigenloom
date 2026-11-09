"""EigenLoom — Unified 5-Surface Cybernetic Engine Manifold Plugin for Hermes Agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PLUGIN_ROOT = str(Path(__file__).resolve().parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from eigenloom_core.config_loader import get_manifold_spec  # noqa: E402
from eigenloom_core.context_engine import (  # noqa: E402
    CONTEXT_GREP_SCHEMA,
    EigenLoomContextEngine,
)
from eigenloom_core.delegation_provider import (  # noqa: E402
    DelegationRouteError,
    EigenLoomDelegationProvider,
)
from eigenloom_core.memory_provider import (  # noqa: E402
    ATTRACTOR_RECALL_SCHEMA,
    EigenLoomMemoryProvider,
)
from eigenloom_core.storage import EigenLoomStore  # noqa: E402
from eigenloom_core.weight_bus import CyberneticWeightBus  # noqa: E402


class EigenLoomRuntime:
    """Per-registration runtime container coordinating all 5 Hermes surfaces."""

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self.spec = get_manifold_spec()
        self.bus = CyberneticWeightBus(self.spec)
        self.store = EigenLoomStore(
            db_path=db_path, spec=self.spec, manifold=self.bus.manifold
        )
        self.context_engine = EigenLoomContextEngine(
            bus=self.bus, store=self.store, spec=self.spec
        )
        self.memory_provider = EigenLoomMemoryProvider(
            bus=self.bus, spec=self.spec
        )
        self.delegation_provider = EigenLoomDelegationProvider(
            bus=self.bus, store=self.store, spec=self.spec
        )
        self.tool_invocations: int = 0
        self.persist_status_snapshot()

    def persist_status_snapshot(self) -> dict[str, Any]:
        status = self.bus.manifold_status()
        status["tool_invocations"] = self.tool_invocations
        status["eigentrust"] = self.store.compute_eigentrust()
        self.store.save_snapshot("latest_status", status)
        return status


def _build_tool_definitions(rt: EigenLoomRuntime) -> list[tuple[str, str, dict[str, Any], Any]]:
    def _handle_fission_dag(args: dict[str, Any], **_: Any) -> str:
        goal = str(args.get("goal", "") or "")
        subtasks = args.get("subtasks")
        max_fanout = args.get("max_fanout")
        out = rt.store.fission_dag(
            goal,
            subtasks=subtasks if isinstance(subtasks, list) else None,
            max_fanout=int(max_fanout) if max_fanout is not None else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_claim_and_fence(args: dict[str, Any], **_: Any) -> str:
        task_id = str(args["task_id"])
        actor_id = str(args.get("actor_id", "primary_maker") or "primary_maker")
        ttl = args.get("lease_ttl_seconds")
        out = rt.store.claim_and_fence(
            task_id,
            actor_id,
            lease_ttl_seconds=float(ttl) if ttl is not None else None,
        )
        return json.dumps(out)

    def _handle_verify_record(args: dict[str, Any], **_: Any) -> str:
        out = rt.store.verify_record(
            str(args["task_id"]),
            maker_id=str(args["maker_id"]),
            checker_id=str(args["checker_id"]),
            passed=bool(args.get("passed", True)),
            quality_score=float(args.get("quality_score", 1.0)),
            output_summary=str(args.get("output_summary", "") or ""),
            tier_id=str(args.get("tier_id", "verify_tier") or "verify_tier"),
            notes=str(args.get("notes", "") or ""),
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_compile_cartridge(args: dict[str, Any], **_: Any) -> str:
        cid = str(args.get("cartridge_id", "compiled_cartridge") or "compiled_cartridge")
        rules = args.get("rules", [])
        target = str(args.get("target", "python") or "python")
        num_layers = args.get("num_layers")
        activation = str(args.get("activation", "gelu") or "gelu")
        early_margin = args.get("early_exit_margin")

        cart = rt.bus.compiler.compile_stack(
            cid,
            rules if isinstance(rules, list) else [],
            num_layers=int(num_layers) if num_layers is not None else None,
            activation=activation,
            early_exit_margin=float(early_margin) if early_margin is not None else None,
        )
        artifact = rt.bus.compiler.lower_to_target(cart, target)
        probe_input = args.get("probe_input")
        cond_vec = args.get("condition_embedding")
        eval_info = None
        if isinstance(probe_input, list):
            eval_info = cart.forward(probe_input, cond_vec=cond_vec)
            rt.bus.cost_ledger.cartridge_evaluations += 1
            rt.bus.cost_ledger.layers_saved_by_early_exit += int(
                eval_info["layers_saved"]
            )
        rt.persist_status_snapshot()
        return json.dumps(
            {
                "cartridge_id": cart.cartridge_id,
                "target": target,
                "layers": len(cart.layers),
                "early_exit_margin": cart.early_exit_margin,
                "fit_mse": cart.fit_mse,
                "weight2vec_16d": cart.weight2vec.slice_prefix(
                    rt.spec.section("matryoshka_manifold")["coarse_dim"]
                ),
                "artifact": artifact,
                "probe_evaluation": eval_info,
            }
        )

    def _handle_analyze_spectrum(args: dict[str, Any], **_: Any) -> str:
        eid = args.get("engine_id")
        mat = args.get("matrix")
        samples = args.get("sample_inputs")
        out = rt.bus.analyze_spectrum(
            engine_id=str(eid) if eid else None,
            matrix=mat if isinstance(mat, list) else None,
            sample_inputs=samples if isinstance(samples, list) else None,
        )
        return json.dumps(out)

    def _handle_surgical_edit(args: dict[str, Any], **_: Any) -> str:
        eid = str(
            args.get("engine_id", rt.bus.engine_order[0]) or rt.bus.engine_order[0]
        )
        mode = str(args.get("mode", "svd_nullspace") or "svd_nullspace")
        out = rt.bus.surgical_edit(
            eid,
            mode=mode,
            input_vec=args.get("input_vec"),
            target_vec=args.get("target_vec"),
            counterexample_probes=args.get("counterexample_probes"),
            step_gain=float(args["step_gain"]) if "step_gain" in args and args["step_gain"] is not None else None,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_coupling_step(args: dict[str, Any], **_: Any) -> str:
        headroom = args.get("compute_headroom_ratio")
        replays = rt.store.recent_verified_records()
        out = rt.bus.coupling_step(
            impulse_engine_id=str(args["impulse_engine_id"]) if args.get("impulse_engine_id") else None,
            impulse_delta=float(args["impulse_delta"]) if args.get("impulse_delta") is not None else None,
            compute_headroom_ratio=float(headroom) if headroom is not None else None,
            replay_records=replays,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_morphism_transfer(args: dict[str, Any], **_: Any) -> str:
        src_dag = args.get("source_dag", [])
        dst_dag = args.get("target_dag", [])
        src_eng = str(
            args.get("source_engine_id", "ridge_polytope_compiler_engine")
            or "ridge_polytope_compiler_engine"
        )
        dst_eng = str(
            args.get("target_engine_id", "multilayer_film_molora_stack_engine")
            or "multilayer_film_molora_stack_engine"
        )
        out = rt.bus.morphism_transfer(
            src_dag if isinstance(src_dag, list) else [],
            dst_dag if isinstance(dst_dag, list) else [],
            source_engine_id=src_eng,
            target_engine_id=dst_eng,
        )
        rt.persist_status_snapshot()
        return json.dumps(out)

    def _handle_calibrate_effort(args: dict[str, Any], **_: Any) -> str:
        goal = str(args.get("goal", "") or "")
        stakes = args.get("stakes")
        stratum = str(args.get("stratum", "compile") or "compile")
        out = rt.bus.calibrate_effort(
            goal,
            stakes=float(stakes) if stakes is not None else None,
            stratum=stratum,
        )
        return json.dumps(out)

    def _handle_manifold_status(args: dict[str, Any], **_: Any) -> str:
        status = rt.persist_status_snapshot()
        engine_filter = args.get("engine_id")
        if engine_filter:
            eid = str(engine_filter)
            q_vec = rt.bus.manifold.embed_task(eid)
            fluid = rt.bus.hypergrid.interpolate_fluid_config(eid, q_vec)
            fluid_clean = {
                k: v for k, v in fluid.items() if k != "config2vec"
            }
            status["interpolated_engine_config"] = fluid_clean
        return json.dumps(status)

    def _handle_context_grep(args: dict[str, Any], **kw: Any) -> str:
        return rt.context_engine.handle_tool_call("eigenloom_context_grep", args, **kw)

    def _handle_attractor_recall(args: dict[str, Any], **kw: Any) -> str:
        return rt.memory_provider.handle_tool_call("eigenloom_attractor_recall", args, **kw)

    return [
        (
            "eigenloom_fission_dag",
            "Decompose a goal into an interface-contracted SQLite WAL DAG with minimum-entropy disambiguation.",
            {
                "name": "eigenloom_fission_dag",
                "description": "Decompose a goal into a fenced SQLite WAL subtask DAG with interface contracts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Composite objective to decompose."},
                        "subtasks": {"type": "array", "items": {"type": "object"}},
                        "max_fanout": {"type": "integer"},
                    },
                    "required": ["goal"],
                },
            },
            _handle_fission_dag,
        ),
        (
            "eigenloom_claim_and_fence",
            "Atomically claim a compare-and-swap task lease or return a verified cache hit.",
            {
                "name": "eigenloom_claim_and_fence",
                "description": "Claim an atomic task lease with monotonic fencing_stamp or reuse verified cache.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "actor_id": {"type": "string"},
                        "lease_ttl_seconds": {"type": "number"},
                    },
                    "required": ["task_id"],
                },
            },
            _handle_claim_and_fence,
        ),
        (
            "eigenloom_verify_record",
            "Record independent maker != checker verification and update Woodbury, Dawid-Skene, and EigenTrust.",
            {
                "name": "eigenloom_verify_record",
                "description": "Record independent maker != checker verification outcome.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "maker_id": {"type": "string"},
                        "checker_id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "quality_score": {"type": "number"},
                        "output_summary": {"type": "string"},
                        "tier_id": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["task_id", "maker_id", "checker_id", "passed"],
                },
            },
            _handle_verify_record,
        ),
        (
            "eigenloom_compile_cartridge",
            "Compile rules or traces into multi-layer FiLM+MoLoRA cartridges across 7 portable targets.",
            {
                "name": "eigenloom_compile_cartridge",
                "description": "Compile symbolic rules or verified traces into safetensors, wasm_base64, wat, wgsl, c99, sql, or python.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cartridge_id": {"type": "string"},
                        "rules": {"type": "array", "items": {"type": "object"}},
                        "target": {
                            "type": "string",
                            "enum": ["safetensors", "wasm_base64", "wat", "wgsl", "c99", "sql", "python"],
                        },
                        "num_layers": {"type": "integer"},
                        "activation": {"type": "string", "enum": ["gelu", "silu", "relu", "identity"]},
                        "early_exit_margin": {"type": "number"},
                        "probe_input": {"type": "array", "items": {"type": "number"}},
                        "condition_embedding": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["rules", "target"],
                },
            },
            _handle_compile_cartridge,
        ),
        (
            "eigenloom_analyze_spectrum",
            "Run Marchenko-Pastur RMT noise-floor, effective-rank, power-law exponent, and Wanda saliency diagnostics.",
            {
                "name": "eigenloom_analyze_spectrum",
                "description": "Run spectral diagnostics on any of the 20 engines or an explicit matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "sample_inputs": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                },
            },
            _handle_analyze_spectrum,
        ),
        (
            "eigenloom_surgical_edit",
            "Apply svd_nullspace, lora_xs, recursive_woodbury, procrustes_align, or hopfield_write edits with 0.0 anchor drift.",
            {
                "name": "eigenloom_surgical_edit",
                "description": "Apply zero-anchor-drift surgical weight edits with CEGAR counterexample healing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["svd_nullspace", "lora_xs", "recursive_woodbury", "procrustes_align", "hopfield_write"],
                        },
                        "input_vec": {"type": "array", "items": {"type": "number"}},
                        "target_vec": {"type": "array", "items": {"type": "number"}},
                        "counterexample_probes": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "step_gain": {"type": "number"},
                    },
                },
            },
            _handle_surgical_edit,
        ),
        (
            "eigenloom_coupling_step",
            "Run a multi-timescale cybernetic co-evolution pass across the 20x20 Jacobian bus + Opportunistic SHA exploration.",
            {
                "name": "eigenloom_coupling_step",
                "description": "Execute Jacobian coupling step with Lyapunov stability gate and Successive Halving exploration.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "impulse_engine_id": {"type": "string"},
                        "impulse_delta": {"type": "number"},
                        "compute_headroom_ratio": {"type": "number"},
                    },
                },
            },
            _handle_coupling_step,
        ),
        (
            "eigenloom_morphism_transfer",
            "Perform cross-domain task-DAG spectral isomorphism matching and Orthogonal Procrustes cartridge transplant.",
            {
                "name": "eigenloom_morphism_transfer",
                "description": "Zero-shot task-graph isomorphism + Orthogonal Procrustes subspace transfer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_dag": {"type": "array", "items": {"type": "object"}},
                        "target_dag": {"type": "array", "items": {"type": "object"}},
                        "source_engine_id": {"type": "string"},
                        "target_engine_id": {"type": "string"},
                    },
                    "required": ["source_dag", "target_dag"],
                },
            },
            _handle_morphism_transfer,
        ),
        (
            "eigenloom_calibrate_effort",
            "Compute Expected Value of Sample Information (EVSI) to select reasoning effort, MRL slice, and escalation.",
            {
                "name": "eigenloom_calibrate_effort",
                "description": "Compute EVSI effort cascade and reflex short-circuit recommendation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "stakes": {"type": "number"},
                        "stratum": {"type": "string"},
                    },
                    "required": ["goal"],
                },
            },
            _handle_calibrate_effort,
        ),
        (
            "eigenloom_manifold_status",
            "Return full 20-engine, 72-sub-engine, 100-dim HyperGrid, GridLedger, and OperationCostLedger status.",
            {
                "name": "eigenloom_manifold_status",
                "description": "Inspect live 20-engine / 72-sub-engine cybernetic manifold telemetry.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine_id": {"type": "string"},
                    },
                },
            },
            _handle_manifold_status,
        ),
        (
            "eigenloom_context_grep",
            CONTEXT_GREP_SCHEMA["description"],
            CONTEXT_GREP_SCHEMA,
            _handle_context_grep,
        ),
        (
            "eigenloom_attractor_recall",
            ATTRACTOR_RECALL_SCHEMA["description"],
            ATTRACTOR_RECALL_SCHEMA,
            _handle_attractor_recall,
        ),
    ]


def register(ctx: Any) -> EigenLoomRuntime:
    """Hermes v0.23.0+ entry point registering all native providers, 12 tools, 4 hooks, command, and skill."""
    rt = EigenLoomRuntime()
    default_headroom = float(
        ctx.get_config(
            "compute_headroom_ratio",
            rt.spec.section("hypergrid_fluid_interpolation")[
                "opportunistic_exploration"
            ]["default_compute_headroom"],
        )
    )
    idle_explore = bool(ctx.get_config("enable_idle_exploration", True))

    # Surface 1, 2, 3 — Native ContextEngine, MemoryProvider, DelegationProvider
    ctx.register_context_engine(rt.context_engine)
    ctx.register_memory_provider(rt.memory_provider)
    ctx.register_delegation_provider(rt.delegation_provider)

    # Surface 4a — 12 Agent Tools
    for tool_name, desc, schema, handler in _build_tool_definitions(rt):
        ctx.register_tool(
            name=tool_name,
            toolset="eigenloom",
            schema=schema,
            handler=handler,
            description=desc,
        )

    # Surface 4b — 4 Lifecycle Hooks
    def _on_session_start(*args: Any, **kwargs: Any) -> None:
        sid = str(kwargs.get("session_id", "default") or "default")
        rt.context_engine.on_session_start(session_id=sid)

    def _pre_tool_call(*args: Any, **kwargs: Any) -> None:
        rt.tool_invocations += 1

    def _post_tool_call(*args: Any, **kwargs: Any) -> None:
        rt.persist_status_snapshot()

    def _on_session_end(*args: Any, **kwargs: Any) -> None:
        if idle_explore:
            replays = rt.store.recent_verified_records()
            rt.bus.coupling_step(
                compute_headroom_ratio=default_headroom,
                replay_records=replays,
            )
        rt.persist_status_snapshot()

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)

    # Surface 4c — /eigenloom Slash Command
    def _slash_command_handler(raw_args: str = "", **_: Any) -> str:
        arg = (raw_args or "").strip().lower()
        if arg.startswith("step"):
            step_res = rt.bus.coupling_step(
                compute_headroom_ratio=default_headroom,
                replay_records=rt.store.recent_verified_records(),
            )
            rt.persist_status_snapshot()
            return (
                f"EigenLoom coupling step #{step_res['coupling_step']} complete | "
                f"Lyapunov delta={step_res['lyapunov_delta']:.6f} | "
                f"SHA probes={step_res['opportunistic_exploration']['candidates_generated']}"
            )
        snap = rt.persist_status_snapshot()
        ledger = snap["grid_ledger_counts"]
        cost = snap["cost_ledger"]
        return (
            f"EigenLoom v{snap['version']} | "
            f"{snap['engine_count']} engines ({snap['sub_engine_count']} sub-engines) | "
            f"{snap['hypergrid_total_dimensions']} HyperGrid dims | "
            f"MRL={snap['matryoshka_prefix_dims']} | "
            f"verified/promoted={ledger.get('verified', 0)}/{ledger.get('promoted', 0)} | "
            f"SHA zero-token probes={cost['sha_zero_token_probes']}"
        )

    ctx.register_command(
        "eigenloom",
        _slash_command_handler,
        description="Inspect EigenLoom 20-engine manifold status or run '/eigenloom step'.",
        args_hint="[status|step]",
    )

    # Surface 4d — Bundled Cybernetics Skill
    skill_path = (
        Path(__file__).resolve().parent
        / "skills"
        / "eigenloom-cybernetics"
        / "SKILL.md"
    )
    if skill_path.is_file() and hasattr(ctx, "register_skill"):
        ctx.register_skill(
            "eigenloom-cybernetics",
            skill_path,
            description="Cybernetic task-fission DAG, maker!=checker verification, and 7-target cartridge compilation.",
        )

    return rt


__all__ = [
    "DelegationRouteError",
    "EigenLoomContextEngine",
    "EigenLoomDelegationProvider",
    "EigenLoomMemoryProvider",
    "EigenLoomRuntime",
    "register",
]

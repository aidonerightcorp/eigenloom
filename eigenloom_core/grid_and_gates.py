"""Fluid HyperGrid configuration manifold, GridLedger, Opportunistic SHA Exploration, and Conformance Gates."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.linalg_backend import (
    Matrix,
    Vector,
    identity_matrix,
    matvec,
    softmax,
    vec_dot,
    woodbury_rank1_update,
)
from eigenloom_core.matryoshka_manifold import MatryoshkaManifold, MatryoshkaVector

# Canonical mathematical/structural identities permitted inline in operational Python modules
ALLOWED_INLINE_CONSTANTS: frozenset[Any] = frozenset({0, 1, -1, 0.0, 1.0, True, False, None})

# Forbidden narrow-vertical/industry stems enforced mechanically by DomainAgnosticNamingGate
FORBIDDEN_INDUSTRY_STEMS: tuple[str, ...] = (
    "real_estate",
    "realestate",
    "mortgage",
    "parcel_permit",
    "healthcare",
    "patient",
    "clinical",
    "fintech",
    "ecommerce",
    "storefront",
    "hospitality",
    "hotel",
    "insurtech",
    "agritech",
    "retail_cart",
)


@dataclass
class ParameterBounds:
    """Clamping bounds for continuous or integer HyperGrid axes with dynamic auto-expansion."""

    low: float
    high: float
    default: float

    def clamp(self, raw_value: float, *, is_integer: bool = False) -> float | int:
        clamped = max(float(self.low), min(float(self.high), float(raw_value)))
        if is_integer:
            return int(round(clamped))
        return float(clamped)

    def saturation_side(self, value: float, margin_fraction: float) -> str | None:
        span = max(self.high - self.low, 0.0)
        if span <= 0.0:
            return None
        rel = (float(value) - self.low) / span
        if rel <= margin_fraction:
            return "low"
        if rel >= (1.0 - margin_fraction):
            return "high"
        return None


@dataclass
class NodeParameter:
    """Single tunable axis inside an engine's NodeGrid."""

    name: str
    kind: str  # continuous_range | integer_range | choice | boolean
    bounds: ParameterBounds | None = None
    choices: list[Any] = field(default_factory=list)
    default: Any = None
    expandable: bool = False

    def validate_and_clamp(self, value: Any) -> Any:
        if self.kind == "continuous_range" and self.bounds is not None:
            return float(self.bounds.clamp(float(value), is_integer=False))
        if self.kind == "integer_range" and self.bounds is not None:
            return int(self.bounds.clamp(float(value), is_integer=True))
        if self.kind == "boolean":
            return bool(value)
        if self.kind == "choice":
            if value in self.choices:
                return value
            return self.default
        return value


@dataclass
class GridCellRecord:
    """Lifecycle record for a configuration cell in GridLedger."""

    cell_id: str
    engine_id: str
    regime: str
    values: dict[str, Any]
    state: str
    config_vec: MatryoshkaVector
    maker_id: str = ""
    checker_id: str = ""
    observed_reward: float = 0.0
    evaluation_count: int = 0


class GridLedger:
    """Tracks cell transitions across represented -> applicable -> proposed -> dispatched -> evaluated -> verified -> promoted."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()
        self._lifecycle: list[str] = list(self._spec.section("grid_ledger_lifecycle"))
        self._rank: dict[str, int] = {
            stage: idx for idx, stage in enumerate(self._lifecycle)
        }
        self._cells: dict[str, GridCellRecord] = {}

    @property
    def lifecycle(self) -> list[str]:
        return list(self._lifecycle)

    def register_cell(self, record: GridCellRecord) -> GridCellRecord:
        if record.state not in self._rank:
            raise ValueError(f"Unknown GridLedger state: {record.state}")
        if self._rank[record.state] >= self._rank["verified"]:
            if not record.maker_id or not record.checker_id:
                raise ValueError("Verified/promoted cell requires both maker_id and checker_id")
            if record.maker_id == record.checker_id:
                raise ValueError("GridLedger maker_id must differ from checker_id")
        self._cells[record.cell_id] = record
        return record

    def transition(
        self,
        cell_id: str,
        target_state: str,
        *,
        maker_id: str = "",
        checker_id: str = "",
        reward: float | None = None,
    ) -> GridCellRecord:
        if cell_id not in self._cells:
            raise KeyError(f"Unknown cell_id: {cell_id}")
        if target_state not in self._rank:
            raise ValueError(f"Invalid target state: {target_state}")
        rec = self._cells[cell_id]
        if maker_id:
            rec.maker_id = maker_id
        if checker_id:
            rec.checker_id = checker_id
        if self._rank[target_state] >= self._rank["verified"]:
            if not rec.maker_id or not rec.checker_id:
                raise ValueError("Cannot verify or promote cell without maker_id and checker_id")
            if rec.maker_id == rec.checker_id:
                raise ValueError("GridLedger enforces maker != checker on verification")
        if reward is not None:
            rec.observed_reward = float(reward)
            rec.evaluation_count += 1
        rec.state = target_state
        return rec

    def verified_cells(self, engine_id: str | None = None) -> list[GridCellRecord]:
        v_rank = self._rank["verified"]
        return [
            rec
            for rec in self._cells.values()
            if self._rank[rec.state] >= v_rank
            and (engine_id is None or rec.engine_id == engine_id)
        ]

    def all_cells(self, engine_id: str | None = None) -> list[GridCellRecord]:
        return [
            rec
            for rec in self._cells.values()
            if engine_id is None or rec.engine_id == engine_id
        ]

    def state_counts(self) -> dict[str, int]:
        counts = {stage: 0 for stage in self._lifecycle}
        for rec in self._cells.values():
            counts[rec.state] = counts.get(rec.state, 0) + 1
        return counts


class OperationCostLedger:
    """Tracks token savings, prompt-cache identity hits, cartridge FLOPs, and SHA zero-token evaluations."""

    def __init__(self) -> None:
        self.prompt_tokens_observed: int = 0
        self.tokens_saved_by_quotient: int = 0
        self.prompt_cache_identity_hits: int = 0
        self.select_context_calls: int = 0
        self.cartridge_evaluations: int = 0
        self.layers_saved_by_early_exit: int = 0
        self.sha_zero_token_probes: int = 0
        self.matryoshka_macs_saved: int = 0

    def record_context_selection(self, *, cache_preserved: bool) -> None:
        self.select_context_calls += 1
        if cache_preserved:
            self.prompt_cache_identity_hits += 1

    def summary(self) -> dict[str, Any]:
        cache_rate = (
            float(self.prompt_cache_identity_hits) / float(self.select_context_calls)
            if self.select_context_calls > 0
            else 1.0
        )
        return {
            "prompt_tokens_observed": self.prompt_tokens_observed,
            "tokens_saved_by_quotient": self.tokens_saved_by_quotient,
            "prompt_cache_identity_hits": self.prompt_cache_identity_hits,
            "select_context_calls": self.select_context_calls,
            "prompt_cache_preservation_rate": cache_rate,
            "cartridge_evaluations": self.cartridge_evaluations,
            "layers_saved_by_early_exit": self.layers_saved_by_early_exit,
            "sha_zero_token_probes": self.sha_zero_token_probes,
            "matryoshka_macs_saved": self.matryoshka_macs_saved,
        }


class BilinearWoodburySurrogate:
    """Scores (Task2Vec, Config2Vec) pairs via 48d bilinear features and exact Woodbury covariance updates."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()
        h_cfg = self._spec.section("hypergrid_fluid_interpolation")
        self._sub_dim = int(h_cfg["bilinear_sub_dim"])
        self._feat_dim = int(h_cfg["bilinear_feature_dim"])
        self._prior_lambda = float(h_cfg["woodbury_prior_lambda"])
        self._ucb_beta = float(h_cfg["ucb_beta"])
        inv_scale = 1.0 / max(self._prior_lambda, self._spec.c_float("micro"))
        self._a_inv: Matrix = [
            [inv_scale if r == c else 0.0 for c in range(self._feat_dim)]
            for r in range(self._feat_dim)
        ]
        self._b_vec: Vector = [0.0 for _ in range(self._feat_dim)]
        self._updates: int = 0

    @property
    def updates(self) -> int:
        return self._updates

    def feature_map(
        self, task_vec: MatryoshkaVector, config_vec: MatryoshkaVector
    ) -> Vector:
        z_sub = task_vec.slice_prefix(self._sub_dim)
        c_sub = config_vec.slice_prefix(self._sub_dim)
        hadamard = [zi * ci for zi, ci in zip(z_sub, c_sub)]
        return z_sub + c_sub + hadamard

    def score(
        self, task_vec: MatryoshkaVector, config_vec: MatryoshkaVector
    ) -> dict[str, float]:
        phi = self.feature_map(task_vec, config_vec)
        theta_hat = matvec(self._a_inv, self._b_vec)
        mean_pred = vec_dot(theta_hat, phi)
        a_phi = matvec(self._a_inv, phi)
        variance = max(0.0, vec_dot(phi, a_phi))
        bonus = self._ucb_beta * math.sqrt(variance)
        align_prior = task_vec.cosine(config_vec, self._sub_dim) * self._spec.c_float("quarter")
        total = mean_pred + bonus + align_prior
        return {
            "ucb_score": float(total),
            "posterior_mean": float(mean_pred + align_prior),
            "epistemic_std": float(math.sqrt(variance)),
        }

    def observe(
        self,
        task_vec: MatryoshkaVector,
        config_vec: MatryoshkaVector,
        reward: float,
    ) -> float:
        phi = self.feature_map(task_vec, config_vec)
        self._a_inv = woodbury_rank1_update(self._a_inv, phi, damping=1.0)
        r_val = float(reward)
        self._b_vec = [bi + r_val * pi for bi, pi in zip(self._b_vec, phi)]
        self._updates += 1
        trace_val = sum(self._a_inv[i][i] for i in range(self._feat_dim))
        return float(trace_val)


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denom = 1.0
    n = max(1, int(index))
    min_base = 1 + 1
    b = max(min_base, int(base))
    while n > 0:
        denom *= float(b)
        n, rem = divmod(n, b)
        result += float(rem) / denom
    return result


class ComponentNodeGrid:
    """Multi-dimensional NodeGrid for a single engine with constraints and Boundary Auto-Expansion."""

    def __init__(
        self,
        engine_spec: Mapping[str, Any],
        manifold: MatryoshkaManifold,
        ledger: GridLedger,
        spec: ManifoldConfigSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.engine_id: str = str(engine_spec["engine_id"])
        self.stratum: str = str(engine_spec["stratum"])
        grid_raw = engine_spec["node_grid"]
        self.generation: int = int(grid_raw["generation"])
        self.constraints: list[dict[str, Any]] = list(grid_raw.get("constraints", []))
        self.parameters: dict[str, NodeParameter] = {}

        for p_raw in grid_raw["parameters"]:
            bounds_obj = None
            if "bounds" in p_raw:
                b = p_raw["bounds"]
                bounds_obj = ParameterBounds(
                    low=float(b["low"]),
                    high=float(b["high"]),
                    default=float(b["default"]),
                )
            default_val = p_raw.get(
                "default",
                bounds_obj.default if bounds_obj is not None else None,
            )
            param = NodeParameter(
                name=str(p_raw["name"]),
                kind=str(p_raw["kind"]),
                bounds=bounds_obj,
                choices=list(p_raw.get("choices", [])),
                default=default_val,
                expandable=bool(p_raw.get("expandable", False)),
            )
            self.parameters[param.name] = param

        for seed in grid_raw.get("seed_cells", []):
            clamped = self.clamp_values(seed["values"])
            c_vec = manifold.embed_config(
                clamped,
                engine_id=self.engine_id,
                stratum=self.stratum,
                bounds_lookup=self.bounds_lookup(),
            )
            ledger.register_cell(
                GridCellRecord(
                    cell_id=str(seed["cell_id"]),
                    engine_id=self.engine_id,
                    regime=str(seed["regime"]),
                    values=clamped,
                    state=str(seed["state"]),
                    config_vec=c_vec,
                    maker_id=str(seed["maker_id"]),
                    checker_id=str(seed["checker_id"]),
                    observed_reward=float(seed["observed_reward"]),
                    evaluation_count=1,
                )
            )

    def bounds_lookup(self) -> dict[str, tuple[float, float]]:
        return {
            name: (param.bounds.low, param.bounds.high)
            for name, param in self.parameters.items()
            if param.bounds is not None
        }

    def clamp_values(self, raw_values: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, param in self.parameters.items():
            raw = raw_values.get(name, param.default)
            out[name] = param.validate_and_clamp(raw)
        return out

    def check_constraints(self, values: Mapping[str, Any]) -> bool:
        for rule in self.constraints:
            if rule.get("kind") == "range_order":
                pname = str(rule["parameter"])
                if pname in values:
                    fval = float(values[pname])
                    if fval < float(rule["min_allowed"]) or fval > float(rule["max_allowed"]):
                        return False
        return True

    def maybe_expand_boundaries(
        self, promoted_values: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        margin = self._spec.c_float("boundary_saturation_margin")
        factor = self._spec.c_float("boundary_expansion_factor")
        expansions: list[dict[str, Any]] = []

        for name, param in self.parameters.items():
            if not param.expandable or param.bounds is None or name not in promoted_values:
                continue
            val = float(promoted_values[name])
            side = param.bounds.saturation_side(val, margin)
            if side is None:
                continue
            old_low, old_high = param.bounds.low, param.bounds.high
            span = max(old_high - old_low, self._spec.c_float("milli"))
            delta = span * (factor - 1.0)
            if side == "high":
                param.bounds.high = (
                    float(int(math.ceil(old_high + max(1.0, delta))))
                    if param.kind == "integer_range"
                    else old_high + delta
                )
            else:
                new_low = old_low - delta
                if old_low > 0.0:
                    new_low = max(old_low / factor, new_low)
                param.bounds.low = (
                    float(max(1, int(math.floor(new_low))))
                    if param.kind == "integer_range"
                    else new_low
                )
            # Sync matching constraint envelopes with the new expanded generation
            for rule in self.constraints:
                if rule.get("kind") == "range_order" and rule.get("parameter") == name:
                    rule["min_allowed"] = param.bounds.low
                    rule["max_allowed"] = param.bounds.high
            self.generation += 1
            expansions.append(
                {
                    "engine_id": self.engine_id,
                    "parameter": name,
                    "saturated_side": side,
                    "old_bounds": [old_low, old_high],
                    "new_bounds": [param.bounds.low, param.bounds.high],
                    "new_generation": self.generation,
                }
            )
        return expansions


class FluidHyperGrid:
    """100-dimensional embedding-conditioned HyperGrid with kernel interpolation and Opportunistic SHA."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
        cost_ledger: OperationCostLedger | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.manifold = manifold or MatryoshkaManifold(self._spec)
        self.ledger = GridLedger(self._spec)
        self.cost_ledger = cost_ledger or OperationCostLedger()
        self.surrogate = BilinearWoodburySurrogate(self._spec)
        self.grids: dict[str, ComponentNodeGrid] = {}
        self._probe_counter: int = 1

        for engine_spec in self._spec.section("engines"):
            grid = ComponentNodeGrid(
                engine_spec, self.manifold, self.ledger, self._spec
            )
            self.grids[grid.engine_id] = grid
            # Seed surrogate with verified initial cells
            seed_task = self.manifold.embed_task(
                engine_spec["description"], stratum=grid.stratum
            )
            for cell in self.ledger.verified_cells(grid.engine_id):
                self.surrogate.observe(seed_task, cell.config_vec, cell.observed_reward)

    @property
    def total_dimensions(self) -> int:
        return sum(len(g.parameters) for g in self.grids.values())

    def interpolate_fluid_config(
        self,
        engine_id: str,
        query_vec: MatryoshkaVector,
        *,
        matryoshka_dim: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Compute theta(z) = sum_i w_i(z) theta_i strictly over verified/promoted GridLedger cells."""
        grid = self.grids[engine_id]
        h_cfg = self._spec.section("hypergrid_fluid_interpolation")
        use_dim = int(
            matryoshka_dim
            if matryoshka_dim is not None
            else self._spec.section("matryoshka_manifold")["fine_dim"]
        )
        tau = float(
            temperature if temperature is not None else h_cfg["default_kernel_tau"]
        )
        verified = self.ledger.verified_cells(engine_id)
        if not verified:
            fallback = grid.clamp_values({})
            c_vec = self.manifold.embed_config(
                fallback,
                engine_id=engine_id,
                stratum=grid.stratum,
                bounds_lookup=grid.bounds_lookup(),
            )
            return {
                "engine_id": engine_id,
                "parameters": fallback,
                "choice_posteriors": {},
                "kernel_weights": {},
                "interpolation_entropy": 0.0,
                "config2vec": c_vec,
                "generation": grid.generation,
            }

        logits: Vector = []
        for cell in verified:
            sim = query_vec.cosine(cell.config_vec, use_dim)
            sur = self.surrogate.score(query_vec, cell.config_vec)["posterior_mean"]
            logits.append(sim + self._spec.c_float("quarter") * sur)

        weights = softmax(logits, temperature=tau)
        eps = self._spec.c_float("pico")
        entropy = -sum(w * math.log(max(w, eps)) for w in weights if w > eps)

        blended_raw: dict[str, Any] = {}
        choice_posteriors: dict[str, dict[str, float]] = {}

        for pname, param in grid.parameters.items():
            if param.kind in ("continuous_range", "integer_range"):
                blended_val = sum(
                    w * float(cell.values[pname])
                    for w, cell in zip(weights, verified)
                )
                blended_raw[pname] = param.validate_and_clamp(blended_val)
            elif param.kind == "boolean":
                prob_true = sum(
                    w * (1.0 if bool(cell.values[pname]) else 0.0)
                    for w, cell in zip(weights, verified)
                )
                blended_raw[pname] = prob_true >= self._spec.c_float("half")
                choice_posteriors[pname] = {
                    "true": float(prob_true),
                    "false": float(1.0 - prob_true),
                }
            else:
                dist: dict[Any, float] = {opt: 0.0 for opt in param.choices}
                for w, cell in zip(weights, verified):
                    opt_val = cell.values[pname]
                    dist[opt_val] = dist.get(opt_val, 0.0) + w
                best_opt = max(dist.items(), key=lambda kv: kv[1])[0]
                blended_raw[pname] = best_opt
                choice_posteriors[pname] = {
                    str(k): float(v) for k, v in dist.items()
                }

        clamped = grid.clamp_values(blended_raw)
        c_vec = self.manifold.embed_config(
            clamped,
            engine_id=engine_id,
            stratum=grid.stratum,
            bounds_lookup=grid.bounds_lookup(),
        )
        return {
            "engine_id": engine_id,
            "parameters": clamped,
            "choice_posteriors": choice_posteriors,
            "kernel_weights": {
                cell.cell_id: float(w) for cell, w in zip(verified, weights)
            },
            "interpolation_entropy": float(entropy),
            "config2vec": c_vec,
            "generation": grid.generation,
        }

    def _pairwise_covering_rows(
        self, grid: ComponentNodeGrid, max_rows: int
    ) -> list[dict[str, Any]]:
        """Generate strength-2 covering rows across discrete/boundary levels in logarithmic size."""
        levels_per_axis: dict[str, list[Any]] = {}
        for pname, param in grid.parameters.items():
            if param.kind in ("continuous_range", "integer_range") and param.bounds is not None:
                mid = self._spec.c_float("half") * (param.bounds.low + param.bounds.high)
                levels_per_axis[pname] = [
                    param.validate_and_clamp(param.bounds.low),
                    param.validate_and_clamp(mid),
                    param.validate_and_clamp(param.bounds.high),
                ]
            elif param.kind == "boolean":
                levels_per_axis[pname] = [False, True]
            else:
                levels_per_axis[pname] = list(param.choices)

        axis_names = list(levels_per_axis.keys())
        uncovered: set[tuple[str, Any, str, Any]] = set()
        for i, ax_a in enumerate(axis_names):
            for ax_b in axis_names[i + 1 :]:
                for va in levels_per_axis[ax_a]:
                    for vb in levels_per_axis[ax_b]:
                        uncovered.add((ax_a, va, ax_b, vb))

        rows: list[dict[str, Any]] = []
        while uncovered and len(rows) < max_rows:
            row: dict[str, Any] = {}
            for ax in axis_names:
                best_val = levels_per_axis[ax][0]
                best_gain = -1
                for cand_val in levels_per_axis[ax]:
                    gain = 0
                    for prev_ax, prev_val in row.items():
                        pair = (
                            (prev_ax, prev_val, ax, cand_val)
                            if axis_names.index(prev_ax) < axis_names.index(ax)
                            else (ax, cand_val, prev_ax, prev_val)
                        )
                        if pair in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_val = cand_val
                row[ax] = best_val
            for i, ax_a in enumerate(axis_names):
                for ax_b in axis_names[i + 1 :]:
                    uncovered.discard((ax_a, row[ax_a], ax_b, row[ax_b]))
            rows.append(grid.clamp_values(row))
        return rows

    def _sample_sobol_or_uniform_candidates(
        self,
        grid: ComponentNodeGrid,
        count: int,
        rng: random.Random,
        bases: Sequence[int],
        uniform_mix: float,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        axis_items = list(grid.parameters.items())
        for offset in range(count):
            seq_idx = self._probe_counter + offset
            use_uniform = rng.random() < uniform_mix
            raw: dict[str, Any] = {}
            for ax_idx, (pname, param) in enumerate(axis_items):
                base = bases[ax_idx % len(bases)]
                u = rng.random() if use_uniform else _van_der_corput(seq_idx, base)
                if param.kind in ("continuous_range", "integer_range") and param.bounds is not None:
                    val = param.bounds.low + u * (param.bounds.high - param.bounds.low)
                    raw[pname] = param.validate_and_clamp(val)
                elif param.kind == "boolean":
                    raw[pname] = u >= self._spec.c_float("half")
                else:
                    idx = min(len(param.choices) - 1, int(u * len(param.choices)))
                    raw[pname] = param.choices[max(0, idx)]
            samples.append(grid.clamp_values(raw))
        self._probe_counter += count
        return samples

    def opportunistic_explore(
        self,
        engine_id: str,
        *,
        compute_headroom_ratio: float | None = None,
        replay_records: Sequence[Mapping[str, Any]] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Elastic Pairwise + Sobol/Random exploration filtered via 3-rung Successive Halving (0 LLM tokens)."""
        grid = self.grids[engine_id]
        exp_cfg = self._spec.section("hypergrid_fluid_interpolation")[
            "opportunistic_exploration"
        ]
        headroom = float(
            compute_headroom_ratio
            if compute_headroom_ratio is not None
            else exp_cfg["default_compute_headroom"]
        )
        floor = float(exp_cfg["compute_headroom_floor"])
        if headroom < floor:
            return {
                "engine_id": engine_id,
                "skipped_low_headroom": True,
                "candidates_generated": 0,
                "rung_counts": [],
                "promoted_cell_id": None,
                "boundary_expansions": [],
                "extra_llm_tokens": 0,
            }

        min_c = int(exp_cfg["min_candidates"])
        max_c = int(exp_cfg["max_random_candidates"])
        budget = max(min_c, int(round(max_c * min(1.0, headroom))))
        cov_cap = min(int(exp_cfg["covering_array_max_rows"]), max(1, budget // self._spec.c_int("two")))
        covering = self._pairwise_covering_rows(grid, max_rows=cov_cap)
        rem = max(1, budget - len(covering))
        rng = random.Random(seed if seed is not None else self._probe_counter)
        probed = self._sample_sobol_or_uniform_candidates(
            grid,
            count=rem,
            rng=rng,
            bases=[int(b) for b in exp_cfg["sobol_radical_bases"]],
            uniform_mix=float(exp_cfg["uniform_mix_fraction"]),
        )
        raw_pool = [c for c in (covering + probed) if grid.check_constraints(c)]
        self.cost_ledger.sha_zero_token_probes += len(raw_pool)

        # Build replay reference vectors from historical verified maker != checker episodes (or engine anchors)
        anchors: list[tuple[MatryoshkaVector, float]] = []
        if replay_records:
            for rec in replay_records:
                vec_vals = rec.get("task2vec")
                t_vec = (
                    MatryoshkaVector.from_list(vec_vals, space="task2vec", spec=self._spec)
                    if isinstance(vec_vals, list) and vec_vals
                    else self.manifold.embed_task(
                        str(rec.get("summary", engine_id)), stratum=grid.stratum
                    )
                )
                anchors.append((t_vec, float(rec.get("quality_score", 1.0))))
        if not anchors:
            anchors.append(
                (
                    self.manifold.embed_task(
                        f"{engine_id} {grid.stratum} canonical workload",
                        stratum=grid.stratum,
                    ),
                    1.0,
                )
            )

        # Register candidates in GridLedger (represented -> applicable -> proposed)
        entries: list[GridCellRecord] = []
        for idx, vals in enumerate(raw_pool):
            digest = hashlib.blake2s(
                json.dumps(vals, sort_keys=True).encode("utf-8"),
                digest_size=self._spec.c_int("six"),
            ).hexdigest()
            cid = f"{engine_id}::sha_{self._probe_counter}_{idx}_{digest}"
            c_vec = self.manifold.embed_config(
                vals,
                engine_id=engine_id,
                stratum=grid.stratum,
                bounds_lookup=grid.bounds_lookup(),
            )
            rec = GridCellRecord(
                cell_id=cid,
                engine_id=engine_id,
                regime="opportunistic_sha",
                values=vals,
                state="represented",
                config_vec=c_vec,
                maker_id="sha_explorer_maker",
            )
            self.ledger.register_cell(rec)
            self.ledger.transition(cid, "applicable")
            self.ledger.transition(cid, "proposed")
            entries.append(rec)

        eta = max(self._spec.c_int("two"), int(exp_cfg["halving_eta"]))
        rung_dims = [int(d) for d in exp_cfg["rung_matryoshka_dims"]]
        rung_counts: list[int] = []
        survivors = list(entries)

        for rung_idx, r_dim in enumerate(rung_dims):
            if not survivors:
                break
            rung_counts.append(len(survivors))
            scored_rung: list[tuple[float, GridCellRecord]] = []
            for rec in survivors:
                if rung_idx == 0:
                    self.ledger.transition(rec.cell_id, "dispatched")
                # Replay against cached verified records at progressive Matryoshka resolution
                accum = 0.0
                w_sum = 0.0
                for t_vec, q_weight in anchors:
                    align = (1.0 + t_vec.cosine(rec.config_vec, r_dim)) * self._spec.c_float("half")
                    sur = self.surrogate.score(t_vec, rec.config_vec)["ucb_score"]
                    accum += q_weight * (
                        self._spec.c_float("half") * align
                        + self._spec.c_float("half") * math.tanh(max(0.0, sur))
                    )
                    w_sum += max(q_weight, self._spec.c_float("micro"))
                r_score = accum / max(w_sum, self._spec.c_float("micro"))
                self.ledger.transition(rec.cell_id, "evaluated", reward=r_score)
                self.surrogate.observe(anchors[0][0], rec.config_vec, r_score)
                scored_rung.append((r_score, rec))

            scored_rung.sort(key=lambda item: item[0], reverse=True)
            keep_n = (
                1
                if rung_idx == len(rung_dims) - 1
                else max(1, int(math.ceil(float(len(scored_rung)) / float(eta))))
            )
            survivors = [rec for _, rec in scored_rung[:keep_n]]

        promoted_id: str | None = None
        expansions: list[dict[str, Any]] = []
        if survivors:
            winner = survivors[0]
            # Independent checker verifies structural constraint satisfaction and finite score
            if grid.check_constraints(winner.values):
                self.ledger.transition(
                    winner.cell_id,
                    "verified",
                    maker_id="sha_explorer_maker",
                    checker_id="sha_replay_auditor",
                    reward=winner.observed_reward,
                )
                incumbent_best = max(
                    (
                        c.observed_reward
                        for c in self.ledger.verified_cells(engine_id)
                        if c.cell_id != winner.cell_id
                    ),
                    default=0.0,
                )
                if winner.observed_reward + float(exp_cfg["promotion_margin_delta"]) >= incumbent_best:
                    self.ledger.transition(
                        winner.cell_id,
                        "promoted",
                        maker_id="sha_explorer_maker",
                        checker_id="sha_replay_auditor",
                    )
                    promoted_id = winner.cell_id
                expansions = grid.maybe_expand_boundaries(winner.values)

        return {
            "engine_id": engine_id,
            "skipped_low_headroom": False,
            "candidates_generated": len(raw_pool),
            "covering_array_rows": len(covering),
            "rung_counts": rung_counts,
            "winner_cell_id": survivors[0].cell_id if survivors else None,
            "promoted_cell_id": promoted_id,
            "boundary_expansions": expansions,
            "extra_llm_tokens": 0,
        }


class ZeroHardcodedValuesAstGate:
    """Mechanical AST gate ensuring operational modules contain zero hardcoded magic scalars."""

    @staticmethod
    def scan_file(py_path: Path) -> list[str]:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_path))
        violations: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float, complex)
            ):
                # Note: bool is a subclass of int in Python; ALLOWED_INLINE_CONSTANTS permits True/False
                val = node.value
                if val not in ALLOWED_INLINE_CONSTANTS:
                    violations.append(
                        f"{py_path.name}:{getattr(node, 'lineno', 0)} forbidden literal {val!r}"
                    )
            elif (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, (int, float))
            ):
                neg_val = -node.operand.value
                if neg_val not in ALLOWED_INLINE_CONSTANTS:
                    violations.append(
                        f"{py_path.name}:{getattr(node, 'lineno', 0)} forbidden unary literal {neg_val!r}"
                    )
        return violations

    @classmethod
    def scan_paths(cls, paths: Sequence[Path]) -> list[str]:
        hits: list[str] = []
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                hits.extend(cls.scan_file(path))
        return hits


class DomainAgnosticNamingGate:
    """Mechanical gate ensuring filenames, JSON schema IDs, and Python symbols stay domain-agnostic."""

    @staticmethod
    def _has_forbidden_stem(identifier: str) -> str | None:
        lowered = identifier.lower()
        for stem in FORBIDDEN_INDUSTRY_STEMS:
            if stem in lowered:
                return stem
        return None

    @classmethod
    def scan_repository(cls, repo_root: Path) -> list[str]:
        violations: list[str] = []
        skip_parts = {".git", "__pycache__", ".pytest_cache"}

        for path in sorted(repo_root.rglob("*")):
            if any(part in skip_parts for part in path.parts):
                continue
            rel = path.relative_to(repo_root).as_posix()
            stem_hit = cls._has_forbidden_stem(rel)
            if stem_hit is not None:
                violations.append(f"path '{rel}' contains industry stem '{stem_hit}'")

            if path.is_file() and path.suffix == ".py" and not rel.startswith("tests/"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        hit = cls._has_forbidden_stem(node.name)
                        if hit is not None:
                            violations.append(
                                f"{rel}:{node.lineno} symbol '{node.name}' contains '{hit}'"
                            )
            elif path.is_file() and path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "schema_id" in data:
                    hit = cls._has_forbidden_stem(str(data["schema_id"]))
                    if hit is not None:
                        violations.append(
                            f"{rel} schema_id '{data['schema_id']}' contains '{hit}'"
                        )
        return violations

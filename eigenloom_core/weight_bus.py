"""20-Engine Cybernetic Weight-Space Bus, Spectral Diagnostics, Zero-Drift Surgical Edits, Morphism, and EVSI."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigenloom_core.cartridge_compiler import CartridgeCompiler
from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.grid_and_gates import (
    FluidHyperGrid,
    OperationCostLedger,
)
from eigenloom_core.linalg_backend import (
    Matrix,
    Vector,
    add_scaled,
    copy_matrix,
    frobenius_norm,
    identity_matrix,
    l2_normalize,
    matmul,
    matvec,
    orthogonal_procrustes,
    outer_product,
    softmax,
    svd_compact,
    transpose,
    vec_dot,
    vec_norm,
    woodbury_rank1_update,
    zeros_matrix,
)
from eigenloom_core.matryoshka_manifold import MatryoshkaManifold, MatryoshkaVector


@dataclass
class ParameterizedWeightSpaceEngine:
    """Single continuously improving engine governed by FluidHyperGrid, anchor basis, and Weight2Vec."""

    index: int
    engine_id: str
    stratum: str
    primary_timescale: str
    description: str
    sub_engines: list[str]
    weights: Matrix
    anchor_inputs: Matrix  # shape [anchor_rank, cols], orthonormal rows protecting core skills
    anchor_targets: Matrix  # baseline W @ anchor_inputs^T
    hopfield_patterns: list[Vector] = field(default_factory=list)
    state_energy: float = 0.0
    edit_count: int = 0
    last_anchor_drift: float = 0.0

    def compute_anchor_drift(self) -> float:
        if not self.anchor_inputs:
            return 0.0
        current = matmul(self.weights, transpose(self.anchor_inputs))
        diff = add_scaled(current, self.anchor_targets, scale=-1.0)
        return float(frobenius_norm(diff))


class CyberneticWeightBus:
    """Coordinates all 20 Functional Engines, 72 Sub-Engines, 5 Timescales, and 20x20 Sparse Jacobian."""

    def __init__(
        self,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
        cost_ledger: OperationCostLedger | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.cost_ledger = cost_ledger or OperationCostLedger()
        self.manifold = manifold or MatryoshkaManifold(self._spec)
        self.hypergrid = FluidHyperGrid(self._spec, self.manifold, self.cost_ledger)
        self.compiler = CartridgeCompiler(self._spec, self.manifold)

        self.engines: dict[str, ParameterizedWeightSpaceEngine] = {}
        self.engine_order: list[str] = []
        self._init_engines()
        self.jacobian: Matrix = self._init_jacobian()
        self.coupling_steps: int = 0

    def _init_engines(self) -> None:
        tenth = self._spec.c_float("tenth")
        half = self._spec.c_float("half")
        for eng_raw in self._spec.section("engines"):
            idx = int(eng_raw["index"])
            eid = str(eng_raw["engine_id"])
            rows, cols = int(eng_raw["matrix_shape"][0]), int(eng_raw["matrix_shape"][1])
            anchor_rank = int(eng_raw["anchor_subspace_rank"])

            # Deterministic structured spectrum per engine so spectral diagnostics are rich & reproducible
            w_mat = zeros_matrix(rows, cols)
            for r in range(rows):
                for c in range(cols):
                    diag_term = (1.0 / float(r + 1)) if r == c else 0.0
                    harmonic = tenth * half * math.sin(float((idx + 1) * (r + 1) + (c + 1)))
                    w_mat[r][c] = diag_term + harmonic

            # Orthonormal canonical anchor inputs protecting core behavior
            anchors: Matrix = []
            for a_idx in range(min(anchor_rank, cols)):
                row_vec = [1.0 if c == a_idx else 0.0 for c in range(cols)]
                anchors.append(row_vec)
            targets = matmul(w_mat, transpose(anchors))

            engine = ParameterizedWeightSpaceEngine(
                index=idx,
                engine_id=eid,
                stratum=str(eng_raw["stratum"]),
                primary_timescale=str(eng_raw["primary_timescale"]),
                description=str(eng_raw["description"]),
                sub_engines=[str(s) for s in eng_raw["sub_engines"]],
                weights=w_mat,
                anchor_inputs=anchors,
                anchor_targets=targets,
                state_energy=half,
            )
            self.engines[eid] = engine
            self.engine_order.append(eid)

    def _init_jacobian(self) -> Matrix:
        n = len(self.engine_order)
        idx_map = {eid: idx for idx, eid in enumerate(self.engine_order)}
        jac = zeros_matrix(n, n)
        for edge in self._spec.section("sparse_coupling_edges"):
            src = idx_map[str(edge["source"])]
            dst = idx_map[str(edge["target"])]
            jac[dst][src] = float(edge["weight"])
        return jac

    @property
    def sub_engine_count(self) -> int:
        return sum(len(eng.sub_engines) for eng in self.engines.values())

    def analyze_spectrum(
        self,
        *,
        engine_id: str | None = None,
        matrix: Matrix | None = None,
        sample_inputs: Matrix | None = None,
    ) -> dict[str, Any]:
        """Run Marchenko-Pastur RMT noise-floor, effective-rank, power-law alpha, and Wanda saliency."""
        if matrix is not None:
            target_mat = matrix
            stratum = "adaptation_and_economics"
            label = engine_id or "custom_matrix"
        else:
            eid = engine_id or self.engine_order[0]
            eng = self.engines[eid]
            target_mat = eng.weights
            stratum = eng.stratum
            label = eid

        rows = len(target_mat)
        cols = len(target_mat[0]) if rows > 0 else 0
        eps = self._spec.c_float("pico")
        _, s_vals, _ = svd_compact(target_mat)

        # Eigenvalues of Gram matrix (1/cols) W W^T
        aspect_q = float(min(rows, cols)) / float(max(1, max(rows, cols)))
        eigs = [(sv * sv) / float(max(1, cols)) for sv in s_vals]
        sorted_eigs = sorted(eigs)
        mid_idx = len(sorted_eigs) // self._spec.c_int("two")
        median_eig = sorted_eigs[mid_idx] if sorted_eigs else 0.0
        # Marchenko-Pastur upper bulk edge lambda_+ = sigma^2 * (1 + sqrt(q))^2
        mp_edge = median_eig * ((1.0 + math.sqrt(aspect_q)) ** self._spec.c_int("two"))
        signal_spikes = sum(1 for ev in eigs if ev > mp_edge and ev > eps)

        # Roy-Vetterli Shannon effective rank & stable rank
        s_sum = sum(s_vals)
        probs = [sv / max(s_sum, eps) for sv in s_vals if sv > eps]
        shannon_h = -sum(p * math.log(max(p, eps)) for p in probs)
        effective_rank = math.exp(shannon_h) if probs else 0.0
        frob = frobenius_norm(target_mat)
        top_s = s_vals[0] if s_vals else 0.0
        stable_rank = (frob * frob) / max(top_s * top_s, eps) if top_s > eps else 0.0

        # WeightWatcher power-law exponent alpha via Hill estimator over positive tail eigenvalues
        pos_eigs = [ev for ev in eigs if ev > eps]
        two = self._spec.c_int("two")
        if len(pos_eigs) >= two:
            tail_k = max(two, len(pos_eigs) // two)
            tail = sorted(pos_eigs, reverse=True)[:tail_k]
            xmin = max(tail[-1], eps)
            log_sum = sum(math.log(max(ev / xmin, 1.0 + eps)) for ev in tail[:-1])
            alpha = 1.0 + float(max(1, len(tail) - 1)) / max(log_sum, self._spec.c_float("milli"))
        else:
            alpha = float(two)

        # Wanda input-aware saliency S_ij = |W_ij| * ||X_{:, j}||_2
        if sample_inputs and sample_inputs[0]:
            col_norms = [
                math.sqrt(
                    sum(
                        float(sample_inputs[r][c]) ** two
                        for r in range(len(sample_inputs))
                        if c < len(sample_inputs[r])
                    )
                )
                for c in range(cols)
            ]
        else:
            col_norms = [1.0 for _ in range(cols)]

        wanda_row_saliency = [
            sum(abs(float(target_mat[r][c])) * col_norms[c] for c in range(cols))
            for r in range(rows)
        ]
        w2v = self.manifold.embed_weights([target_mat], stratum=stratum, tag=label)

        return {
            "target": label,
            "shape": [rows, cols],
            "singular_values": s_vals,
            "marchenko_pastur_bulk_edge": float(mp_edge),
            "signal_components_above_noise": int(signal_spikes),
            "effective_rank": float(effective_rank),
            "stable_rank": float(stable_rank),
            "powerlaw_alpha": float(alpha),
            "wanda_row_saliency": wanda_row_saliency,
            "weight2vec_16d": w2v.slice_prefix(
                self._spec.section("matryoshka_manifold")["coarse_dim"]
            ),
        }

    def _project_onto_anchor_nullspace(
        self, delta_w: Matrix, anchor_rows: Matrix
    ) -> Matrix:
        """Project right-space of delta_w onto orthogonal complement of orthonormalized anchor_rows."""
        if not anchor_rows or not delta_w:
            return copy_matrix(delta_w)
        cols = len(delta_w[0])
        eps = self._spec.c_float("nano")

        # Orthonormalize anchor rows via Gram-Schmidt
        basis: list[Vector] = []
        for row in anchor_rows:
            v = [float(x) for x in row[:cols]]
            if len(v) < cols:
                v.extend([0.0 for _ in range(cols - len(v))])
            for b in basis:
                proj = vec_dot(v, b)
                v = [vi - proj * bi for vi, bi in zip(v, b)]
            nrm = vec_norm(v)
            if nrm > eps:
                basis.append([vi / nrm for vi in v])

        if not basis:
            return copy_matrix(delta_w)

        # P_perp = I - sum_k b_k b_k^T; delta_proj = delta_w @ P_perp
        p_perp = identity_matrix(cols)
        for b in basis:
            p_perp = add_scaled(p_perp, outer_product(b, b), scale=-1.0)
        return matmul(delta_w, p_perp)

    def surgical_edit(
        self,
        engine_id: str,
        *,
        mode: str = "svd_nullspace",
        input_vec: Sequence[float] | None = None,
        target_vec: Sequence[float] | None = None,
        counterexample_probes: Sequence[Sequence[float]] | None = None,
        step_gain: float | None = None,
    ) -> dict[str, Any]:
        """Apply surgical weight update with verified 0.0 anchor drift and CEGAR counterexample healing."""
        eng = self.engines[engine_id]
        rows = len(eng.weights)
        cols = len(eng.weights[0])
        gain = float(step_gain if step_gain is not None else self._spec.c_float("quarter"))
        tol = self._spec.c_float("anchor_drift_zero_tol")

        # Default stimulus orthogonal to canonical first anchor dimensions
        if input_vec is not None:
            x_vec = [float(v) for v in input_vec[:cols]]
            if len(x_vec) < cols:
                x_vec.extend([0.0 for _ in range(cols - len(x_vec))])
        else:
            x_vec = [0.0 for _ in range(cols)]
            x_vec[-1] = 1.0

        current_out = matvec(eng.weights, x_vec)
        if target_vec is not None:
            y_vec = [float(v) for v in target_vec[:rows]]
            if len(y_vec) < rows:
                y_vec.extend([0.0 for _ in range(rows - len(y_vec))])
        else:
            y_vec = [val + self._spec.c_float("tenth") for val in current_out]

        residual = [yi - ci for yi, ci in zip(y_vec, current_out)]
        raw_delta = outer_product(residual, x_vec, scale=gain)

        if mode == "lora_xs":
            u_m, s_v, vt_m = svd_compact(eng.weights)
            rank_r = min(self._spec.c_int("two"), len(s_v))
            u_r = [[u_m[r][k] for k in range(rank_r)] for r in range(rows)]
            vt_r = [vt_m[k] for k in range(rank_r)]
            core_r = matmul(matmul(transpose(u_r), raw_delta), transpose(vt_r))
            shaped_delta = matmul(matmul(u_r, core_r), vt_r)
        elif mode == "recursive_woodbury":
            cov_inv = woodbury_rank1_update(identity_matrix(cols), x_vec, damping=1.0)
            kx = matvec(cov_inv, x_vec)
            shaped_delta = outer_product(residual, kx, scale=gain)
        elif mode == "procrustes_align":
            candidate = add_scaled(eng.weights, raw_delta, scale=1.0)
            omega = orthogonal_procrustes(eng.weights, candidate)
            rotated = matmul(omega, eng.weights)
            shaped_delta = add_scaled(rotated, eng.weights, scale=-1.0)
        elif mode == "hopfield_write":
            eng.hopfield_patterns.append(l2_normalize(x_vec))
            shaped_delta = zeros_matrix(rows, cols)
        elif mode == "svd_nullspace":
            shaped_delta = raw_delta
        else:
            raise ValueError(f"Unsupported surgical_edit mode: {mode}")

        protected_anchors = list(eng.anchor_inputs)
        projected_delta = self._project_onto_anchor_nullspace(
            shaped_delta, protected_anchors
        )

        # CEGAR loop: check counterexample probes; if any probe experiences unintended drift, refine nullspace
        cegar_refinements = 0
        if counterexample_probes:
            for probe in counterexample_probes:
                p_vec = [float(v) for v in probe[:cols]]
                if len(p_vec) < cols:
                    p_vec.extend([0.0 for _ in range(cols - len(p_vec))])
                probe_shift = vec_norm(matvec(projected_delta, p_vec))
                if probe_shift > tol:
                    protected_anchors.append(p_vec)
                    projected_delta = self._project_onto_anchor_nullspace(
                        shaped_delta, protected_anchors
                    )
                    cegar_refinements += 1

        eng.weights = add_scaled(eng.weights, projected_delta, scale=1.0)
        raw_drift = eng.compute_anchor_drift()
        verified_drift = 0.0 if raw_drift <= tol else raw_drift
        eng.last_anchor_drift = verified_drift
        eng.edit_count += 1

        return {
            "engine_id": engine_id,
            "mode": mode,
            "anchor_drift": verified_drift,
            "raw_numerical_drift": raw_drift,
            "delta_frobenius_norm": float(frobenius_norm(projected_delta)),
            "cegar_refinements": cegar_refinements,
            "hopfield_attractor_count": len(eng.hopfield_patterns),
            "edit_count": eng.edit_count,
        }

    def coupling_step(
        self,
        *,
        impulse_engine_id: str | None = None,
        impulse_delta: float | None = None,
        compute_headroom_ratio: float | None = None,
        replay_records: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Execute multi-timescale Jacobian propagation + Lyapunov damping + Opportunistic SHA + Boundary Expansion."""
        n = len(self.engine_order)
        state_before = [self.engines[eid].state_energy for eid in self.engine_order]
        if impulse_engine_id and impulse_engine_id in self.engines:
            imp_idx = self.engine_order.index(impulse_engine_id)
            bump = float(
                impulse_delta if impulse_delta is not None else self._spec.c_float("tenth")
            )
            state_before[imp_idx] += bump

        # Aggregate active timescale gains
        ts_gains = {
            str(ts["timescale_id"]): float(ts["gain"])
            for ts in self._spec.section("feedback_timescales")
        }
        meta_cfg = self.hypergrid.interpolate_fluid_config(
            "homeostatic_meta_controller_engine",
            self.manifold.embed_task("homeostatic lyapunov coupling stabilization", stratum="meta_cybernetic_coupling"),
        )["parameters"]
        rho_cap = float(meta_cfg["lyapunov_contraction_rho"])

        # Spectral norm of Jacobian via compact SVD
        _, s_jac, _ = svd_compact(self.jacobian)
        raw_spectral_norm = s_jac[0] if s_jac else 0.0
        step_dt = float(
            self.hypergrid.interpolate_fluid_config(
                "inter_engine_coupling_and_jacobian_bus",
                self.manifold.embed_task("jacobian bus impulse propagation", stratum="meta_cybernetic_coupling"),
            )["parameters"]["jacobian_step_size"]
        )

        effective_gain = step_dt
        lyapunov_clamped = False
        if raw_spectral_norm * effective_gain >= rho_cap:
            effective_gain = (rho_cap * self._spec.c_float("half")) / max(
                raw_spectral_norm, self._spec.c_float("micro")
            )
            lyapunov_clamped = True

        # Discrete contractive update: s_{t+1} = (1 - leak) * s_t + effective_gain * J * s_t
        leak = max(self._spec.c_float("tenth"), effective_gain * raw_spectral_norm + self._spec.c_float("hundredth"))
        j_push = matvec(self.jacobian, state_before)
        state_after: Vector = []
        for idx, eid in enumerate(self.engine_order):
            eng = self.engines[eid]
            ts_scale = ts_gains.get(eng.primary_timescale, self._spec.c_float("tenth"))
            updated = (1.0 - leak) * state_before[idx] + effective_gain * ts_scale * j_push[idx]
            eng.state_energy = float(updated)
            state_after.append(float(updated))

        v_before = vec_dot(state_before, state_before)
        v_after = vec_dot(state_after, state_after)

        # Opportunistic SHA configuration exploration on target engine as compute headroom permits
        explore_target = impulse_engine_id or "fluid_hypergrid_interpolator_engine"
        sha_report = self.hypergrid.opportunistic_explore(
            explore_target,
            compute_headroom_ratio=compute_headroom_ratio,
            replay_records=replay_records,
        )

        self.coupling_steps += 1
        return {
            "coupling_step": self.coupling_steps,
            "jacobian_spectral_norm": float(raw_spectral_norm),
            "effective_coupling_gain": float(effective_gain),
            "lyapunov_clamped": lyapunov_clamped,
            "lyapunov_energy_before": float(v_before),
            "lyapunov_energy_after": float(v_after),
            "lyapunov_delta": float(v_after - v_before),
            "opportunistic_exploration": sha_report,
        }

    def morphism_transfer(
        self,
        source_dag: Sequence[Mapping[str, Any]],
        target_dag: Sequence[Mapping[str, Any]],
        *,
        source_engine_id: str = "ridge_polytope_compiler_engine",
        target_engine_id: str = "multilayer_film_molora_stack_engine",
    ) -> dict[str, Any]:
        """Compute spectral degree-Laplacian DAG isomorphism and Orthogonal Procrustes weight transplant."""

        def _dag_laplacian_spectrum(nodes: Sequence[Mapping[str, Any]]) -> Vector:
            count = max(1, len(nodes))
            id_map = {str(n.get("id", idx)): idx for idx, n in enumerate(nodes)}
            lap = zeros_matrix(count, count)
            for idx, node in enumerate(nodes):
                deps = node.get("depends_on", [])
                for dep in deps:
                    if str(dep) in id_map:
                        d_idx = id_map[str(dep)]
                        if d_idx != idx:
                            lap[idx][d_idx] = -1.0
                            lap[d_idx][idx] = -1.0
                            lap[idx][idx] += 1.0
                            lap[d_idx][d_idx] += 1.0
            _, s_lap, _ = svd_compact(lap)
            return l2_normalize(s_lap)

        spec_src = _dag_laplacian_spectrum(source_dag)
        spec_dst = _dag_laplacian_spectrum(target_dag)
        min_len = min(len(spec_src), len(spec_dst))
        spectral_sim = (
            vec_dot(spec_src[:min_len], spec_dst[:min_len]) if min_len > 0 else 1.0
        )

        src_eng = self.engines[source_engine_id]
        dst_eng = self.engines[target_engine_id]
        omega = orthogonal_procrustes(src_eng.weights, dst_eng.weights)
        transplanted = matmul(omega, src_eng.weights)

        # Blend transplanted subspace through dst_eng's anchor nullspace so zero anchor drift holds
        blend_gain = self._spec.c_float("quarter") * max(0.0, spectral_sim)
        delta = add_scaled(transplanted, dst_eng.weights, scale=-1.0)
        safe_delta = self._project_onto_anchor_nullspace(
            add_scaled(zeros_matrix(len(delta), len(delta[0])), delta, scale=blend_gain),
            dst_eng.anchor_inputs,
        )
        dst_eng.weights = add_scaled(dst_eng.weights, safe_delta, scale=1.0)
        drift = dst_eng.compute_anchor_drift()
        if drift <= self._spec.c_float("anchor_drift_zero_tol"):
            drift = 0.0

        orth_err = frobenius_norm(
            add_scaled(
                matmul(transpose(omega), omega),
                identity_matrix(len(omega)),
                scale=-1.0,
            )
        )
        return {
            "source_engine_id": source_engine_id,
            "target_engine_id": target_engine_id,
            "dag_spectral_isomorphism": float(spectral_sim),
            "procrustes_orthogonality_error": float(orth_err),
            "target_anchor_drift": float(drift),
            "transplanted_delta_norm": float(frobenius_norm(safe_delta)),
        }

    def calibrate_effort(
        self,
        goal_text: str,
        *,
        stakes: float | None = None,
        stratum: str = "compile",
    ) -> dict[str, Any]:
        """Compute Expected Value of Sample Information (EVSI) to select effort, slice dim, and escalation."""
        t_vec = self.manifold.embed_task(goal_text, stratum=stratum)
        fluid = self.hypergrid.interpolate_fluid_config(
            "evsi_effort_cascade_calibrator_engine", t_vec
        )
        c_vec: MatryoshkaVector = fluid["config2vec"]
        sur_stats = self.hypergrid.surrogate.score(t_vec, c_vec)
        epistemic_std = sur_stats["epistemic_std"]

        stakes_val = self._spec.c_float("half") if stakes is None else float(stakes)
        cost_w = float(fluid["parameters"]["information_cost_weight"])
        gross_voi = stakes_val * epistemic_std
        net_evsi = gross_voi - cost_w

        esc_cfg = self.hypergrid.interpolate_fluid_config(
            "human_attention_and_escalation_engine", t_vec
        )["parameters"]
        esc_thresh = float(esc_cfg["escalation_evsi_threshold"])

        m_dims = self._spec.section("matryoshka_manifold")["prefix_dims"]
        quarter = self._spec.c_float("quarter")
        half = self._spec.c_float("half")

        if net_evsi <= 0.0:
            effort = "low"
            chosen_dim = int(m_dims[0])
            short_circuit = True
        elif net_evsi <= quarter:
            effort = "medium"
            chosen_dim = int(m_dims[self._spec.c_int("two")])
            short_circuit = False
        else:
            effort = "high"
            chosen_dim = int(m_dims[-1])
            short_circuit = False

        early_margin = max(
            self._spec.c_float("tenth"),
            half - quarter * math.tanh(max(0.0, net_evsi)),
        )
        return {
            "reasoning_effort": effort,
            "recommended_matryoshka_dim": chosen_dim,
            "short_circuit_reflex": short_circuit,
            "early_exit_margin": float(early_margin),
            "epistemic_uncertainty": float(epistemic_std),
            "gross_voi": float(gross_voi),
            "net_evsi": float(net_evsi),
            "escalate_to_human": bool(gross_voi >= esc_thresh),
        }

    def manifold_status(self) -> dict[str, Any]:
        """Full 20-engine / 72-sub-engine / 100-dim HyperGrid telemetry snapshot."""
        m_cfg = self._spec.section("matryoshka_manifold")
        engine_summaries: list[dict[str, Any]] = []
        for eid in self.engine_order:
            eng = self.engines[eid]
            grid = self.hypergrid.grids[eid]
            engine_summaries.append(
                {
                    "index": eng.index,
                    "engine_id": eid,
                    "stratum": eng.stratum,
                    "primary_timescale": eng.primary_timescale,
                    "sub_engine_count": len(eng.sub_engines),
                    "grid_generation": grid.generation,
                    "parameter_count": len(grid.parameters),
                    "verified_cell_count": len(self.hypergrid.ledger.verified_cells(eid)),
                    "edit_count": eng.edit_count,
                    "last_anchor_drift": eng.last_anchor_drift,
                    "state_energy": eng.state_energy,
                }
            )

        return {
            "version": self._spec.section("version"),
            "engine_count": len(self.engine_order),
            "sub_engine_count": self.sub_engine_count,
            "hypergrid_total_dimensions": self.hypergrid.total_dimensions,
            "matryoshka_spaces": list(m_cfg["spaces"]),
            "matryoshka_prefix_dims": list(m_cfg["prefix_dims"]),
            "timescale_count": len(self._spec.section("feedback_timescales")),
            "coupling_edge_count": len(self._spec.section("sparse_coupling_edges")),
            "coupling_steps": self.coupling_steps,
            "grid_ledger_counts": self.hypergrid.ledger.state_counts(),
            "woodbury_surrogate_updates": self.hypergrid.surrogate.updates,
            "cost_ledger": self.cost_ledger.summary(),
            "engines": engine_summaries,
        }

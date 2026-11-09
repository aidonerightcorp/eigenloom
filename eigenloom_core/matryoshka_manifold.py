"""Unified 4-Space Matryoshka Embedding Manifold (Task2Vec, Config2Vec, Weight2Vec, Trace2Vec)."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.linalg_backend import (
    Matrix,
    Vector,
    frobenius_norm,
    l2_normalize,
    svd_compact,
    vec_dot,
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_STRATUM_REGIME_ANCHORS: tuple[str, ...] = (
    "reflex",
    "ingestion_and_memory",
    "compile",
    "compilation_and_lowering",
    "verify",
    "routing_and_governance",
    "architect",
    "adaptation_and_economics",
    "meta_cybernetic_coupling",
)


@dataclass(frozen=True)
class MatryoshkaVector:
    """Nested 128d representation where every prefix (16d, 32d, 64d, 128d) is unit-normalized."""

    space: str
    values: tuple[float, ...]
    prefix_dims: tuple[int, ...]

    def slice_prefix(self, dim: int) -> Vector:
        capped = min(max(1, int(dim)), len(self.values))
        return l2_normalize(list(self.values[:capped]))

    def cosine(self, other: "MatryoshkaVector", dim: int | None = None) -> float:
        use_dim = len(self.values) if dim is None else int(dim)
        va = self.slice_prefix(use_dim)
        vb = other.slice_prefix(use_dim)
        return float(vec_dot(va, vb))

    def to_dict(self) -> dict[str, Any]:
        return {
            "space": self.space,
            "full_dim": len(self.values),
            "prefix_dims": list(self.prefix_dims),
            "values": [float(x) for x in self.values],
        }

    @classmethod
    def from_list(
        cls,
        values: Sequence[float],
        space: str = "trace2vec",
        spec: ManifoldConfigSpec | None = None,
    ) -> "MatryoshkaVector":
        cfg = spec or get_manifold_spec()
        m_cfg = cfg.section("matryoshka_manifold")
        full_dim = int(m_cfg["full_dim"])
        p_dims = tuple(int(d) for d in m_cfg["prefix_dims"])
        padded = [float(x) for x in values[:full_dim]]
        if len(padded) < full_dim:
            padded.extend([0.0 for _ in range(full_dim - len(padded))])
        normed = l2_normalize(padded)
        return cls(space=space, values=tuple(normed), prefix_dims=p_dims)


class MatryoshkaManifold:
    """Projects tasks, configurations, weights, and traces into one shared MRL geometry."""

    def __init__(self, spec: ManifoldConfigSpec | None = None) -> None:
        self._spec = spec or get_manifold_spec()
        m_cfg = self._spec.section("matryoshka_manifold")
        self._full_dim = int(m_cfg["full_dim"])
        self._coarse_dim = int(m_cfg["coarse_dim"])
        self._medium_dim = int(m_cfg["medium_dim"])
        self._fine_dim = int(m_cfg["fine_dim"])
        self._prefix_dims = tuple(int(d) for d in m_cfg["prefix_dims"])
        self._bands: dict[str, list[int]] = dict(m_cfg["band_ranges"])
        self._keep_mult = int(m_cfg["cascade_keep_multiplier"])

    @property
    def prefix_dims(self) -> tuple[int, ...]:
        return self._prefix_dims

    @property
    def full_dim(self) -> int:
        return self._full_dim

    def _signed_hash_slot(self, tag: str, start: int, end: int) -> tuple[int, float]:
        span = max(1, end - start)
        digest = hashlib.blake2s(
            tag.encode("utf-8"),
            digest_size=self._spec.c_int("eight"),
        ).digest()
        idx_raw = int.from_bytes(digest[: self._spec.c_int("four")], "little")
        sign_raw = digest[self._spec.c_int("four")]
        two = self._spec.c_int("two")
        sign = 1.0 if (sign_raw % two) == 0 else -1.0
        return start + (idx_raw % span), sign

    def _accumulate_tokens(
        self,
        buf: Vector,
        tokens: Sequence[str],
        weight: float = 1.0,
    ) -> None:
        r_lo, r_hi = self._bands["regime_band"]
        s_lo, s_hi = self._bands["structural_band"]
        l_lo, l_hi = self._bands["spectral_lexical_band"]
        h_lo, h_hi = self._bands["high_res_band"]
        ngram_w = self._spec.c_int("three")

        for idx, tok in enumerate(tokens):
            slot_r, sgn_r = self._signed_hash_slot(f"reg:{tok}", r_lo, r_hi)
            buf[slot_r] += weight * sgn_r * self._spec.c_float("half")

            slot_l, sgn_l = self._signed_hash_slot(f"lex:{tok}", l_lo, l_hi)
            buf[slot_l] += weight * sgn_l

            if idx + 1 < len(tokens):
                bigram = f"{tok}_{tokens[idx + 1]}"
                slot_s, sgn_s = self._signed_hash_slot(f"bi:{bigram}", s_lo, s_hi)
                buf[slot_s] += weight * sgn_s

            if len(tok) >= ngram_w:
                for pos in range(len(tok) - ngram_w + 1):
                    shingle = tok[pos : pos + ngram_w]
                    slot_h, sgn_h = self._signed_hash_slot(
                        f"sh:{shingle}", h_lo, h_hi
                    )
                    buf[slot_h] += weight * sgn_h * self._spec.c_float("half")
            else:
                slot_h, sgn_h = self._signed_hash_slot(f"tok:{tok}", h_lo, h_hi)
                buf[slot_h] += weight * sgn_h

    def _finalize(self, space: str, buf: Vector) -> MatryoshkaVector:
        # Balance energy across nested bands so every prefix slice carries rich geometry
        for _, bounds in self._bands.items():
            lo, hi = int(bounds[0]), int(bounds[1])
            band_slice = l2_normalize(buf[lo:hi])
            for offset, val in enumerate(band_slice):
                buf[lo + offset] = val
        normed = l2_normalize(buf)
        if all(x == 0.0 for x in normed):
            normed[0] = 1.0
        return MatryoshkaVector(
            space=space,
            values=tuple(normed),
            prefix_dims=self._prefix_dims,
        )

    def embed_task(
        self,
        goal_text: str,
        *,
        stratum: str = "compile",
        verification_role: str = "maker",
        fanout: int = 1,
    ) -> MatryoshkaVector:
        """Task2Vec: embed task goal, stratum affinity, role, and structural fanout."""
        buf = [0.0 for _ in range(self._full_dim)]
        r_lo, r_hi = self._bands["regime_band"]
        s_lo, s_hi = self._bands["structural_band"]

        slot_st, sgn_st = self._signed_hash_slot(f"stratum:{stratum}", r_lo, r_hi)
        buf[slot_st] += sgn_st
        slot_rl, sgn_rl = self._signed_hash_slot(
            f"role:{verification_role}", r_lo, r_hi
        )
        buf[slot_rl] += sgn_rl * self._spec.c_float("half")

        tokens = _TOKEN_RE.findall(f"{stratum} {verification_role} {goal_text}".lower())
        complexity = math.tanh(
            float(len(tokens) + max(0, int(fanout)))
            / float(self._spec.c_int("thirty_two"))
        )
        buf[r_lo] += complexity
        buf[s_lo] += math.tanh(float(max(1, int(fanout))) / float(self._spec.c_int("eight")))

        self._accumulate_tokens(buf, tokens, weight=1.0)
        return self._finalize("task2vec", buf)

    def embed_config(
        self,
        config_values: dict[str, Any],
        *,
        engine_id: str = "",
        stratum: str = "compile",
        bounds_lookup: dict[str, tuple[float, float]] | None = None,
    ) -> MatryoshkaVector:
        """Config2Vec: embed up to 100+ HyperGrid parameters via normalized harmonic projections."""
        buf = [0.0 for _ in range(self._full_dim)]
        r_lo, r_hi = self._bands["regime_band"]
        s_lo, s_hi = self._bands["structural_band"]
        l_lo, l_hi = self._bands["spectral_lexical_band"]
        h_lo, h_hi = self._bands["high_res_band"]

        if stratum:
            slot_st, sgn_st = self._signed_hash_slot(f"stratum:{stratum}", r_lo, r_hi)
            buf[slot_st] += sgn_st
        if engine_id:
            tokens = _TOKEN_RE.findall(engine_id.lower())
            self._accumulate_tokens(buf, tokens, weight=self._spec.c_float("half"))

        pi_val = math.pi
        two = float(self._spec.c_int("two"))
        for key in sorted(config_values.keys()):
            val = config_values[key]
            if isinstance(val, bool):
                scalar = 1.0 if val else -1.0
                slot_b, sgn_b = self._signed_hash_slot(f"cfg_b:{key}", s_lo, s_hi)
                buf[slot_b] += sgn_b * scalar
            elif isinstance(val, (int, float)):
                fval = float(val)
                if bounds_lookup and key in bounds_lookup:
                    lo_b, hi_b = bounds_lookup[key]
                    span = max(hi_b - lo_b, self._spec.c_float("micro"))
                    unit = max(0.0, min(1.0, (fval - lo_b) / span))
                else:
                    unit = self._spec.c_float("half") * (1.0 + math.tanh(fval))
                centered = two * unit - 1.0
                slot_r, sgn_r = self._signed_hash_slot(f"cfg_r:{key}", r_lo, r_hi)
                slot_s, sgn_s = self._signed_hash_slot(f"cfg_s:{key}", s_lo, s_hi)
                slot_l, sgn_l = self._signed_hash_slot(f"cfg_c:{key}", l_lo, l_hi)
                slot_h, sgn_h = self._signed_hash_slot(f"cfg_h:{key}", h_lo, h_hi)
                buf[slot_r] += sgn_r * centered
                buf[slot_s] += sgn_s * math.sin(pi_val * unit)
                buf[slot_l] += sgn_l * math.cos(pi_val * unit)
                buf[slot_h] += sgn_h * math.sin(two * pi_val * unit)
            else:
                text_val = str(val).lower()
                tokens = _TOKEN_RE.findall(f"{key}_{text_val}")
                self._accumulate_tokens(buf, tokens, weight=1.0)
        return self._finalize("config2vec", buf)

    def embed_weights(
        self,
        matrices: Sequence[Matrix],
        *,
        stratum: str = "compile",
        tag: str = "",
    ) -> MatryoshkaVector:
        """Weight2Vec: embed single or multi-layer operator stacks via spectral invariants."""
        buf = [0.0 for _ in range(self._full_dim)]
        r_lo, r_hi = self._bands["regime_band"]
        s_lo, s_hi = self._bands["structural_band"]
        l_lo, l_hi = self._bands["spectral_lexical_band"]
        h_lo, h_hi = self._bands["high_res_band"]

        slot_st, sgn_st = self._signed_hash_slot(f"stratum:{stratum}", r_lo, r_hi)
        buf[slot_st] += sgn_st
        if tag:
            self._accumulate_tokens(
                buf, _TOKEN_RE.findall(tag.lower()), weight=self._spec.c_float("quarter")
            )

        eps = self._spec.c_float("nano")
        layer_count = len(matrices)
        buf[r_lo + 1] += math.tanh(
            float(layer_count) / float(self._spec.c_int("six"))
        )

        for layer_idx, mat in enumerate(matrices):
            if not mat or not mat[0]:
                continue
            frob = frobenius_norm(mat)
            _, s_vals, _ = svd_compact(mat)
            top_s = s_vals[0] if s_vals else 0.0
            stable_rank = (frob * frob) / max(top_s * top_s, eps) if top_s > eps else 0.0
            s_sum = sum(s_vals)
            probs = [sv / max(s_sum, eps) for sv in s_vals if sv > eps]
            entropy = -sum(p * math.log(max(p, eps)) for p in probs)
            eff_rank = math.exp(entropy) if probs else 0.0

            s_span = max(1, s_hi - s_lo)
            buf[s_lo + ((layer_idx * self._spec.c_int("two")) % s_span)] += math.tanh(
                stable_rank / float(self._spec.c_int("eight"))
            )
            buf[s_lo + ((layer_idx * self._spec.c_int("two") + 1) % s_span)] += math.tanh(
                eff_rank / float(self._spec.c_int("eight"))
            )

            l_span = max(1, l_hi - l_lo)
            for s_idx, sv in enumerate(s_vals):
                norm_sv = sv / max(top_s, eps) if top_s > eps else 0.0
                buf[l_lo + ((layer_idx * self._spec.c_int("four") + s_idx) % l_span)] += norm_sv

            h_span = max(1, h_hi - h_lo)
            for r_idx, row in enumerate(mat):
                row_energy = math.sqrt(sum(float(v) * float(v) for v in row))
                buf[h_lo + ((layer_idx * self._spec.c_int("eight") + r_idx) % h_span)] += math.tanh(
                    row_energy
                )

        return self._finalize("weight2vec", buf)

    def embed_trace(
        self,
        content: str,
        *,
        role: str = "user",
        stratum: str = "ingestion_and_memory",
        saliency: float | None = None,
    ) -> MatryoshkaVector:
        """Trace2Vec: embed conversation turns and episodic Hopfield attractors."""
        buf = [0.0 for _ in range(self._full_dim)]
        r_lo, r_hi = self._bands["regime_band"]
        s_lo, _ = self._bands["structural_band"]

        slot_st, sgn_st = self._signed_hash_slot(f"stratum:{stratum}", r_lo, r_hi)
        buf[slot_st] += sgn_st * self._spec.c_float("half")
        slot_ro, sgn_ro = self._signed_hash_slot(f"msg_role:{role}", r_lo, r_hi)
        buf[slot_ro] += sgn_ro * self._spec.c_float("half")
        sal_val = self._spec.c_float("half") if saliency is None else float(saliency)
        buf[s_lo] += max(0.0, min(1.0, sal_val))

        tokens = _TOKEN_RE.findall(content.lower())
        self._accumulate_tokens(buf, tokens, weight=1.0)
        return self._finalize("trace2vec", buf)

    def progressive_cascade_search(
        self,
        query: MatryoshkaVector,
        candidates: Sequence[tuple[str, MatryoshkaVector, Any]],
        top_k: int,
    ) -> dict[str, Any]:
        """Coarse-to-fine 16d -> 32d -> 64d -> 128d Matryoshka funnel search."""
        if not candidates or top_k <= 0:
            return {
                "matches": [],
                "mac_operations_used": 0,
                "exhaustive_mac_baseline": 0,
                "compute_savings_ratio": 0.0,
            }

        k_cap = max(1, int(top_k))
        exhaustive_macs = len(candidates) * self._full_dim
        macs_used = 0
        pool = list(candidates)
        total_stages = len(self._prefix_dims)

        scored: list[tuple[float, str, MatryoshkaVector, Any]] = []
        for stage_idx, dim in enumerate(self._prefix_dims):
            q_slice = query.slice_prefix(dim)
            scored = []
            for item_id, vec, payload in pool:
                macs_used += dim
                sim = vec_dot(q_slice, vec.slice_prefix(dim))
                scored.append((sim, item_id, vec, payload))
            scored.sort(key=lambda item: item[0], reverse=True)
            rem_stages = total_stages - stage_idx - 1
            if rem_stages > 0:
                keep_count = max(k_cap, k_cap * (self._keep_mult ** rem_stages))
                pool = [
                    (item_id, vec, payload)
                    for _, item_id, vec, payload in scored[:keep_count]
                ]

        winners = [
            {
                "id": item_id,
                "similarity": float(sim),
                "space": vec.space,
                "payload": payload,
            }
            for sim, item_id, vec, payload in scored[:k_cap]
        ]
        saved_ratio = (
            max(0.0, 1.0 - (float(macs_used) / float(exhaustive_macs)))
            if exhaustive_macs > 0
            else 0.0
        )
        return {
            "matches": winners,
            "mac_operations_used": macs_used,
            "exhaustive_mac_baseline": exhaustive_macs,
            "compute_savings_ratio": saved_ratio,
        }

"""Surface 3 — Native Hermes DelegationProvider with Contextual LinUCB Woodbury routing and maker != checker."""

from __future__ import annotations

from abc import ABC
from typing import Any, Mapping

from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.storage import EigenLoomStore
from eigenloom_core.weight_bus import CyberneticWeightBus

try:
    from agent.delegation_provider import (
        DelegationProvider as _HermesDelegationProvider,
        DelegationRouteError,
    )
except ImportError:

    class DelegationRouteError(RuntimeError):  # type: ignore[no-redef]
        """Raised when a delegation route violates verification separation or cannot be satisfied."""

    class _HermesDelegationProvider(ABC):  # type: ignore[no-redef]
        pass


class EigenLoomDelegationProvider(_HermesDelegationProvider):
    """Routes subtasks across candidate model tiers via Task2Vec x Config2Vec Woodbury UCB."""

    def __init__(
        self,
        *,
        bus: CyberneticWeightBus | None = None,
        store: EigenLoomStore | None = None,
        spec: ManifoldConfigSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._bus = bus or CyberneticWeightBus(self._spec)
        self._store = store or EigenLoomStore(
            spec=self._spec, manifold=self._bus.manifold
        )

    @property
    def name(self) -> str:
        return "eigenloom"

    def is_available(self, config: Mapping[str, Any]) -> bool:
        return True

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Declare non-reserved per-task routing and verification separation parameters."""
        return {
            "engine_stratum": {
                "type": "string",
                "enum": ["reflex", "compile", "verify", "architect"],
                "description": "Target functional engine stratum for this delegated subtask.",
            },
            "verification_role": {
                "type": "string",
                "enum": ["maker", "checker"],
                "description": "Whether this subtask produces an artifact (maker) or audits it (checker).",
            },
            "maker_id": {
                "type": "string",
                "description": "Identity/route label of the maker when verification_role is 'checker'.",
            },
        }

    def route_task(
        self, task: Mapping[str, Any], config: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Score candidate tiers via Contextual LinUCB Woodbury UCB and enforce maker != checker."""
        goal = str(task.get("goal", task.get("prompt", "delegated subtask")) or "delegated subtask")
        stratum = str(task.get("engine_stratum", "compile") or "compile")
        role = str(task.get("verification_role", "maker") or "maker")
        maker_id = str(task.get("maker_id", "") or "").strip()

        raw_tiers = config.get("tiers") if isinstance(config, Mapping) else None
        candidates: list[dict[str, Any]] = (
            [dict(t) for t in raw_tiers]
            if isinstance(raw_tiers, list) and raw_tiers
            else [dict(t) for t in self._spec.section("delegation_tiers")]
        )

        # Enforce strict maker != checker independence
        if role == "checker" and maker_id:
            requested_checker = str(task.get("checker_id", "") or "").strip()
            if requested_checker and requested_checker == maker_id:
                raise DelegationRouteError(
                    f"Verification separation violation: checker_id '{requested_checker}' matches maker_id '{maker_id}'."
                )
            filtered: list[dict[str, Any]] = []
            for cand in candidates:
                identifiers = {
                    str(cand.get("tier_id", "")),
                    str(cand.get("model", "")),
                    str(cand.get("profile", "")),
                    str(cand.get("label", "")),
                }
                if maker_id not in identifiers:
                    filtered.append(cand)
            if not filtered:
                raise DelegationRouteError(
                    f"No independent checker tier available distinct from maker_id '{maker_id}'."
                )
            candidates = filtered

        t_vec = self._bus.manifold.embed_task(
            goal, stratum=stratum, verification_role=role
        )
        router_cfg = self._bus.hypergrid.interpolate_fluid_config(
            "contextual_linucb_router_engine", t_vec
        )["parameters"]
        ucb_alpha = float(router_cfg["ucb_exploration_alpha"])
        objective = str(router_cfg["routing_objective"])

        best_tier = candidates[0]
        best_score = -float(self._spec.c_int("one_thousand"))

        for cand in candidates:
            tid = str(cand.get("tier_id", cand.get("model", "default_tier")))
            c_vec = self._bus.manifold.embed_config(
                cand, engine_id="contextual_linucb_router_engine", stratum=stratum
            )
            bilinear_ucb = self._bus.hypergrid.surrogate.score(t_vec, c_vec)["ucb_score"]
            tier_ucb = self._store.score_delegation_tier(tid, t_vec, ucb_alpha)
            affinity_bonus = (
                self._spec.c_float("half")
                if str(cand.get("stratum_affinity", "")) == stratum
                else 0.0
            )
            rel_cost = float(cand.get("relative_cost", self._spec.c_float("half")))
            cost_penalty = (
                self._spec.c_float("quarter") * rel_cost
                if objective == "reward_per_cost"
                else 0.0
            )
            total = bilinear_ucb + tier_ucb + affinity_bonus - cost_penalty
            if total > best_score:
                best_score = total
                best_tier = cand

        route: dict[str, Any] = {
            "reasoning_effort": str(best_tier.get("reasoning_effort", "medium")),
            "label": str(best_tier.get("tier_id", "eigenloom_route")),
        }
        if best_tier.get("model"):
            route["model"] = str(best_tier["model"])
        if best_tier.get("provider") and str(best_tier["provider"]) != "default":
            route["provider"] = str(best_tier["provider"])
        if best_tier.get("service_tier"):
            route["service_tier"] = str(best_tier["service_tier"])
        if best_tier.get("profile"):
            route["profile"] = str(best_tier["profile"])
        return route

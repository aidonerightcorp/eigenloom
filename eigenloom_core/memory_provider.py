"""Surface 2 — Native Hermes MemoryProvider over SQLite WAL Hopfield attractors and safetensors consolidation."""

from __future__ import annotations

import json
import threading
from abc import ABC
from typing import Any

from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.storage import EigenLoomStore
from eigenloom_core.weight_bus import CyberneticWeightBus

try:
    from agent.memory_provider import MemoryProvider as _HermesMemoryProvider
except ImportError:

    class _HermesMemoryProvider(ABC):  # type: ignore[no-redef]
        pass


ATTRACTOR_RECALL_SCHEMA: dict[str, Any] = {
    "name": "eigenloom_attractor_recall",
    "description": (
        "Recall episodic memories from the local SQLite WAL Hopfield attractor bank "
        "via progressive 16d->32d->64d->128d Matryoshka cascade search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concept, entity, or goal to recall from Hopfield episodic memory.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of attractors to return.",
            },
        },
        "required": ["query"],
    },
}


class EigenLoomMemoryProvider(_HermesMemoryProvider):
    """Zero-network local Hopfield + LoRA-XS safetensors micro-cartridge memory provider."""

    _SILENT_AGENT_CONTEXTS: frozenset[str] = frozenset({"cron", "subagent", "flush"})

    def __init__(
        self,
        *,
        bus: CyberneticWeightBus | None = None,
        spec: ManifoldConfigSpec | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._bus = bus or CyberneticWeightBus(self._spec)
        self._store: EigenLoomStore | None = None
        self._session_id: str = "default"
        self._agent_context: str = "primary"
        self._hermes_home: str = ""
        self._prefetched_context: str = ""
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "eigenloom"

    def is_available(self) -> bool:
        """Always available locally via profile-scoped SQLite WAL with zero network calls."""
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        with self._lock:
            self._session_id = str(session_id or "default")
            self._agent_context = str(kwargs.get("agent_context", "primary") or "primary")
            hermes_home = str(kwargs.get("hermes_home", "") or "")
            self._hermes_home = hermes_home
            self._store = EigenLoomStore(
                hermes_home=hermes_home or None,
                spec=self._spec,
                manifold=self._bus.manifold,
            )

    def _ensure_store(self) -> EigenLoomStore:
        with self._lock:
            if self._store is None:
                self._store = EigenLoomStore(
                    hermes_home=self._hermes_home or None,
                    spec=self._spec,
                    manifold=self._bus.manifold,
                )
            return self._store

    def _writes_suppressed(self) -> bool:
        return self._agent_context in self._SILENT_AGENT_CONTEXTS

    def system_prompt_block(self) -> str:
        with self._lock:
            if not self._prefetched_context:
                return ""
            return (
                "### EigenLoom Hopfield Attractor Recall\n"
                + self._prefetched_context
            )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        store = self._ensure_store()
        fluid = self._bus.hypergrid.interpolate_fluid_config(
            "hopfield_attractor_memory_engine",
            self._bus.manifold.embed_trace(query),
        )["parameters"]
        beta = float(fluid.get("inverse_temperature_beta", self._spec.c_int("eight")))
        res = store.recall_attractors(
            query, top_k=self._spec.c_int("three"), beta=beta
        )
        matches = res["matches"]
        if not matches:
            return ""
        lines = [
            f"- [{m['id']}] (hopfield={m.get('hopfield_attention_weight', 0.0):.2f}): {m['payload']['summary']}"
            for m in matches
        ]
        rendered = "\n".join(lines)
        with self._lock:
            self._prefetched_context = rendered
        return rendered

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self.prefetch(query, session_id=session_id)

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        if message and message.strip():
            self.prefetch(message, session_id=self._session_id)

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Persist turn episode unless running inside cron/subagent/flush context."""
        if self._writes_suppressed():
            return
        combined = f"Q: {user_content.strip()} | A: {assistant_content.strip()}".strip()
        if not combined or combined == "Q: | A:":
            return
        store = self._ensure_store()
        max_len = self._spec.c_int("five_twelve")
        clipped = combined[:max_len]
        sal = min(
            1.0,
            float(len(set(clipped.lower().split())))
            / float(self._spec.c_int("sixty_four")),
        )
        store.store_attractor(
            session_id or self._session_id,
            clipped,
            saliency=sal,
        )

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Consolidate pre-compaction messages into a Hopfield attractor + safetensors micro-cartridge."""
        if not messages:
            return ""
        snippets: list[str] = []
        rules: list[dict[str, Any]] = []
        cap = self._spec.c_int("eight")
        for idx, msg in enumerate(messages[:cap]):
            raw = msg.get("content", "")
            txt = raw if isinstance(raw, str) else json.dumps(raw)
            txt = txt.strip()
            if txt:
                snippets.append(txt[: self._spec.c_int("one_twenty_eight")])
                rules.append(
                    {
                        "condition": f"turn_{idx}_{msg.get('role', 'user')}",
                        "action": txt[: self._spec.c_int("sixty_four")],
                    }
                )
        if not snippets:
            return ""
        summary = " ; ".join(snippets[: self._spec.c_int("four")])

        if not self._writes_suppressed():
            cart = self._bus.compiler.compile_stack(
                f"mem_{self._session_id}",
                rules,
                num_layers=self._spec.c_int("two"),
            )
            st_b64 = self._bus.compiler.lower_to_target(cart, "safetensors")
            self._ensure_store().store_attractor(
                self._session_id,
                summary,
                saliency=1.0,
                cartridge_b64=st_b64,
            )
        return summary

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs: Any
    ) -> None:
        if self._writes_suppressed():
            return
        if task and result:
            clip = self._spec.c_int("two_fifty_six")
            self._ensure_store().store_attractor(
                self._session_id,
                f"Delegated [{task[:clip]}] -> {result[:clip]}",
                saliency=self._spec.c_float("half") + self._spec.c_float("quarter"),
            )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(ATTRACTOR_RECALL_SCHEMA)]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name != "eigenloom_attractor_recall":
            return json.dumps({"error": f"Unknown MemoryProvider tool: {tool_name}"})
        query = str(args.get("query", ""))
        top_k = args.get("top_k")
        res = self._ensure_store().recall_attractors(
            query, top_k=int(top_k) if top_k is not None else None
        )
        return json.dumps(res)

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        store = EigenLoomStore(
            hermes_home=hermes_home or self._hermes_home or None,
            spec=self._spec,
            manifold=self._bus.manifold,
        )
        store.save_snapshot("memory_provider_config", values)

    def shutdown(self) -> None:
        with self._lock:
            self._prefetched_context = ""

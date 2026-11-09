"""Surface 1 — Native Hermes ContextEngine over message-UID quotient DAG and Wanda saliency."""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC
from typing import Any, Optional

from eigenloom_core.config_loader import ManifoldConfigSpec, get_manifold_spec
from eigenloom_core.storage import EigenLoomStore
from eigenloom_core.weight_bus import CyberneticWeightBus

try:
    from agent.context_engine import ContextEngine as _HermesContextEngine
except ImportError:

    class _HermesContextEngine(ABC):  # type: ignore[no-redef]
        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0


CONTEXT_GREP_SCHEMA: dict[str, Any] = {
    "name": "eigenloom_context_grep",
    "description": (
        "Search compacted and active session context over the message-UID quotient DAG "
        "using hybrid lexical + Trace2Vec Matryoshka semantic saliency."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or conceptual query to retrieve from quotient context.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum matching turns to return.",
            },
        },
        "required": ["query"],
    },
}


class EigenLoomContextEngine(_HermesContextEngine):
    """Thread-safe quotient DAG context engine preserving prompt-cache object identity."""

    def __init__(
        self,
        *,
        bus: CyberneticWeightBus | None = None,
        store: EigenLoomStore | None = None,
        spec: ManifoldConfigSpec | None = None,
        context_length: int | None = None,
        threshold_percent: float | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self._bus = bus or CyberneticWeightBus(self._spec)
        self._store = store or EigenLoomStore(
            spec=self._spec, manifold=self._bus.manifold
        )
        self._lock = threading.RLock()
        self._session_id: str = "default"

        self.context_length: int = int(
            context_length
            if context_length is not None
            else self._spec.c_int("default_context_length")
        )
        self.threshold_percent: float = float(
            threshold_percent
            if threshold_percent is not None
            else self._spec.c_float("default_threshold_percent")
        )
        self._config_threshold_percent: float = self.threshold_percent
        self.threshold_tokens: int = int(self.context_length * self.threshold_percent)
        self.protect_first_n: int = self._spec.c_int("two")
        self.protect_last_n: int = self._spec.c_int("four")
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.compression_count: int = 0
        self.emit_automatic_compaction_status: bool = True

    @property
    def name(self) -> str:
        return "eigenloom"

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        chars_per_tok = max(1, self._spec.c_int("chars_per_token_estimate"))
        total_chars = 0
        for msg in messages:
            raw = msg.get("content", "")
            text = raw if isinstance(raw, str) else json.dumps(raw)
            total_chars += len(text)
        return total_chars // chars_per_tok

    def on_session_start(self, session_id: str = "", **kwargs: Any) -> None:
        with self._lock:
            if session_id:
                self._session_id = str(session_id)

    def on_session_end(self, session_id: str = "", messages: list[dict[str, Any]] | None = None) -> None:
        with self._lock:
            if messages:
                self._store.index_quotient_messages(
                    session_id or self._session_id, messages
                )

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        with self._lock:
            self.context_length = max(1, int(context_length))
            self.threshold_percent = self._config_threshold_percent
            self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def update_from_response(self, usage: dict[str, Any]) -> None:
        with self._lock:
            prompt_t = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            comp_t = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            tot_t = int(usage.get("total_tokens", prompt_t + comp_t) or (prompt_t + comp_t))
            self.last_prompt_tokens = prompt_t
            self.last_completion_tokens = comp_t
            self.last_total_tokens = tot_t
            self._bus.cost_ledger.prompt_tokens_observed += max(0, prompt_t)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        with self._lock:
            observed = (
                int(prompt_tokens)
                if prompt_tokens is not None
                else self.last_prompt_tokens
            )
            if self.threshold_tokens <= 0:
                return False
            return observed >= self.threshold_tokens

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        """Return exact request_messages reference when within budget to preserve prompt caching."""
        with self._lock:
            if budget_tokens <= 0 or self._estimate_tokens(request_messages) <= budget_tokens:
                self._bus.cost_ledger.record_context_selection(cache_preserved=True)
                return request_messages

            self._bus.cost_ledger.record_context_selection(cache_preserved=False)
            focus = ""
            if incoming_message and isinstance(incoming_message.get("content"), str):
                focus = str(incoming_message["content"])
            return self.compress(
                request_messages,
                current_tokens=self._estimate_tokens(request_messages),
                focus_topic=focus or None,
                force=True,
            )

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Deduplicate into quotient DAG and prune low-saliency middle turns safely across pooled threads."""
        with self._lock:
            if not messages:
                return messages

            self._store.index_quotient_messages(self._session_id, messages)
            before_tokens = (
                int(current_tokens)
                if current_tokens is not None
                else self._estimate_tokens(messages)
            )
            if not force and self.threshold_tokens > 0 and before_tokens < self.threshold_tokens:
                return messages

            focus_vec = self._bus.manifold.embed_trace(
                focus_topic or "session context continuity"
            )
            fluid_q = self._bus.hypergrid.interpolate_fluid_config(
                "quotient_context_dag_engine", focus_vec
            )["parameters"]
            fluid_w = self._bus.hypergrid.interpolate_fluid_config(
                "wanda_saliency_pruning_engine", focus_vec
            )["parameters"]

            head_n = min(len(messages), max(1, self.protect_first_n))
            tail_n = min(
                max(0, len(messages) - head_n),
                max(1, int(fluid_q.get("protect_tail_turns", self.protect_last_n))),
            )
            if head_n + tail_n >= len(messages):
                return list(messages)

            head = list(messages[:head_n])
            middle = list(messages[head_n : len(messages) - tail_n])
            tail = list(messages[len(messages) - tail_n :])

            seen_uids: set[str] = set()
            scored_middle: list[tuple[float, int, dict[str, Any]]] = []
            evicted_count = 0

            for idx, msg in enumerate(middle):
                role = str(msg.get("role", "user"))
                raw = msg.get("content", "")
                text = raw if isinstance(raw, str) else json.dumps(raw)
                # Always preserve tool-call / tool-result structural pairs intact
                if role == "tool" or msg.get("tool_calls"):
                    scored_middle.append((1.0 + 1.0, idx, msg))
                    continue
                uid = hashlib.blake2s(
                    f"{role}:{text}".encode("utf-8"),
                    digest_size=self._spec.c_int("eight"),
                ).hexdigest()
                if uid in seen_uids:
                    evicted_count += 1
                    continue
                seen_uids.add(uid)
                m_vec = self._bus.manifold.embed_trace(text, role=role)
                sem = (1.0 + focus_vec.cosine(m_vec)) * self._spec.c_float("half")
                wanda_len = min(
                    1.0,
                    float(len(set(text.lower().split())))
                    / float(self._spec.c_int("thirty_two")),
                )
                score = self._spec.c_float("half") * sem + self._spec.c_float("half") * wanda_len
                scored_middle.append((score, idx, msg))

            floor = float(fluid_w.get("saliency_quantile_floor", self._spec.c_float("quarter")))
            target_ratio = float(fluid_q.get("compression_ratio_target", self._spec.c_float("half")))
            keep_cap = max(0, int(len(scored_middle) * target_ratio))
            ranked = sorted(scored_middle, key=lambda item: item[0], reverse=True)

            kept_indices: set[int] = set()
            for rank_pos, (sc, orig_idx, _) in enumerate(ranked):
                if sc > 1.0 or (rank_pos < keep_cap and sc >= floor):
                    kept_indices.add(orig_idx)

            retained_middle = [
                msg for _, orig_idx, msg in sorted(scored_middle, key=lambda x: x[1])
                if orig_idx in kept_indices
            ]
            evicted_total = len(middle) - len(retained_middle)

            marker_note = (
                f"[EigenLoom Quotient Compaction: retained {len(retained_middle)}/{len(middle)} "
                f"middle turns, deduplicated {evicted_count}, pruned {evicted_total} low-saliency turns. "
                f"Use eigenloom_context_grep to inspect archived turns.]"
            )
            if memory_context:
                marker_note = f"{marker_note}\nMemory Summary: {memory_context.strip()}"

            marker_msg: dict[str, Any] = {"role": "system", "content": marker_note}
            compacted = head + [marker_msg] + retained_middle + tail
            after_tokens = self._estimate_tokens(compacted)
            self._bus.cost_ledger.tokens_saved_by_quotient += max(
                0, before_tokens - after_tokens
            )
            self.compression_count += 1
            return compacted

    def on_turn_complete(
        self,
        messages: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            if usage:
                self.update_from_response(usage)
            if messages:
                self._store.index_quotient_messages(self._session_id, messages)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            pct = (
                min(
                    float(self._spec.c_int("one_hundred")),
                    float(max(0, self.last_prompt_tokens))
                    / float(self.context_length)
                    * float(self._spec.c_int("one_hundred")),
                )
                if self.context_length > 0
                else 0.0
            )
            return {
                "engine": self.name,
                "last_prompt_tokens": max(0, self.last_prompt_tokens),
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": pct,
                "compression_count": self.compression_count,
                "tokens_saved_by_quotient": self._bus.cost_ledger.tokens_saved_by_quotient,
                "prompt_cache_identity_hits": self._bus.cost_ledger.prompt_cache_identity_hits,
            }

    def clone_for_agent(self) -> "EigenLoomContextEngine":
        """Create per-agent instance sharing durable store & weight bus without copying locks."""
        with self._lock:
            clone = EigenLoomContextEngine(
                bus=self._bus,
                store=self._store,
                spec=self._spec,
                context_length=self.context_length,
                threshold_percent=self._config_threshold_percent,
            )
            clone.protect_first_n = self.protect_first_n
            clone.protect_last_n = self.protect_last_n
            return clone

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(CONTEXT_GREP_SCHEMA)]

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if name != "eigenloom_context_grep":
            return json.dumps({"error": f"Unknown ContextEngine tool: {name}"})
        query = str(args.get("query", ""))
        limit = args.get("limit")
        hits = self._store.grep_quotient_messages(
            query, limit=int(limit) if limit is not None else None
        )
        return json.dumps({"query": query, "count": len(hits), "matches": hits})

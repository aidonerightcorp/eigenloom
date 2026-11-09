"""Profile-scoped SQLite WAL storage for CAS task fission, fencing leases, quorum, trust, and Hopfield memory."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import time
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

try:
    from plugins.plugin_storage import plugin_data_dir as _hermes_plugin_data_dir
except ImportError:
    _hermes_plugin_data_dir = None  # type: ignore[assignment]

try:
    from hermes_constants import get_hermes_home as _hermes_get_home
except ImportError:
    _hermes_get_home = None  # type: ignore[assignment]


def resolve_eigenloom_db_path(hermes_home: str | Path | None = None) -> Path:
    """Resolve profile-scoped SQLite WAL path without reading environment variables directly."""
    if hermes_home:
        base = Path(hermes_home) / "plugin-data" / "eigenloom"
        base.mkdir(parents=True, exist_ok=True)
        return base / "eigenloom.db"
    if _hermes_plugin_data_dir is not None:
        try:
            base = Path(_hermes_plugin_data_dir("eigenloom"))
            base.mkdir(parents=True, exist_ok=True)
            return base / "eigenloom.db"
        except Exception:
            pass
    if _hermes_get_home is not None:
        try:
            base = Path(_hermes_get_home()) / "plugin-data" / "eigenloom"
            base.mkdir(parents=True, exist_ok=True)
            return base / "eigenloom.db"
        except Exception:
            pass
    fallback = Path(tempfile.gettempdir()) / "eigenloom-standalone"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback / "eigenloom.db"


class EigenLoomStore:
    """Thread-safe profile-scoped SQLite WAL state store."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        hermes_home: str | Path | None = None,
        spec: ManifoldConfigSpec | None = None,
        manifold: MatryoshkaManifold | None = None,
    ) -> None:
        self._spec = spec or get_manifold_spec()
        self.manifold = manifold or MatryoshkaManifold(self._spec)
        self.db_path = (
            Path(db_path)
            if db_path is not None
            else resolve_eigenloom_db_path(hermes_home)
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # In-process Dawid-Skene confusion counts, EigenTrust endorsements, and tier LinUCB Woodbury state
        self._actor_stats: dict[str, list[float]] = {}  # [true_pos_prior, false_pos_prior]
        self._endorsements: dict[tuple[str, str], float] = {}
        r_dim = int(self._spec.section("matryoshka_manifold")["coarse_dim"])
        self._tier_a_inv: dict[str, Matrix] = {}
        self._tier_b_vec: dict[str, Vector] = {}
        for tier in self._spec.section("delegation_tiers"):
            tid = str(tier["tier_id"])
            self._tier_a_inv[tid] = identity_matrix(r_dim)
            self._tier_b_vec[tid] = [0.0 for _ in range(r_dim)]

        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        timeout_sec = float(self._spec.c_int("busy_timeout_ms")) / float(
            self._spec.c_int("one_thousand")
        )
        conn = sqlite3.connect(
            str(self.db_path), timeout=timeout_sec, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _initialize_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS fission_tasks (
                        task_id TEXT PRIMARY KEY,
                        dag_id TEXT NOT NULL,
                        step_order INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        stratum TEXT NOT NULL,
                        interface_contract TEXT NOT NULL,
                        depends_on_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        fencing_version INTEGER NOT NULL DEFAULT 0,
                        fencing_stamp TEXT NOT NULL DEFAULT '',
                        lease_holder TEXT NOT NULL DEFAULT '',
                        lease_expires_at REAL NOT NULL DEFAULT 0.0,
                        maker_id TEXT NOT NULL DEFAULT '',
                        checker_id TEXT NOT NULL DEFAULT '',
                        output_summary TEXT NOT NULL DEFAULT '',
                        task2vec_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS verification_records (
                        record_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        maker_id TEXT NOT NULL,
                        checker_id TEXT NOT NULL,
                        passed INTEGER NOT NULL,
                        quality_score REAL NOT NULL,
                        posterior_truth REAL NOT NULL,
                        woodbury_trace REAL NOT NULL,
                        notes TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS hopfield_attractors (
                        attractor_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        saliency REAL NOT NULL,
                        trace2vec_json TEXT NOT NULL,
                        cartridge_b64 TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS quotient_messages (
                        msg_uid TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        saliency REAL NOT NULL,
                        trace2vec_json TEXT NOT NULL,
                        turn_index INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS manifold_snapshots (
                        snapshot_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def fission_dag(
        self,
        goal: str,
        *,
        subtasks: Sequence[Mapping[str, Any]] | None = None,
        max_fanout: int | None = None,
    ) -> dict[str, Any]:
        """Decompose a goal into a content-addressed SQLite WAL DAG with interface contracts."""
        now = time.time()
        fanout_cap = int(
            max_fanout if max_fanout is not None else self._spec.c_int("four")
        )
        prefix_len = self._spec.c_int("uid_digest_chars")
        dag_id = "dag_" + hashlib.blake2s(
            goal.encode("utf-8"), digest_size=self._spec.c_int("eight")
        ).hexdigest()[:prefix_len]

        if subtasks:
            raw_nodes = list(subtasks)[:fanout_cap]
        else:
            clauses = [
                seg.strip()
                for seg in goal.replace(";", ".").split(".")
                if seg.strip()
            ]
            if len(clauses) < self._spec.c_int("two"):
                clauses = [
                    f"Specify entities and invariants for: {goal}",
                    f"Implement core operator logic for: {goal}",
                    f"Verify independent invariants and edge behavior for: {goal}",
                ]
            strata_cycle = ["architect", "compile", "verify", "reflex"]
            raw_nodes = [
                {
                    "title": clause,
                    "stratum": strata_cycle[idx % len(strata_cycle)],
                    "interface_contract": f"contract::{dag_id}::stage_{idx}",
                }
                for idx, clause in enumerate(clauses[:fanout_cap])
            ]

        created_tasks: list[dict[str, Any]] = []
        prev_id: str | None = None

        with self._lock:
            conn = self._connect()
            try:
                for idx, node in enumerate(raw_nodes):
                    title = str(node.get("title", f"Subtask {idx}"))
                    stratum = str(node.get("stratum", "compile"))
                    contract = str(
                        node.get(
                            "interface_contract", f"contract::{dag_id}::node_{idx}"
                        )
                    )
                    deps = list(node.get("depends_on", [prev_id] if prev_id else []))
                    task_key = f"{dag_id}::{idx}::{title}"
                    task_id = "task_" + hashlib.blake2s(
                        task_key.encode("utf-8"),
                        digest_size=self._spec.c_int("eight"),
                    ).hexdigest()[:prefix_len]
                    t_vec = self.manifold.embed_task(
                        title, stratum=stratum, fanout=len(raw_nodes)
                    )

                    existing = conn.execute(
                        "SELECT status, maker_id, checker_id, output_summary FROM fission_tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO fission_tasks (
                                task_id, dag_id, step_order, title, stratum, interface_contract,
                                depends_on_json, status, task2vec_json, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                            """,
                            (
                                task_id,
                                dag_id,
                                idx,
                                title,
                                stratum,
                                contract,
                                json.dumps(deps),
                                json.dumps(list(t_vec.values)),
                                now,
                            ),
                        )
                        status_val = "pending"
                    else:
                        status_val = str(existing["status"])

                    created_tasks.append(
                        {
                            "task_id": task_id,
                            "step_order": idx,
                            "title": title,
                            "stratum": stratum,
                            "interface_contract": contract,
                            "depends_on": deps,
                            "status": status_val,
                            "task2vec_16d": t_vec.slice_prefix(
                                self._spec.section("matryoshka_manifold")["coarse_dim"]
                            ),
                        }
                    )
                    prev_id = task_id
                conn.commit()
            finally:
                conn.close()

        # Synthesize minimum-entropy disambiguation questions when ambiguity exists
        questions = [
            {
                "question_id": f"{dag_id}::q_scope",
                "prompt": "Should execution prioritize strict latency bounds or exhaustive verification coverage?",
                "options": ["balanced_pareto", "strict_latency", "exhaustive_verification"],
                "expected_entropy_bits_reduced": self._spec.c_float("half") + self._spec.c_float("quarter"),
            }
        ]
        return {
            "dag_id": dag_id,
            "task_count": len(created_tasks),
            "tasks": created_tasks,
            "disambiguation_questions": questions,
        }

    def claim_and_fence(
        self,
        task_id: str,
        actor_id: str,
        *,
        lease_ttl_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Atomically claim a fenced lease via SQLite CAS or return a verified cache hit."""
        now = time.time()
        ttl = float(
            lease_ttl_seconds
            if lease_ttl_seconds is not None
            else self._spec.c_int("one_hundred") + self._spec.c_int("twenty")
        )
        prefix_len = self._spec.c_int("uid_digest_chars")

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM fission_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown task_id: {task_id}")

                if (
                    str(row["status"]) == "verified"
                    and str(row["maker_id"])
                    and str(row["checker_id"])
                    and str(row["maker_id"]) != str(row["checker_id"])
                ):
                    return {
                        "task_id": task_id,
                        "cache_hit": True,
                        "acquired": False,
                        "status": "verified",
                        "maker_id": str(row["maker_id"]),
                        "checker_id": str(row["checker_id"]),
                        "output_summary": str(row["output_summary"]),
                        "fencing_version": int(row["fencing_version"]),
                        "fencing_stamp": str(row["fencing_stamp"]),
                    }

                next_ver = int(row["fencing_version"]) + 1
                stamp_src = f"{task_id}:{actor_id}:{next_ver}:{now}"
                stamp = "fence_" + hashlib.blake2s(
                    stamp_src.encode("utf-8"),
                    digest_size=self._spec.c_int("eight"),
                ).hexdigest()[:prefix_len]
                expires = now + ttl

                cur = conn.execute(
                    """
                    UPDATE fission_tasks
                    SET status = 'leased',
                        fencing_version = ?,
                        fencing_stamp = ?,
                        lease_holder = ?,
                        lease_expires_at = ?,
                        maker_id = CASE WHEN maker_id = '' THEN ? ELSE maker_id END,
                        updated_at = ?
                    WHERE task_id = ?
                      AND fencing_version = ?
                      AND (status = 'pending' OR lease_expires_at <= ? OR lease_holder = ?)
                    """,
                    (
                        next_ver,
                        stamp,
                        actor_id,
                        expires,
                        actor_id,
                        now,
                        task_id,
                        int(row["fencing_version"]),
                        now,
                        actor_id,
                    ),
                )
                conn.commit()
                acquired = cur.rowcount == 1
                latest = conn.execute(
                    "SELECT * FROM fission_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
            finally:
                conn.close()

        return {
            "task_id": task_id,
            "cache_hit": False,
            "acquired": acquired,
            "fencing_version": int(latest["fencing_version"]),
            "fencing_stamp": str(latest["fencing_stamp"]) if acquired else "",
            "lease_holder": str(latest["lease_holder"]),
            "lease_expires_at": float(latest["lease_expires_at"]),
            "resume_packet": {
                "dag_id": str(latest["dag_id"]),
                "title": str(latest["title"]),
                "stratum": str(latest["stratum"]),
                "interface_contract": str(latest["interface_contract"]),
                "depends_on": json.loads(str(latest["depends_on_json"])),
            },
        }

    def verify_record(
        self,
        task_id: str,
        *,
        maker_id: str,
        checker_id: str,
        passed: bool,
        quality_score: float = 1.0,
        output_summary: str = "",
        tier_id: str = "verify_tier",
        notes: str = "",
    ) -> dict[str, Any]:
        """Record independent maker != checker verification and update Woodbury, Dawid-Skene, and EigenTrust."""
        if not maker_id or not checker_id:
            raise ValueError("Both maker_id and checker_id are required")
        if maker_id == checker_id:
            raise ValueError(
                f"Independent verification violation: maker_id '{maker_id}' cannot equal checker_id '{checker_id}'"
            )

        now = time.time()
        q_clamped = max(0.0, min(1.0, float(quality_score)))
        prefix_len = self._spec.c_int("uid_digest_chars")

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM fission_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                t_vals = (
                    json.loads(str(row["task2vec_json"]))
                    if row is not None
                    else list(self.manifold.embed_task(task_id, stratum="verify").values)
                )
                t_vec = MatryoshkaVector.from_list(
                    t_vals, space="task2vec", spec=self._spec
                )

                # 1. Dawid-Skene Dirichlet confusion posterior update
                prior = float(self._spec.c_int("two"))
                c_stats = self._actor_stats.setdefault(checker_id, [prior, 1.0])
                m_stats = self._actor_stats.setdefault(maker_id, [prior, 1.0])
                if passed:
                    c_stats[0] += q_clamped
                    m_stats[0] += q_clamped
                else:
                    c_stats[1] += 1.0
                    m_stats[1] += 1.0 - q_clamped

                tp_rate = c_stats[0] / max(c_stats[0] + c_stats[1], 1.0)
                fp_rate = c_stats[1] / max(c_stats[0] + c_stats[1], 1.0)
                if passed:
                    num = tp_rate * self._spec.c_float("half")
                    den = num + fp_rate * self._spec.c_float("half")
                else:
                    num = (1.0 - tp_rate) * self._spec.c_float("half")
                    den = num + (1.0 - fp_rate) * self._spec.c_float("half")
                posterior_truth = num / max(den, self._spec.c_float("micro"))

                # 2. EigenTrust endorsement update + power iteration stationary reputation
                if passed:
                    edge = (checker_id, maker_id)
                    self._endorsements[edge] = (
                        self._endorsements.get(edge, 0.0) + q_clamped
                    )
                trust_scores = self.compute_eigentrust()

                # 3. Tier LinUCB Recursive Woodbury covariance update
                use_tier = (
                    tier_id
                    if tier_id in self._tier_a_inv
                    else next(iter(self._tier_a_inv.keys()))
                )
                r_dim = int(self._spec.section("matryoshka_manifold")["coarse_dim"])
                phi = t_vec.slice_prefix(r_dim)
                self._tier_a_inv[use_tier] = woodbury_rank1_update(
                    self._tier_a_inv[use_tier], phi, damping=1.0
                )
                signed_reward = q_clamped if passed else -self._spec.c_float("quarter")
                self._tier_b_vec[use_tier] = [
                    bi + signed_reward * pi
                    for bi, pi in zip(self._tier_b_vec[use_tier], phi)
                ]
                woodbury_trace = sum(
                    self._tier_a_inv[use_tier][i][i] for i in range(r_dim)
                )

                rec_id = "rec_" + hashlib.blake2s(
                    f"{task_id}:{maker_id}:{checker_id}:{now}".encode("utf-8"),
                    digest_size=self._spec.c_int("eight"),
                ).hexdigest()[:prefix_len]

                conn.execute(
                    """
                    INSERT INTO verification_records (
                        record_id, task_id, maker_id, checker_id, passed,
                        quality_score, posterior_truth, woodbury_trace, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec_id,
                        task_id,
                        maker_id,
                        checker_id,
                        1 if passed else 0,
                        q_clamped,
                        posterior_truth,
                        woodbury_trace,
                        notes,
                        now,
                    ),
                )
                if row is not None:
                    new_status = "verified" if passed else "pending"
                    conn.execute(
                        """
                        UPDATE fission_tasks
                        SET status = ?, maker_id = ?, checker_id = ?, output_summary = ?, updated_at = ?
                        WHERE task_id = ?
                        """,
                        (
                            new_status,
                            maker_id,
                            checker_id,
                            output_summary or str(row["output_summary"]),
                            now,
                            task_id,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

        return {
            "record_id": rec_id,
            "task_id": task_id,
            "maker_id": maker_id,
            "checker_id": checker_id,
            "passed": passed,
            "dawid_skene_posterior_truth": float(posterior_truth),
            "eigentrust_scores": trust_scores,
            "tier_id": use_tier,
            "woodbury_inverse_trace": float(woodbury_trace),
        }

    def compute_eigentrust(self) -> dict[str, float]:
        """Compute stationary EigenTrust reputation vector via damped power iteration."""
        actors = sorted(self._actor_stats.keys())
        if not actors:
            return {}
        n = len(actors)
        idx_map = {act: i for i, act in enumerate(actors)}
        uniform = 1.0 / float(n)
        alpha = self._spec.c_float("tenth")
        steps = self._spec.c_int("twelve")

        c_mat: Matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for (src, dst), weight in self._endorsements.items():
            if src in idx_map and dst in idx_map and weight > 0.0:
                c_mat[idx_map[src]][idx_map[dst]] += weight

        for r in range(n):
            row_sum = sum(c_mat[r])
            if row_sum > 0.0:
                c_mat[r] = [val / row_sum for val in c_mat[r]]
            else:
                c_mat[r] = [uniform for _ in range(n)]

        t_vec: Vector = [uniform for _ in range(n)]
        for _ in range(steps):
            nxt = [0.0 for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    nxt[j] += t_vec[i] * c_mat[i][j]
            t_vec = [(1.0 - alpha) * nxt[j] + alpha * uniform for j in range(n)]
        return {actors[i]: float(t_vec[i]) for i in range(n)}

    def score_delegation_tier(
        self, tier_id: str, task_vec: MatryoshkaVector, ucb_alpha: float
    ) -> float:
        with self._lock:
            r_dim = int(self._spec.section("matryoshka_manifold")["coarse_dim"])
            a_inv = self._tier_a_inv.get(tier_id)
            b_vec = self._tier_b_vec.get(tier_id)
            if a_inv is None or b_vec is None:
                return 0.0
            phi = task_vec.slice_prefix(r_dim)
            theta = matvec(a_inv, b_vec)
            mean_r = vec_dot(theta, phi)
            var_r = max(0.0, vec_dot(phi, matvec(a_inv, phi)))
            return float(mean_r + float(ucb_alpha) * math.sqrt(var_r))

    def recent_verified_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        cap = int(limit if limit is not None else self._spec.c_int("sixteen"))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT vr.record_id, vr.task_id, vr.quality_score, vr.posterior_truth, ft.task2vec_json, ft.title
                    FROM verification_records AS vr
                    LEFT JOIN fission_tasks AS ft ON ft.task_id = vr.task_id
                    WHERE vr.passed = 1
                    ORDER BY vr.created_at DESC
                    LIMIT ?
                    """,
                    (cap,),
                ).fetchall()
            finally:
                conn.close()
        out: list[dict[str, Any]] = []
        for r in rows:
            t_json = r["task2vec_json"]
            out.append(
                {
                    "record_id": str(r["record_id"]),
                    "task_id": str(r["task_id"]),
                    "quality_score": float(r["quality_score"]),
                    "summary": str(r["title"] or r["task_id"]),
                    "task2vec": json.loads(str(t_json)) if t_json else [],
                }
            )
        return out

    def store_attractor(
        self,
        session_id: str,
        summary: str,
        *,
        saliency: float | None = None,
        cartridge_b64: str = "",
    ) -> dict[str, Any]:
        """Store an episodic Hopfield attractor and optional micro-cartridge in SQLite WAL."""
        now = time.time()
        sal = self._spec.c_float("half") if saliency is None else max(0.0, min(1.0, float(saliency)))
        vec = self.manifold.embed_trace(summary, saliency=sal)
        prefix_len = self._spec.c_int("uid_digest_chars")
        aid = "att_" + hashlib.blake2s(
            f"{session_id}:{summary}".encode("utf-8"),
            digest_size=self._spec.c_int("eight"),
        ).hexdigest()[:prefix_len]

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hopfield_attractors (
                        attractor_id, session_id, summary, saliency, trace2vec_json, cartridge_b64, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        aid,
                        session_id,
                        summary,
                        sal,
                        json.dumps(list(vec.values)),
                        cartridge_b64,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        return {
            "attractor_id": aid,
            "session_id": session_id,
            "saliency": sal,
            "has_cartridge": bool(cartridge_b64),
        }

    def recall_attractors(
        self,
        query_text: str,
        *,
        top_k: int | None = None,
        beta: float | None = None,
    ) -> dict[str, Any]:
        """Retrieve episodic attractors via Matryoshka cascade + continuous Hopfield energy attention."""
        k_cap = int(top_k if top_k is not None else self._spec.c_int("four"))
        inv_temp = float(beta if beta is not None else self._spec.c_int("eight"))
        q_vec = self.manifold.embed_trace(query_text)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT attractor_id, session_id, summary, saliency, trace2vec_json, cartridge_b64 FROM hopfield_attractors ORDER BY created_at DESC LIMIT ?",
                    (self._spec.c_int("two_fifty_six"),),
                ).fetchall()
            finally:
                conn.close()

        candidates: list[tuple[str, MatryoshkaVector, dict[str, Any]]] = []
        for r in rows:
            m_vec = MatryoshkaVector.from_list(
                json.loads(str(r["trace2vec_json"])), space="trace2vec", spec=self._spec
            )
            candidates.append(
                (
                    str(r["attractor_id"]),
                    m_vec,
                    {
                        "session_id": str(r["session_id"]),
                        "summary": str(r["summary"]),
                        "saliency": float(r["saliency"]),
                        "has_cartridge": bool(r["cartridge_b64"]),
                    },
                )
            )

        cascade = self.manifold.progressive_cascade_search(q_vec, candidates, top_k=k_cap)
        matches = cascade["matches"]
        if matches:
            sims = [float(m["similarity"]) for m in matches]
            hopfield_weights = softmax(sims, temperature=1.0 / max(inv_temp, 1.0))
            for m, hw in zip(matches, hopfield_weights):
                m["hopfield_attention_weight"] = float(hw)

        return {
            "query": query_text,
            "matches": matches,
            "matryoshka_compute_savings_ratio": cascade["compute_savings_ratio"],
            "mac_operations_used": cascade["mac_operations_used"],
        }

    def index_quotient_messages(
        self, session_id: str, messages: Sequence[Mapping[str, Any]]
    ) -> int:
        """Deduplicate and index conversation turns by content-addressed UID for fast grep."""
        prefix_len = self._spec.c_int("hex_digest_prefix_chars")
        inserted = 0
        with self._lock:
            conn = self._connect()
            try:
                for idx, msg in enumerate(messages):
                    role = str(msg.get("role", "user"))
                    raw_content = msg.get("content", "")
                    content = (
                        raw_content
                        if isinstance(raw_content, str)
                        else json.dumps(raw_content)
                    )
                    if not content.strip():
                        continue
                    uid = "msg_" + hashlib.blake2s(
                        f"{role}:{content}".encode("utf-8"),
                        digest_size=self._spec.c_int("eight"),
                    ).hexdigest()[:prefix_len]
                    sal = min(
                        1.0,
                        float(len(set(content.lower().split())))
                        / float(self._spec.c_int("sixty_four")),
                    )
                    t_vec = self.manifold.embed_trace(content, role=role, saliency=sal)
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO quotient_messages (
                            msg_uid, session_id, role, content, saliency, trace2vec_json, turn_index
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uid,
                            session_id,
                            role,
                            content,
                            sal,
                            json.dumps(list(t_vec.values)),
                            idx,
                        ),
                    )
                    inserted += cur.rowcount
                conn.commit()
            finally:
                conn.close()
        return inserted

    def grep_quotient_messages(
        self, query: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Hybrid lexical + Trace2Vec Matryoshka search over the message-UID quotient DAG."""
        cap = int(limit if limit is not None else self._spec.c_int("eight"))
        q_lower = query.lower().strip()
        q_vec = self.manifold.embed_trace(query)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT msg_uid, session_id, role, content, saliency, trace2vec_json, turn_index FROM quotient_messages ORDER BY turn_index DESC LIMIT ?",
                    (self._spec.c_int("two_fifty_six"),),
                ).fetchall()
            finally:
                conn.close()

        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            content = str(r["content"])
            lexical_hit = 1.0 if (q_lower and q_lower in content.lower()) else 0.0
            m_vec = MatryoshkaVector.from_list(
                json.loads(str(r["trace2vec_json"])), space="trace2vec", spec=self._spec
            )
            sem_sim = q_vec.cosine(m_vec)
            sal = float(r["saliency"])
            combined = (
                lexical_hit
                + self._spec.c_float("half") * sem_sim
                + self._spec.c_float("quarter") * sal
            )
            scored.append(
                (
                    combined,
                    {
                        "msg_uid": str(r["msg_uid"]),
                        "role": str(r["role"]),
                        "turn_index": int(r["turn_index"]),
                        "saliency": sal,
                        "semantic_similarity": float(sem_sim),
                        "lexical_match": bool(lexical_hit > 0.0),
                        "snippet": content[: self._spec.c_int("two_fifty_six")],
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:cap]]

    def save_snapshot(self, key: str, payload: Mapping[str, Any]) -> None:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO manifold_snapshots (snapshot_key, payload_json, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (key, json.dumps(payload), now),
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def read_readonly_snapshot(db_path: Path) -> dict[str, Any]:
        """Read persisted status and counts in SQLite read-only mode without write locks."""
        if not db_path.exists():
            return {"persisted": False, "tasks": 0, "verifications": 0, "attractors": 0, "snapshot": None}
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            t_cnt = int(conn.execute("SELECT COUNT(*) AS c FROM fission_tasks").fetchone()["c"])
            v_cnt = int(conn.execute("SELECT COUNT(*) AS c FROM verification_records").fetchone()["c"])
            a_cnt = int(conn.execute("SELECT COUNT(*) AS c FROM hopfield_attractors").fetchone()["c"])
            snap_row = conn.execute(
                "SELECT payload_json FROM manifold_snapshots WHERE snapshot_key = 'latest_status'"
            ).fetchone()
            snap = json.loads(str(snap_row["payload_json"])) if snap_row is not None else None
            return {
                "persisted": True,
                "tasks": t_cnt,
                "verifications": v_cnt,
                "attractors": a_cnt,
                "snapshot": snap,
            }
        finally:
            conn.close()

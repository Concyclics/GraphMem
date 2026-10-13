"""Single-writer, crash-recoverable persistent Memory facade."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping, Sequence

from .config import ServiceConfig
from ..answer import AnswerConfig, AnswerStage
from ..answer.optimized import prepare_optimized_readout
from ..build.canonicalize import PredicateCanonicalizer
from ..build.pipeline import GraphBuildPipeline
from ..build.semantic import QwenSemanticDistiller
from ..config import runtime_config_hash
from ..domain import SourceTurn, canonical_json, stable_id
from ..embedding import QwenEmbeddingIndex
from ..retrieval.navigator import GraphNavigator
from ..storage import SQLiteGraphStore
from ..tokenization import resolve_token_counter


class MemoryNotFound(KeyError):
    pass


class MemoryConflict(ValueError):
    pass


class MemoryNotReady(MemoryConflict):
    pass


class ModelUnavailable(RuntimeError):
    pass


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class MemoryService:
    """Append, rebuild, retrieve and answer in a tenant-scoped data directory.

    One process owns a directory; calls are serialized so raw rows, graph views
    and prompt sources cannot cross revisions. A failed build leaves committed
    raw messages pending, recoverable by repeating add or calling rebuild.
    """

    def __init__(self, config: ServiceConfig, *, llm_client: Any = None,
                 embedding_client: Any = None, answer_client: Any = None) -> None:
        self.config = config
        self.build_config, self.runtime = config.profiles()
        self.counter = resolve_token_counter(config.llm_model,
                                             require_exact=config.require_exact_tokenizer)
        self._mutex = threading.RLock()
        self._closed = False
        self._clients = {"llm": llm_client, "embedding": embedding_client,
                         "answer": answer_client}
        self._owned_clients: list[Any] = []
        self._embedding: QwenEmbeddingIndex | None = None
        self._navigator: GraphNavigator | None = None
        self.data_dir = Path(config.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lease = (self.data_dir / "writer.lock").open("a+")
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lease.close()
            raise MemoryConflict("data_dir already has a MemoryService writer") from None
        try:
            self.store = SQLiteGraphStore(self.data_dir / "memory.sqlite")
            # Raw acknowledgements survive power loss as well as process exit.
            self.store._connection.execute("PRAGMA synchronous=FULL")
            with self.store.transaction() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS memory_service_state (
                    memory_id TEXT PRIMARY KEY REFERENCES conversations(memory_id) ON DELETE CASCADE,
                    tenant_id TEXT NOT NULL, external_id TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    ready_revision INTEGER NOT NULL DEFAULT 0,
                    build_identity TEXT NOT NULL DEFAULT '',
                    UNIQUE(tenant_id, external_id))""")
                db.execute("""CREATE TABLE IF NOT EXISTS memory_service_writes (
                    memory_id TEXT NOT NULL REFERENCES memory_service_state(memory_id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    revision INTEGER NOT NULL, turn_ids_json TEXT NOT NULL,
                    PRIMARY KEY(memory_id, request_id))""")
            portable_build = asdict(self.build_config)
            portable_build["storage"].pop("sqlite_path", None)
            self.build_identity = _digest({
                "build": portable_build,
                "dense": config.dense_retrieval,
                "request_profile": config.request_profile})
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            self._lease.close()
            raise

    @contextmanager
    def _operation(self):
        with self._mutex:
            if self._closed:
                raise RuntimeError("MemoryService is closed")
            yield

    def _key(self, tenant_id: str, memory_id: str) -> str:
        return stable_id("service-memory", _identifier(tenant_id, "tenant_id"),
                         _identifier(memory_id, "memory_id"))

    def _state(self, key: str):
        row = self.store._read_one("SELECT * FROM memory_service_state WHERE memory_id=?", (key,))
        if row is None:
            raise MemoryNotFound("memory does not exist in this tenant")
        return row

    def _ready(self, key: str):
        row = self._state(key)
        if row["revision"] != row["ready_revision"] or row["build_identity"] != self.build_identity:
            raise MemoryNotReady("raw messages are durable; call rebuild before retrieval")
        return row

    def _client(self, kind: str):
        if self._clients[kind] is None:
            from openai import OpenAI
            prefix = "llm" if kind == "llm" else kind
            env = getattr(self.config, f"{prefix}_api_key_env")
            base_url = getattr(self.config, f"{prefix}_base_url")
            if kind == "answer":
                env = env or self.config.llm_api_key_env
                base_url = base_url or self.config.llm_base_url
            key = os.environ.get(env, "") if env else "local"
            if not key:
                raise ModelUnavailable(f"required model credential environment variable is unset: {env}")
            client = OpenAI(base_url=base_url, api_key=key, max_retries=0,
                            timeout=self.config.model_timeout_seconds)
            self._clients[kind] = client
            self._owned_clients.append(client)
        return self._clients[kind]

    def _index(self) -> QwenEmbeddingIndex | None:
        if self.config.dense_retrieval and self._embedding is None:
            self._embedding = QwenEmbeddingIndex(
                self.store, self.build_config, client=self._client("embedding"),
                **self.runtime.retrieval.embedding_options())
        return self._embedding

    @staticmethod
    def _messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(messages, (list, tuple)) or not 1 <= len(messages) <= 256:
            raise ValueError("messages must contain 1..256 message objects")
        normalized = []
        for row in messages:
            if not isinstance(row, Mapping) or set(row) - {"role", "content", "speaker", "listener", "timestamp"}:
                raise ValueError("unknown message fields")
            content = row.get("content")
            role = row.get("role", "user")
            if not isinstance(content, str) or not content.strip() or len(content) > 65536:
                raise ValueError("content must contain 1..65536 characters")
            if role not in {"user", "assistant", "system", "tool"}:
                raise ValueError("invalid message role")
            timestamp = row.get("timestamp")
            if timestamp is not None:
                if not isinstance(timestamp, str):
                    raise ValueError("timestamp must be ISO 8601 text or null")
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            speaker, listener = row.get("speaker", role), row.get("listener", "")
            if not isinstance(speaker, str) or not isinstance(listener, str):
                raise ValueError("speaker and listener must be strings")
            normalized.append(dict(role=role, content=content, speaker=speaker,
                                   listener=listener, timestamp=timestamp))
        if len(canonical_json(normalized).encode()) > 1024 * 1024:
            raise ValueError("message batch exceeds 1 MiB")
        return normalized

    def add(self, memory_id: str, messages: Sequence[Mapping[str, Any]], *,
            tenant_id: str, session_id: str, request_id: str) -> dict[str, Any]:
        """Durably append once and synchronously publish the updated graph.

        Retrying the same request_id and payload is safe across restarts.
        Reusing an id for different content raises MemoryConflict.
        """
        key = self._key(tenant_id, memory_id)
        _identifier(session_id, "session_id")
        _identifier(request_id, "request_id")
        rows = self._messages(messages)
        signature = _digest({"session_id": session_id, "messages": rows})
        session_key = stable_id("service-session", key, session_id)
        with self._operation():
            with self.store.transaction() as db:
                receipt = db.execute(
                    "SELECT * FROM memory_service_writes WHERE memory_id=? AND request_id=?",
                    (key, request_id)).fetchone()
                replayed = receipt is not None
                if receipt is not None:
                    if receipt["payload_hash"] != signature:
                        raise MemoryConflict("request_id was already used with a different payload")
                    turn_ids = json.loads(receipt["turn_ids_json"])
                else:
                    db.execute("INSERT OR IGNORE INTO conversations VALUES(?,?,?,?,?)",
                               (key, "memory-service", memory_id, signature, "graphmem-v5"))
                    db.execute("INSERT OR IGNORE INTO memory_service_state "
                               "(memory_id,tenant_id,external_id) VALUES(?,?,?)",
                               (key, tenant_id, memory_id))
                    state = db.execute("SELECT revision FROM memory_service_state WHERE memory_id=?",
                                       (key,)).fetchone()
                    revision = int(state[0]) + 1
                    session = db.execute("SELECT ordinal FROM sessions WHERE memory_id=? AND session_id=?",
                                         (key, session_key)).fetchone()
                    ordinal = (int(session[0]) if session else int(db.execute(
                        "SELECT COALESCE(MAX(ordinal),-1)+1 FROM sessions WHERE memory_id=?", (key,)).fetchone()[0]))
                    offset = int(db.execute("SELECT COALESCE(MAX(turn_index),-1)+1 FROM source_turns "
                                            "WHERE memory_id=? AND session_id=?", (key, session_key)).fetchone()[0])
                    db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?) ON CONFLICT(memory_id,session_id) "
                               "DO UPDATE SET content_hash=excluded.content_hash",
                               (session_key, key, ordinal, rows[0]["timestamp"], signature, "graphmem-v5"))
                    turns = [SourceTurn(
                        stable_id("service-turn", key, request_id, i), key, session_key, offset + i,
                        row["speaker"], row["listener"], row["role"], row["timestamp"], row["content"],
                        hashlib.sha256(row["content"].encode()).hexdigest()) for i, row in enumerate(rows)]
                    self.store._upsert_turns(db, turns)
                    turn_ids = [turn.turn_id for turn in turns]
                    db.execute("INSERT INTO memory_service_writes VALUES(?,?,?,?,?)",
                               (key, request_id, signature, revision, json.dumps(turn_ids)))
                    db.execute("UPDATE memory_service_state SET revision=? WHERE memory_id=?", (revision, key))
                    db.execute("UPDATE conversations SET content_hash=? WHERE memory_id=?",
                               (_digest([key, revision, signature]), key))
            state = self._state(key)
            if state["revision"] != state["ready_revision"] or state["build_identity"] != self.build_identity:
                self._build(key)
            return {**self.status(memory_id, tenant_id=tenant_id), "request_id": request_id,
                    "turn_ids": turn_ids, "replayed": replayed}

    def _build(self, key: str) -> None:
        self._navigator = None
        embedding = self._index()
        distiller = (QwenSemanticDistiller(
            self.store, self.build_config, "memory-service-v1", client=self._client("llm"),
            worker_limit=4, request_profile=self.config.request_profile)
            if self.config.semantic_build else None)
        canonicalizer = (PredicateCanonicalizer(
            self.store, self.build_config, client=self._client("embedding"))
            if embedding is not None and distiller is not None else None)
        pipeline = GraphBuildPipeline(
            self.store, dataset_hash="memory-service-v1", distiller=distiller,
            predicate_canonicalizer=canonicalizer,
            coarsen_vector_provider=embedding.embed_graph_nodes if embedding else None,
            relation_vector_provider=embedding.embed_graph_nodes if embedding else None)
        try:
            if embedding:
                embedding.index_memory(key)
            pipeline.build(key, self.build_config)
            if embedding:
                embedding.embed_graph_nodes(key, self.store.nodes(key))
            with self.store.transaction() as db:
                db.execute("UPDATE memory_service_state SET ready_revision=revision,build_identity=? "
                           "WHERE memory_id=?", (self.build_identity, key))
        except Exception as error:
            raise ModelUnavailable("raw messages are durable but indexing failed; retry the same "
                                   "request_id or call rebuild after restoring model service") from error

    def rebuild(self, memory_id: str, *, tenant_id: str) -> dict[str, Any]:
        key = self._key(tenant_id, memory_id)
        with self._operation():
            self._state(key)
            # Keep the pending marker durable even when rebuilding an already
            # ready memory fails halfway through graph/vector publication.
            with self.store.transaction() as db:
                db.execute("UPDATE memory_service_state SET build_identity='' WHERE memory_id=?", (key,))
            self._build(key)
            return self.status(memory_id, tenant_id=tenant_id)

    def status(self, memory_id: str, *, tenant_id: str) -> dict[str, Any]:
        key = self._key(tenant_id, memory_id)
        with self._operation():
            row = self._state(key)
            ready = row["revision"] == row["ready_revision"] and row["build_identity"] == self.build_identity
            return {"memory_id": memory_id, "tenant_id": tenant_id,
                    "status": "ready" if ready else "pending", "revision": row["revision"],
                    "ready_revision": row["ready_revision"], "graph_version": self.store.graph_version(key),
                    "turn_count": int(self.store._read_one(
                        "SELECT COUNT(*) FROM source_turns WHERE memory_id=?", (key,))[0])}

    def _navigate(self, key: str, query: str):
        if not isinstance(query, str) or not query.strip() or len(query) > 16000:
            raise ValueError("query must contain 1..16000 characters")
        self._ready(key)
        if self._navigator is None:
            embedding = self._index()
            self._navigator = GraphNavigator(
                self.store, token_counter=self.counter,
                dense_search=embedding.search if embedding else None,
                dense_search_many=embedding.search_many if embedding else None,
                fact_dense_search=(embedding.search_items if embedding else None),
                predicate_dense_search=(
                    lambda memory_id, query, ids, limit: embedding.search_items(
                        memory_id, query, ids, limit,
                        index_model_id=embedding.model_id + ":predicate-v1")) if embedding else None,
                **self.runtime.retrieval.navigator_options())
        return self._navigator.navigate(key, query, self.runtime.query_budget)

    def retrieve(self, memory_id: str, query: str, *, tenant_id: str) -> dict[str, Any]:
        with self._operation():
            key = self._key(tenant_id, memory_id)
            result = self._navigate(key, query)
            turns = {row.turn_id: row for row in self.store.turns(key)}
            return {"memory_id": memory_id, "graph_version": self.store.graph_version(key),
                    "evidence": [asdict(turns[t]) for t in result.retrieved_turn_ids],
                    "evidence_tokens": result.evidence_tokens, "trace": dict(result.trace),
                    "token_counter": self.counter.describe()}

    def _stage(self, *, answering=False):
        return AnswerStage(
            self.store, self.build_config, "memory-service-v1",
            answer_config=AnswerConfig.v5_73(
                max_output_tokens=self.config.max_output_tokens,
                sampling_frequency_penalty=(self.config.answer_frequency_penalty
                    if self.config.answer_request_profile == "qwen" else 0.0)),
            # Preparation is model independent; never construct a client or
            # require answer credentials until a completion is actually asked for.
            client=self._client("answer") if answering else object(),
            answer_model=self.config.answer_model or self.config.llm_model,
            answer_request_profile=self.config.answer_request_profile,
            answer_reasoning_effort=self.config.answer_reasoning_effort,
            require_exact_tokenizer=self.config.require_exact_tokenizer)

    def prepare(self, memory_id: str, query: str, *, tenant_id: str,
                question_date: str | None = None):
        """Return messages and provenance for the product's own answer model."""
        if question_date is not None:
            if not isinstance(question_date, str):
                raise ValueError("question_date must be ISO 8601 text or null")
            datetime.fromisoformat(question_date.replace("Z", "+00:00"))
        with self._operation():
            key = self._key(tenant_id, memory_id)
            navigation = self._navigate(key, query)
            stage = self._stage()
            prepared = stage.prepare(
                stable_id("service-question", key, query, question_date), query, navigation,
                self.runtime.query_budget, question_date=question_date)
            if self.config.readout == "v5_81":
                prepared = prepare_optimized_readout(
                    prepared, question=query, question_date=question_date, navigation=navigation,
                    turns={turn.turn_id: turn for turn in self.store.turns(key)},
                    counter=self.counter, max_prompt_tokens=self.runtime.query_budget.max_answer_tokens_hard,
                    embedding=self._index())
            return replace(prepared, trace={
                **prepared.trace, "retrieval": dict(navigation.trace),
                "graph_version": self.store.graph_version(key),
                "memory_revision": self._state(key)["ready_revision"]})

    def answer(self, memory_id: str, query: str, *, tenant_id: str,
               question_date: str | None = None):
        if not self.config.generate_answers:
            raise ModelUnavailable("generate_answers is disabled; use prepare with your product's LLM")
        with self._operation():
            prepared = self.prepare(memory_id, query, tenant_id=tenant_id, question_date=question_date)
            return self._stage(answering=True).complete(prepared)

    def health(self) -> dict[str, Any]:
        # Do not wait for the facade's model-call mutex: readiness monitoring
        # must still respond while a write is building a graph. Store reads
        # hold only SQLite's short-lived connection lock.
        if self._closed:
            raise RuntimeError("MemoryService is closed")
        pending = self.store._read_one("SELECT COUNT(*) FROM memory_service_state WHERE "
                                       "revision!=ready_revision OR build_identity!=?", (self.build_identity,))[0]
        return {"status": "ok", "profile": "v5_81_product", "persistent": True,
                "pending_memories": pending, "semantic_build": self.config.semantic_build,
                "dense_retrieval": self.config.dense_retrieval, "readout": self.config.readout,
                "generate_answers": self.config.generate_answers,
                "runtime_config_hash": runtime_config_hash(self.runtime),
                "token_counter": self.counter.describe()}

    def backup(self, destination: str | Path) -> Path:
        """SQLite online backup includes raw data, graph, model cache and receipts."""
        path = Path(destination).expanduser().resolve()
        with self._operation():
            with path.open("xb"):
                pass
            try:
                with sqlite3.connect(path) as target:
                    self.store._connection.backup(target)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return path

    def close(self) -> None:
        with self._mutex:
            if not self._closed:
                self._closed = True
                try:
                    self.store.close()
                    for client in self._owned_clients:
                        client.close()
                finally:
                    self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

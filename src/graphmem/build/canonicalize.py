from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from ..config import GraphMemV5Config
from ..domain import stable_id
from ..storage import SQLiteGraphStore


PredicateKey = tuple[str, str, str, str, str]


class PredicateCanonicalizer:
    """Conservative mutual-nearest predicate clustering within compatible fact slots."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 client: Any | None = None) -> None:
        self.store = store; self.config = config
        self.model_id = config.models.embedding_model
        self.index_model_id = self.model_id + ":predicate-v1"
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.embedding_base_url, api_key="local")
        self.client = client

    def canonicalize(self, memory_id: str, keys: Sequence[PredicateKey]) -> dict[PredicateKey, str]:
        unique = tuple(sorted(set(keys))); predicates = tuple(sorted({key[1] for key in unique}))
        vectors = self._vectors(memory_id, predicates)
        groups: dict[tuple[str, str, str, str], list[PredicateKey]] = defaultdict(list)
        for key in unique:
            groups[(key[0], key[2], key[3], key[4])].append(key)
        result = {key: key[1] for key in unique}; threshold = self.config.edges.predicate_embedding_threshold
        for compatible in groups.values():
            labels = sorted({key[1] for key in compatible})
            if len(labels) < 2:
                continue
            nearest = {}
            for left in labels:
                choices = [(float(np.dot(vectors[left], vectors[right])), right)
                           for right in labels if right != left]
                nearest[left] = max(choices, key=lambda row: (row[0], row[1]))
            pairs = []
            for left in labels:
                score, right = nearest[left]
                if score >= threshold and nearest.get(right, (0.0, ""))[1] == left:
                    pairs.append(tuple(sorted((left, right))))
            representative = {label: label for label in labels}
            for left, right in sorted(set(pairs)):
                winner = min(representative[left], representative[right])
                loser = max(representative[left], representative[right])
                for label, current in tuple(representative.items()):
                    if current == loser:
                        representative[label] = winner
            for key in compatible:
                result[key] = representative[key[1]]
        return result

    def _vectors(self, memory_id: str, predicates: Sequence[str]) -> dict[str, np.ndarray]:
        rows = {}; pending = []
        for predicate in predicates:
            item_id = stable_id("predicate", memory_id, predicate)
            content_hash = hashlib.sha256(predicate.encode()).hexdigest()
            row = self.store._connection.execute(
                "SELECT dimension,vector,content_hash FROM embeddings WHERE memory_id=? AND model_id=? AND item_id=?",
                (memory_id, self.index_model_id, item_id)).fetchone()
            if row and row["content_hash"] == content_hash:
                rows[predicate] = np.frombuffer(row["vector"], dtype=np.float32,
                                                count=int(row["dimension"])).copy()
            else:
                pending.append((predicate, item_id, content_hash))
        if pending:
            started = time.perf_counter()
            response = self.client.embeddings.create(model=self.model_id,
                                                     input=[row[0] for row in pending])
            latency = (time.perf_counter() - started) * 1000
            ordered = sorted(response.data, key=lambda row: row.index)
            inserts = []
            for (predicate, item_id, content_hash), item in zip(pending, ordered):
                vector = np.asarray(item.embedding, dtype=np.float32)
                rows[predicate] = vector; inserts.append((item_id, content_hash, vector))
            self.store.upsert_embeddings(memory_id, self.index_model_id, inserts)
            usage = getattr(response, "usage", None)
            tokens = int(getattr(usage, "prompt_tokens", 0) or
                         getattr(usage, "total_tokens", 0) or 0)
            self.store.log_embedding_call(
                stable_id("embedding-call", memory_id, "predicate", *(row[2] for row in pending)),
                memory_id, self.index_model_id, len(pending), tokens, latency)
        return {key: value / max(float(np.linalg.norm(value)), 1e-12) for key, value in rows.items()}

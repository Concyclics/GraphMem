from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..config import CacheIdentity, GraphMemV5Config, config_hash
from ..domain import SourceTurn, canonical_json, stable_id
from ..storage import SQLiteGraphStore


PROMPT_VERSION = "graphmem-v5.1-scene-semantic-v1"
SYSTEM_PROMPT = """Extract grounded memory facts. Return compact JSON {"s":[{"i":scene_id,"m":summary,"f":[{"o":owner,"p":predicate,"v":value,"y":value_type,"g":scope,"n":"positive|negative","t":time_or_null,"c":confidence,"e":[{"i":turn_id,"a":start,"b":end}]}],"u":[]}]}. Every fact must cite exact supplied offsets. No invention, markdown, explanations, or reasoning. Summary <=64 tokens. Omit empty optional fields."""
HIERARCHY_PROMPT = """Compress supplied child semantic records for routing. Return compact JSON with summary, owners, predicates, values, scopes, times, child_postings mapping keys to child ID arrays, and aliases as arrays of equivalent value strings found in the children. Only copy supported values and IDs. Never put IDs in scopes or aliases. No markdown or reasoning."""


@dataclass(frozen=True, slots=True)
class SemanticFact:
    owner: str
    predicate: str
    value: str
    value_type: str
    scope: str
    polarity: str
    time: str | None
    confidence: float
    evidence: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class ScenePacket:
    scene_id: str
    summary: str
    facts: tuple[SemanticFact, ...]
    unresolved: tuple[str, ...]
    fallback: bool = False


class QwenSemanticDistiller:
    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, client: Any | None = None) -> None:
        self.store, self.config, self.dataset_hash = store, config, dataset_hash
        self.prompt_hash = hashlib.sha256((PROMPT_VERSION + SYSTEM_PROMPT + HIERARCHY_PROMPT).encode()).hexdigest()
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
        self.client = client

    def extract(self, memory_id: str, scenes: Sequence[Any]) -> tuple[ScenePacket, ...]:
        batches: list[list[Any]] = []
        current: list[Any] = []
        estimate = 0
        for scene in scenes:
            scene_tokens = sum(max(1, len(turn.raw_text.split()) * 13 // 10) for turn in scene.turns)
            if current and (len(current) >= self.config.models.semantic_batch_scenes
                            or estimate + scene_tokens > self.config.models.semantic_batch_input_tokens):
                batches.append(current); current = []; estimate = 0
            current.append(scene); estimate += scene_tokens
        if current:
            batches.append(current)
        packets: list[ScenePacket] = []
        workers = min(16, self.config.models.max_concurrency, max(1, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for rows in executor.map(lambda batch: self._extract_batch(memory_id, batch), batches):
                packets.extend(rows)
        return tuple(packets)

    def compress_many(self, memory_id: str, level: int,
                      requests: Sequence[tuple[str, Sequence[Mapping[str, Any]], int]]) -> tuple[Mapping[str, Any], ...]:
        workers = min(16, self.config.models.max_concurrency, max(1, len(requests)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return tuple(executor.map(
                lambda row: self.compress(memory_id, level, row[0], row[1], row[2]), requests
            ))

    def compress(self, memory_id: str, level: int, parent_id: str,
                 children: Sequence[Mapping[str, Any]], limit: int) -> Mapping[str, Any]:
        payload = {"level": level, "parent_id": parent_id, "children": list(children), "summary_limit": limit}
        response, _ = self._call(memory_id, f"hierarchy_l{level}", HIERARCHY_PROMPT, payload,
                                 len(children), max_tokens=1536 if level == 3 else 768)
        parsed = self._parse_objects(str(response.get("content", "")))
        row = parsed[0] if parsed else {}
        summary = str(row.get("summary", "")).strip()
        if not summary:
            summary = " ".join(str(child.get("summary", "")) for child in children)
        valid_ids = {str(child["child_id"]) for child in children}
        postings = {}
        for key, ids in dict(row.get("child_postings", {})).items() if isinstance(row.get("child_postings"), dict) else ():
            if isinstance(ids, list):
                postings[str(key)] = tuple(item for item in map(str, ids) if item in valid_ids)
        available_values = {str(value) for child in children for value in child.get("values", ())}
        aliases = []
        for group in row.get("aliases", ()) if isinstance(row.get("aliases"), list) else ():
            if isinstance(group, list):
                valid = tuple(dict.fromkeys(str(value) for value in group if str(value) in available_values))
                if len(valid) >= 2: aliases.append(valid)
        return {
            "summary": " ".join(summary.split()[:limit]),
            "owners": self._strings(row.get("owners")), "predicates": self._strings(row.get("predicates")),
            "values": self._strings(row.get("values")), "scopes": self._strings(row.get("scopes")),
            "times": self._strings(row.get("times")), "child_postings": postings,
            "aliases": tuple(aliases),
        }

    def _extract_batch(self, memory_id: str, batch: Sequence[Any]) -> list[ScenePacket]:
        payload = {"s": [{"i": scene.scene_id, "r": [
            {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp, "l": len(turn.raw_text),
             "t": turn.raw_text}
            for turn in scene.turns]} for scene in batch]}
        system = self._scene_prompt()
        response, _ = self._call(memory_id, "scene_semantic", system, payload, len(batch))
        objects = self._parse_objects(str(response.get("content", "")))
        root = objects[0] if objects else {}
        rows = root.get("s", root.get("scenes", [])) if isinstance(root, dict) else []
        if not isinstance(rows, list):
            rows = []
        by_scene = {str(row.get("i", row.get("scene_id"))): row for row in rows if isinstance(row, dict)}
        missing = [scene for scene in batch if scene.scene_id not in by_scene]
        if missing:
            repair_batches = [[scene] for scene in missing] if self.config.models.semantic_individual_repair else [missing]
            for repair_batch in repair_batches:
                repair_payload = {"repair": "Return complete valid JSON for only these missing scenes.",
                "s": [{"i": scene.scene_id, "r": [
                    {"i": turn.turn_id, "s": turn.speaker, "d": turn.timestamp,
                     "l": len(turn.raw_text), "t": turn.raw_text}
                    for turn in scene.turns]} for scene in repair_batch]}
                repaired, _ = self._call(memory_id, "scene_semantic_repair", system,
                    repair_payload, len(repair_batch),
                    max_tokens=self.config.models.semantic_repair_output_tokens)
                repaired_objects = self._parse_objects(str(repaired.get("content", "")))
                repaired_root = repaired_objects[0] if repaired_objects else {}
                for item in repaired_root.get("s", repaired_root.get("scenes", ())) if isinstance(repaired_root, dict) else ():
                    if isinstance(item, dict):
                        by_scene[str(item.get("i", item.get("scene_id")))] = item
        result = []
        for scene in batch:
            row = by_scene.get(scene.scene_id)
            result.append(self._validate_scene(scene, row) if row else self._fallback(scene))
        return result

    def _validate_scene(self, scene: Any, row: Mapping[str, Any]) -> ScenePacket:
        turns = {turn.turn_id: turn for turn in scene.turns}
        facts = []
        fact_rows = row.get("f", row.get("facts", ()))
        for item in (fact_rows[:self.config.models.semantic_max_facts_per_scene]
                     if isinstance(fact_rows, list) else ()):
            if not isinstance(item, dict):
                continue
            evidence = []
            for ref in item.get("e", item.get("evidence", ())):
                try:
                    turn = turns[str(ref.get("i", ref.get("turn_id")))]
                    start, end = int(ref.get("a", ref.get("start"))), int(ref.get("b", ref.get("end")))
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= start < end <= len(turn.raw_text):
                    evidence.append((turn.turn_id, start, end))
            owner = str(item.get("o", item.get("owner", ""))).strip()
            predicate = str(item.get("p", item.get("predicate", ""))).strip()
            value = str(item.get("v", item.get("value", ""))).strip()
            if not evidence or not owner or not predicate or not value:
                if owner and predicate and value:
                    for turn in turns.values():
                        start = turn.raw_text.casefold().find(value.casefold())
                        if start >= 0:
                            evidence.append((turn.turn_id, start, start + len(value))); break
                if not evidence or not owner or not predicate or not value:
                    continue
            facts.append(SemanticFact(owner, predicate, value,
                                      str(item.get("y", item.get("value_type", "unknown"))),
                                      str(item.get("g", item.get("scope", "scene"))),
                                      str(item.get("n", item.get("polarity", "positive"))),
                                      str(item.get("t", item.get("time"))) if item.get("t", item.get("time")) else None,
                                      min(1.0, max(0.0, float(item.get("c", item.get("confidence", 0.5))))), tuple(evidence)))
        summary = " ".join(str(row.get("m", row.get("summary", ""))).split()[
            :self.config.models.semantic_summary_tokens])
        if not summary:
            summary = " ".join(scene.summary.split()[:96])
        return ScenePacket(scene.scene_id, summary, tuple(facts),
                           self._strings(row.get("u", row.get("unresolved"))), False)

    def _scene_prompt(self) -> str:
        if (self.config.models.semantic_max_facts_per_scene == 12
                and self.config.models.semantic_summary_tokens == 64):
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + (
            f" Hard limits: at most {self.config.models.semantic_max_facts_per_scene} facts per scene; "
            f"summary at most {self.config.models.semantic_summary_tokens} tokens. Prioritize state, event, "
            "time, preference and cross-session routing facts; omit greetings and acknowledgements."
        )

    @staticmethod
    def _fallback(scene: Any) -> ScenePacket:
        return ScenePacket(scene.scene_id, " ".join(scene.summary.split()[:96]), (), ("llm_parse_fallback",), True)

    def _call(self, memory_id: str, stage: str, system: str, payload: Any,
              batch_size: int, max_tokens: int | None = None) -> tuple[Mapping[str, Any], Mapping[str, int]]:
        serialized = canonical_json(payload)
        semantic_settings = {
            "schema": self.config.schema_version, "model": self.config.models.llm_model,
            "batch_scenes": self.config.models.semantic_batch_scenes,
            "batch_input": self.config.models.semantic_batch_input_tokens,
            "scene_input": self.config.models.semantic_scene_input_tokens,
            "batch_output": self.config.models.semantic_batch_output_tokens,
        }
        if (self.config.models.semantic_max_facts_per_scene,
                self.config.models.semantic_summary_tokens,
                self.config.models.semantic_repair_output_tokens,
                self.config.models.semantic_constrained_json,
                self.config.models.semantic_individual_repair) != (12, 64, 4096, False, False):
            semantic_settings.update({
                "max_facts": self.config.models.semantic_max_facts_per_scene,
                "summary_tokens": self.config.models.semantic_summary_tokens,
                "repair_output": self.config.models.semantic_repair_output_tokens,
                "constrained_json": self.config.models.semantic_constrained_json,
                "individual_repair": self.config.models.semantic_individual_repair,
            })
        semantic_config = hashlib.sha256(canonical_json(semantic_settings).encode()).hexdigest()
        identity = CacheIdentity(self.dataset_hash, self.config.models.llm_model, self.prompt_hash,
                                 self.config.schema_version, semantic_config,
                                 stage + ":" + hashlib.sha256(serialized.encode()).hexdigest())
        key = identity.key(); request = {"model": self.config.models.llm_model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": serialized}],
            "temperature": 0, "max_tokens": max_tokens or self.config.models.semantic_batch_output_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        if self.config.models.semantic_constrained_json:
            request["response_format"] = {"type": "json_object"}
        cached = self.store.cache_get(key); started = time.perf_counter()
        if cached:
            response = cached["response"]; old = cached["usage"]
            usage = {"cached_input_tokens": int(old.get("uncached_input_tokens", 0)),
                     "uncached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                     "total_tokens": int(old.get("uncached_input_tokens", 0))}
            is_cached = True
        else:
            result = self.client.chat.completions.create(**request); message = result.choices[0].message
            if getattr(message, "reasoning_content", None):
                raise RuntimeError("semantic distillation returned reasoning content")
            response = {"content": message.content or "", "model": getattr(result, "model", "")}
            raw = getattr(result, "usage", None); prompt = int(getattr(raw, "prompt_tokens", 0) or 0)
            output = int(getattr(raw, "completion_tokens", 0) or 0)
            usage = {"cached_input_tokens": 0, "uncached_input_tokens": prompt,
                     "output_tokens": output, "reasoning_tokens": 0, "total_tokens": prompt + output}
            self.store.cache_put(key, stage, request, response, usage, self.prompt_hash); is_cached = False
        occurrence = self.store._connection.execute(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?", (memory_id, key)).fetchone()[0]
        self.store.log_llm_call(call_id=stable_id("llm-call", memory_id, key, is_cached, occurrence),
            memory_id=memory_id, stage=stage, cache_key=key, cached=is_cached, request=request,
            response=response, usage=usage, latency_ms=(time.perf_counter()-started)*1000,
            retry_count=0, batch_size=batch_size, prompt_hash=self.prompt_hash)
        return response, usage

    @staticmethod
    def _strings(value: Any) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip())) if isinstance(value, list) else ()

    @staticmethod
    def _parse_objects(text: str) -> list[Mapping[str, Any]]:
        clean = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            row = json.loads(clean)
            return [row] if isinstance(row, dict) else [item for item in row if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            decoder = json.JSONDecoder(); rows = []; index = 0
            while index < len(clean):
                while index < len(clean) and clean[index] in " \r\n\t,": index += 1
                try: row, index = decoder.raw_decode(clean, index)
                except json.JSONDecodeError: break
                if isinstance(row, dict): rows.append(row)
            return rows

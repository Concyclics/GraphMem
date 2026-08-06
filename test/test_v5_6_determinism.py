from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
from graphmem.domain import CandidateScore, EvidenceUnit, QueryBudget, SourceTurn, stable_id
from graphmem.retrieval import GraphNavigator, HarnessProfile
from graphmem.retrieval.packer import pack
from graphmem.storage import SQLiteGraphStore

from test_v5_1_llm_graph import StrictCompletions, _strict_config
from test_v5_gate_b_core import _store


TEST_DIR = Path(__file__).resolve().parent
SRC_DIR = TEST_DIR.parent / "src"


CITIES = ("Seattle", "Boston", "Austin", "Denver", "Chicago", "Portland")


class _CityCompletions:
    """Stub extractor: one ``visit`` fact per turn that names a city.

    The shared travel fixture yields a single proof unit, which would make the
    cross-process check pass no matter how the packer ordered things.  Several
    distinct values give the algebra several answer members, and therefore the
    packer several mandatory units to order.
    """

    def __init__(self) -> None:
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        payload = json.loads(request["messages"][1]["content"])
        scenes = []
        for scene in payload["s"]:
            facts = []
            for index, turn in enumerate(scene["r"]):
                city = next((name for name in CITIES if name in turn["t"]), None)
                if city:
                    facts.append({"o": "Alice", "p": "visit", "v": city, "g": "travel",
                                  "n": "positive", "r": [index], "q": city})
            scenes.append({"i": scene["i"], "f": facts})
        message = SimpleNamespace(content=json.dumps({"s": scenes}), reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], model="qwen30b",
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=100))


def _city_store(path: Path) -> SQLiteGraphStore:
    from graphmem.domain import Conversation, Session

    store = SQLiteGraphStore(path)
    memory_id = "travel"
    sessions, turns = [], []
    for session_index in range(3):
        session_id = f"s{session_index + 1}"
        sessions.append(Session(session_id, memory_id, session_index,
                                f"2025-0{session_index + 1}-01", f"{session_id}h"))
        for turn_index in range(4):
            city = CITIES[(session_index * 2 + turn_index // 2) % len(CITIES)]
            text = (f"I visited {city} and the trip was memorable." if turn_index % 2 == 0
                    else f"That sounds like a good time in {city}.")
            turns.append(SourceTurn(
                stable_id("turn", memory_id, session_id, turn_index), memory_id, session_id,
                turn_index, "Alice" if turn_index % 2 == 0 else "Bob",
                "Bob" if turn_index % 2 == 0 else "Alice",
                "user" if turn_index % 2 == 0 else "assistant", None, text,
                hashlib.sha256(text.encode()).hexdigest()))
    store.ingest_conversation(
        Conversation(memory_id, "golden", memory_id, "memory-hash"), sessions, turns)
    config = _strict_config()
    distiller = QwenSemanticDistiller(
        store, config, "v5.6-determinism",
        client=SimpleNamespace(chat=SimpleNamespace(completions=_CityCompletions())))
    GraphBuildPipeline(store, dataset_hash="v5.6-determinism", distiller=distiller).build(
        memory_id, config)
    return store


def _semantic_store(path: Path) -> SQLiteGraphStore:
    store = _store(path)
    distiller = QwenSemanticDistiller(
        store, _strict_config(), "v5.6-determinism",
        client=SimpleNamespace(chat=SimpleNamespace(completions=StrictCompletions())),
    )
    GraphBuildPipeline(store, dataset_hash="v5.6-determinism", distiller=distiller).build(
        "travel", _strict_config())
    return store


def _turn(turn_id: str, text: str) -> SourceTurn:
    return SourceTurn(turn_id, "m", "s1", 0, "Alice", "Bob", "user", None, text,
                      hashlib.sha256(text.encode()).hexdigest())


def _candidate(turn_id: str, score: float) -> CandidateScore:
    return CandidateScore(turn_id, "s1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, score, ())


def test_pack_emits_mandatory_turns_in_declared_unit_order() -> None:
    """Packed order must follow the proof units, not set-iteration order.

    V5.5 built ``mandatory`` as a ``set[str]`` and then iterated it, so the
    packed turn order -- and under the turn cap, the packed turn *membership* --
    varied with PYTHONHASHSEED.
    """
    turn_ids = [f"turn:{index:02d}" for index in range(12)]
    turns = {turn_id: _turn(turn_id, f"evidence {turn_id}") for turn_id in turn_ids}
    units = (
        EvidenceUnit("unit-b", ("o1",), ("b1",), (turn_ids[7], turn_ids[3]), (), 0, True),
        EvidenceUnit("unit-a", ("o2",), ("b2",), (turn_ids[11], turn_ids[0]), (), 0, True),
    )
    declared = [turn_ids[7], turn_ids[3], turn_ids[11], turn_ids[0]]
    candidates = [_candidate(turn_id, 1.0) for turn_id in reversed(turn_ids)]

    packed, _dropped, _flags = pack(units, candidates, turns, max_turns=32, max_tokens=10_000)

    assert list(packed[:len(declared)]) == declared


def test_pack_is_stable_across_repeated_calls() -> None:
    turn_ids = [f"turn:{index:02d}" for index in range(20)]
    turns = {turn_id: _turn(turn_id, f"evidence {turn_id}") for turn_id in turn_ids}
    units = tuple(
        EvidenceUnit(f"unit-{index}", ("o",), (f"b{index}",), (turn_ids[index],), (), 0, True)
        for index in range(0, 20, 3)
    )
    candidates = [_candidate(turn_id, float(20 - index)) for index, turn_id in enumerate(turn_ids)]

    results = {pack(units, candidates, turns, max_turns=8, max_tokens=10_000)[0]
               for _ in range(25)}

    assert len(results) == 1


RUNNER = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {tests!r})
from graphmem.domain import QueryBudget, dataclass_dict
from graphmem.retrieval import GraphNavigator, HarnessProfile
from test_v5_6_determinism import _city_store

store = _city_store(Path({db!r}))
result = GraphNavigator(store, harness_profile=HarnessProfile.H6_PROOF_PACKING).navigate(
    "travel", "What places did Alice visit?", QueryBudget(max_evidence_turns=32))
payload = {{
    "packed": list(result.packed_turn_ids),
    "dropped": list(result.dropped_turn_ids),
    "units": [dataclass_dict(unit) for unit in result.proof_units],
    "unit_count": len(result.proof_units),
    "certificate": dataclass_dict(result.certificate),
    "seeds": list(result.seed_node_ids),
    "visited": list(result.visited_path_node_ids),
}}
blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
print(json.dumps({{"units": len(result.proof_units),
                   "packed": len(result.packed_turn_ids),
                   "digest": hashlib.sha256(blob.encode()).hexdigest()}}))
store.close()
"""


def _run_under_seed(tmp_path: Path, seed: str) -> dict:
    script = tmp_path / f"runner_{seed}.py"
    script.write_text(RUNNER.format(src=str(SRC_DIR), tests=str(TEST_DIR),
                                    db=str(tmp_path / f"graph_{seed}.sqlite")))
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                               env=environment, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_navigation_is_byte_identical_across_hash_seeds(tmp_path: Path) -> None:
    """The whole H6 path must be reproducible across processes, not just in one."""
    left = _run_under_seed(tmp_path, "0")
    right = _run_under_seed(tmp_path, "1")

    # A single-unit fixture would pass no matter how the packer ordered turns.
    assert left["units"] > 1, "fixture no longer exercises multi-unit packing"
    assert left["packed"] > 1
    assert left["digest"] == right["digest"]

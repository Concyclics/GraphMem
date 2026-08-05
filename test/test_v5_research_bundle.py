from __future__ import annotations

import importlib.util
from pathlib import Path


def load_bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_v5_gate_a_research_bundle.py"
    spec = importlib.util.spec_from_file_location("v5_research_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locomo_evidence_maps_to_zero_based_turn_ids() -> None:
    module = load_bundle_module()
    assert module.parse_locomo_gold("q", ["D4:8; D6:6", "D4:8"]) == [
        "q:session_4:turn:7",
        "q:session_6:turn:5",
    ]


def test_packed_evidence_blocks_preserve_source_boundaries() -> None:
    module = load_bundle_module()
    context = (
        "prefix\n[SOURCE_EVIDENCE q:s1:turn:0]\nfirst evidence\n"
        "[SOURCE_EVIDENCE q:s2:turn:3]\nsecond evidence\n"
    )
    assert module.packed_evidence_blocks(context) == {
        "q:s1:turn:0": "first evidence\n",
        "q:s2:turn:3": "second evidence\n",
    }


def test_span_matching_normalizes_case_and_whitespace() -> None:
    module = load_bundle_module()
    assert module.normalized_text("  Project   ALPHA\nstarted ") == "project alpha started"

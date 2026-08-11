from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_v5_54_report_accuracy_manifest",
    ROOT / "scripts" / "build_v5_54_report_accuracy_manifest.py")
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_report_accuracy_uses_nearest_rank_tokens() -> None:
    stats = REPORT.token_stats(list(range(1, 101)))

    assert stats["count"] == 100
    assert stats["mean"] == 50.5
    assert stats["p95"] == 95
    assert stats["p99"] == 99
    assert stats["max"] == 100
    assert stats["percentile_method"] == "nearest_rank"


def test_report_accuracy_covers_all_published_question_types() -> None:
    assert set(REPORT.TYPE_KEYS.values()) == {
        "single-session-user", "single-session-assistant",
        "single-session-preference", "multi-session",
        "temporal-reasoning", "knowledge-update",
        "category_1", "category_2", "category_3", "category_4",
    }

import json

from scripts import render_v5_20_report_assets as render
from scripts import summarize_v5_20_budget_benchmark as summarize


def test_build_stats_preserves_benchmark_owner_units(tmp_path):
    rows = [
        {
            "memory_id": f"lme-{index}",
            "input_tokens": 10,
            "cached_input_tokens": 1,
            "output_tokens": 5,
            "tokens": 15,
        }
        for index in range(500)
    ] + [
        {
            "memory_id": f"locomo:conv-{index}",
            "input_tokens": 20,
            "cached_input_tokens": 2,
            "output_tokens": 7,
            "tokens": 27,
        }
        for index in range(10)
    ]
    report = tmp_path / "build_report.json"
    report.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    result = summarize.build_stats(report)

    assert result["longmemeval"]["total"] == {
        "count": 500,
        "mean": 15,
        "p95": 15,
        "p99": 15,
        "max": 15,
        "unit": "tokens_per_memory",
        "percentile_method": "nearest_rank",
    }
    assert result["locomo"]["total"]["count"] == 10
    assert result["locomo"]["total"]["mean"] == 27
    assert result["locomo"]["total"]["unit"] == "tokens_per_conversation"


def test_accuracy_by_type_keeps_counts_and_accuracy():
    result = summarize.accuracy_by_type([
        {"question_type": "temporal", "correct": True},
        {"question_type": "temporal", "correct": False},
        {"question_type": "multi-hop", "correct": True},
    ])

    assert result["temporal"] == {
        "questions": 2, "correct": 1, "accuracy": 0.5,
    }
    assert result["multi-hop"] == {
        "questions": 1, "correct": 1, "accuracy": 1.0,
    }


def test_report_rows_render_missing_mem0_type_and_build_as_pending():
    payload = {
        "graphmem": [{
            "benchmark": "longmemeval",
            "retrieval_setting": "32-turn",
            "build_tokens": {
                "count": 500, "mean": 100, "p95": 110,
                "p99": 120, "max": 130,
            },
            "accuracy_by_type": {},
        }],
        "mem0": [{
            "benchmark": "longmemeval",
            "retrieval_setting": "top-50",
            "status": "complete",
            "questions": 500,
            "build_tokens": None,
            "accuracy_by_type": None,
        }],
    }

    assert "运行中" in render.build_rows(payload)
    assert "运行中" in render.type_accuracy_rows(payload)

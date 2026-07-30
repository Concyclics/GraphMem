from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from graphmem_demo.data import load_longmemeval_cases
from graphmem_demo.pipeline import DemoConfig, _memory_cache_fingerprint, run_demo


def _row(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": "What time did I reach the clinic on Monday?",
        "answer": "9:00 AM",
        "question_date": "2026/05/22",
        "answer_session_ids": ["answer-session"],
        "haystack_session_ids": ["distractor-session", "answer-session"],
        "haystack_dates": ["2026/05/20", "2026/05/21"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I practiced guitar after work."},
                {"role": "assistant", "content": "Keep a steady schedule."},
            ],
            [
                {
                    "role": "user",
                    "content": "On Monday I reached the clinic at 9:00 AM.",
                },
                {
                    "role": "assistant",
                    "content": "I will remember the clinic arrival time.",
                },
            ],
        ],
        "locomo_sample_id": "conv-1",
    }


def _run(data_path: Path, output_dir: Path, cache_dir: Path) -> None:
    run_demo(
        DemoConfig(
            data_path=data_path,
            question_type="all",
            output_dir=output_dir,
            memory_cache_dir=cache_dir,
            max_questions=1,
            variants=("hierarchical_hypergraph_v3",),
            mock_services=True,
        )
    )


def test_v3_disk_cache_relabels_build_calls_to_new_split_owner(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "shared-cache"
    first_data = tmp_path / "first.json"
    first_data.write_text(json.dumps([_row("old-owner")]), encoding="utf-8")
    _run(first_data, tmp_path / "first-run", cache_dir)

    second_data = tmp_path / "second.json"
    second_data.write_text(json.dumps([_row("new-owner")]), encoding="utf-8")
    second_output = tmp_path / "second-run"
    _run(second_data, second_output, cache_dir)

    variant_dir = second_output / "hierarchical_hypergraph_v3"
    calls = [
        json.loads(line)
        for line in (variant_dir / "llm_calls.jsonl").read_text().splitlines()
    ]
    build_calls = [call for call in calls if call["stage"].startswith("build_")]
    assert build_calls
    assert {call["question_id"] for call in build_calls} == {"new-owner"}
    stats = json.loads((variant_dir / "question_stats.jsonl").read_text().strip())
    assert stats["question_id"] == "new-owner"
    assert stats["build_total_tokens"] > 0


def test_v3_cache_fingerprint_normalizes_api_base_url_trailing_slash(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps([_row("q")]), encoding="utf-8")
    case = load_longmemeval_cases(data_path, "all", 1)[0]
    config = DemoConfig(
        data_path=data_path,
        output_dir=tmp_path / "out",
        variants=("hierarchical_hypergraph_v3",),
        deepseek_model="gpt-5.4-mini",
        deepseek_base_url="https://example.test/v1",
        embedding_model="Qwen3-Embedding-0.6B",
    )
    assert _memory_cache_fingerprint(
        config, case, "hierarchical_hypergraph_v3"
    ) == _memory_cache_fingerprint(
        replace(config, deepseek_base_url="https://example.test/v1/"),
        case,
        "hierarchical_hypergraph_v3",
    )

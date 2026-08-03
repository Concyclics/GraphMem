from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "summarize_unified_benchmark.py"
    spec = importlib.util.spec_from_file_location("summarize_unified_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_report_usage_and_percentiles() -> None:
    module = _module()
    calls = [{
        "prompt_cache_hit_tokens": 20,
        "prompt_cache_miss_tokens": 80,
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "total_tokens": 105,
        "reasoning_tokens": 0,
        "breakdown_inferred": True,
    }]
    result = module.usage(calls)
    assert result["cached_input_tokens"] == 20
    assert result["uncached_input_tokens"] == 80
    assert result["output_tokens"] == 5
    assert result["accounting_valid"] is True
    assert module.distribution([1, 2, 3, 100])["p95"] == 100


def test_unified_report_accepts_both_judge_contracts() -> None:
    module = _module()
    assert module.judged_correct({"correct": True}) is True
    assert module.judged_correct({"verdict": "yes"}) is True
    assert module.judged_correct({"label": "CORRECT"}) is True
    assert module.judged_correct({"verdict": "no"}) is False


def test_unified_report_excludes_cached_call_replays_by_provider_id() -> None:
    module = _module()
    first = {"model": "qwen", "call_id": "call-1", "total_tokens": 100}
    replay = {**first, "question_id": "replayed-owner"}
    no_id_a = {"model": "qwen", "total_tokens": 20}
    no_id_b = {"model": "qwen", "total_tokens": 20}
    unique, replayed = module.unique_provider_calls(
        [first, replay, no_id_a, no_id_b]
    )
    assert unique == [first, no_id_a, no_id_b]
    assert replayed == 1

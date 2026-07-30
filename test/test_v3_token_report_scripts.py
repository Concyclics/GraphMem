from scripts.summarize_locomo_v2_tokens import (
    _call_usage_valid as locomo_call_usage_valid,
)
from scripts.summarize_locomo_v2_tokens import _matches_stats as locomo_matches_stats
from scripts.summarize_locomo_v2_tokens import _usage as locomo_usage
from scripts.summarize_longmemeval_v3_tokens import (
    _call_usage_valid as lme_call_usage_valid,
)
from scripts.summarize_longmemeval_v3_tokens import _matches_stats as lme_matches_stats
from scripts.summarize_longmemeval_v3_tokens import _usage as lme_usage


def _call(*, hit: int = 20, miss: int = 30, output: int = 5) -> dict:
    return {
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "prompt_tokens": hit + miss,
        "completion_tokens": output,
        "total_tokens": hit + miss + output,
        "reasoning_tokens": 0,
    }


def test_token_report_helpers_reconcile_raw_call_components() -> None:
    calls = [_call(), _call(hit=4, miss=6, output=2)]
    expected = {
        "cache_miss_input_tokens": 36,
        "cache_hit_input_tokens": 24,
        "output_tokens": 7,
        "total_tokens": 67,
        "reasoning_tokens": 0,
        "breakdown_inferred_calls": 0,
    }
    assert lme_usage(calls) == expected
    assert locomo_usage(calls) == expected
    assert all(lme_call_usage_valid(call) for call in calls)
    assert all(locomo_call_usage_valid(call) for call in calls)


def test_token_report_helpers_reject_inconsistent_provider_totals() -> None:
    invalid = _call()
    invalid["prompt_tokens"] += 1
    assert not lme_call_usage_valid(invalid)
    assert not locomo_call_usage_valid(invalid)


def test_token_report_helpers_reject_stats_that_do_not_match_calls() -> None:
    usage = lme_usage([_call()])
    stats = {
        "answer_cache_miss_input_tokens": 30,
        "answer_cache_hit_input_tokens": 20,
        "answer_output_tokens": 5,
        "answer_total_tokens": 55,
    }
    assert lme_matches_stats(stats, "answer", usage)
    assert locomo_matches_stats(stats, "answer", usage)
    stats["answer_output_tokens"] = 6
    assert not lme_matches_stats(stats, "answer", usage)
    assert not locomo_matches_stats(stats, "answer", usage)

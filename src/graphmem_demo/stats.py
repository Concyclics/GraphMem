from __future__ import annotations

from dataclasses import asdict

from .models import DeepSeekCallRecord, QuestionStats, VariantStats

BUILD_BUDGET_TOKENS = 300_000
ANSWER_BUDGET_TOKENS = 10_000


def _phase(records: list[DeepSeekCallRecord], prefix: str) -> list[DeepSeekCallRecord]:
    return [record for record in records if record.stage.startswith(prefix) and not record.excluded_from_budget]


def _valid_record(record: DeepSeekCallRecord) -> bool:
    return (
        record.prompt_cache_hit_tokens + record.prompt_cache_miss_tokens == record.prompt_tokens
        and record.prompt_tokens + record.completion_tokens == record.total_tokens
        and (record.thinking_mode not in {"disabled", "none"} or record.reasoning_tokens == 0)
    )


def build_question_stats(
    *, question_id: str, variant: str, session_count: int, leaf_count: int,
    summary_count: int, edge_count: int, records: list[DeepSeekCallRecord],
    build_latency_sec: float, retrieval_latency_sec: float, answer_latency_sec: float,
    answer_session_hit: bool, answer_session_all_hit: bool = False,
    answer_session_recall: float = 0.0, retrieved_answer_session_count: int = 0,
    gold_answer_session_count: int = 0, wall_time_sec: float = 0.0,
    summary_parse_error_count: int = 0, summary_truncation_count: int = 0,
    ready_job_counts: list[dict] | None = None, peak_inflight_deepseek: int = 0,
    build_budget_tokens: int = BUILD_BUDGET_TOKENS,
    answer_budget_tokens: int = ANSWER_BUDGET_TOKENS,
) -> QuestionStats:
    budget_records = [record for record in records if not record.excluded_from_budget]
    build_records = _phase(records, "build_")
    answer_records = _phase(records, "answer_")
    build_total = _sum(build_records, "total_tokens")
    answer_total = _sum(answer_records, "total_tokens")
    return QuestionStats(
        question_id=question_id, variant=variant, session_count=session_count,
        leaf_count=leaf_count, summary_count=summary_count, edge_count=edge_count,
        build_prompt_tokens=_sum(build_records, "prompt_tokens"),
        build_completion_tokens=_sum(build_records, "completion_tokens"),
        answer_prompt_tokens=_sum(answer_records, "prompt_tokens"),
        answer_completion_tokens=_sum(answer_records, "completion_tokens"),
        reasoning_tokens=_sum(budget_records, "reasoning_tokens"),
        total_deepseek_tokens=_sum(budget_records, "total_tokens"),
        deepseek_call_count=len(budget_records), build_latency_sec=build_latency_sec,
        retrieval_latency_sec=retrieval_latency_sec, answer_latency_sec=answer_latency_sec,
        retrieved_answer_session_hit=answer_session_hit,
        retrieved_answer_session_all_hit=answer_session_all_hit,
        retrieved_answer_session_recall=answer_session_recall,
        retrieved_answer_session_count=retrieved_answer_session_count,
        gold_answer_session_count=gold_answer_session_count, wall_time_sec=wall_time_sec,
        summary_parse_error_count=summary_parse_error_count,
        summary_truncation_count=summary_truncation_count,
        build_calls_per_session=len(build_records) / session_count if session_count else 0.0,
        ready_job_counts=ready_job_counts or [], peak_inflight_deepseek=peak_inflight_deepseek,
        build_cache_miss_input_tokens=_sum(build_records, "prompt_cache_miss_tokens"),
        build_cache_hit_input_tokens=_sum(build_records, "prompt_cache_hit_tokens"),
        build_output_tokens=_sum(build_records, "completion_tokens"), build_total_tokens=build_total,
        answer_cache_miss_input_tokens=_sum(answer_records, "prompt_cache_miss_tokens"),
        answer_cache_hit_input_tokens=_sum(answer_records, "prompt_cache_hit_tokens"),
        answer_output_tokens=_sum(answer_records, "completion_tokens"), answer_total_tokens=answer_total,
        build_budget_pass=build_total <= build_budget_tokens,
        answer_budget_pass=answer_total <= answer_budget_tokens,
        token_accounting_valid=all(_valid_record(record) for record in budget_records),
    )


def aggregate_variant_stats(question_stats: list[QuestionStats], variant: str) -> VariantStats:
    n=len(question_stats); sessions=_sum(question_stats,"session_count"); total=_sum(question_stats,"total_deepseek_tokens")
    over_build=[s.question_id for s in question_stats if not s.build_budget_pass]
    over_answer=[s.question_id for s in question_stats if not s.answer_budget_pass]
    metadata={
        "build_total_tokens_percentiles": _percentiles([s.build_total_tokens for s in question_stats]),
        "answer_total_tokens_percentiles": _percentiles([s.answer_total_tokens for s in question_stats]),
        "token_accounting_valid": all(s.token_accounting_valid for s in question_stats),
        "budgets": {"build_per_question": BUILD_BUDGET_TOKENS, "answer_per_question": ANSWER_BUDGET_TOKENS},
    }
    return VariantStats(
        variant=variant, question_count=n, session_count=sessions,
        leaf_count=_sum(question_stats,"leaf_count"), summary_count=_sum(question_stats,"summary_count"),
        edge_count=_sum(question_stats,"edge_count"), build_prompt_tokens=_sum(question_stats,"build_prompt_tokens"),
        build_completion_tokens=_sum(question_stats,"build_completion_tokens"), answer_prompt_tokens=_sum(question_stats,"answer_prompt_tokens"),
        answer_completion_tokens=_sum(question_stats,"answer_completion_tokens"), reasoning_tokens=_sum(question_stats,"reasoning_tokens"),
        total_deepseek_tokens=total, deepseek_call_count=_sum(question_stats,"deepseek_call_count"),
        avg_tokens_per_question=total/n if n else 0.0, avg_tokens_per_session=total/sessions if sessions else 0.0,
        retrieval_answer_session_hit_rate=sum(s.retrieved_answer_session_hit for s in question_stats)/n if n else 0.0,
        retrieval_answer_session_all_hit_rate=sum(s.retrieved_answer_session_all_hit for s in question_stats)/n if n else 0.0,
        avg_retrieved_answer_session_recall=_sum(question_stats,"retrieved_answer_session_recall")/n if n else 0.0,
        token_budget_avg_under_300k=not over_build and not over_answer,
        build_latency_sec=_sum(question_stats,"build_latency_sec"), retrieval_latency_sec=_sum(question_stats,"retrieval_latency_sec"),
        answer_latency_sec=_sum(question_stats,"answer_latency_sec"), wall_time_sec=_sum(question_stats,"wall_time_sec"),
        summary_parse_error_count=_sum(question_stats,"summary_parse_error_count"), summary_truncation_count=_sum(question_stats,"summary_truncation_count"),
        build_calls_per_session=sum(s.build_calls_per_session*s.session_count for s in question_stats)/sessions if sessions else 0.0,
        peak_inflight_deepseek=max((s.peak_inflight_deepseek for s in question_stats),default=0), metadata=metadata,
        build_cache_miss_input_tokens=_sum(question_stats,"build_cache_miss_input_tokens"), build_cache_hit_input_tokens=_sum(question_stats,"build_cache_hit_input_tokens"),
        build_output_tokens=_sum(question_stats,"build_output_tokens"), build_total_tokens=_sum(question_stats,"build_total_tokens"),
        answer_cache_miss_input_tokens=_sum(question_stats,"answer_cache_miss_input_tokens"), answer_cache_hit_input_tokens=_sum(question_stats,"answer_cache_hit_input_tokens"),
        answer_output_tokens=_sum(question_stats,"answer_output_tokens"), answer_total_tokens=_sum(question_stats,"answer_total_tokens"),
        build_budget_pass_count=n-len(over_build), answer_budget_pass_count=n-len(over_answer),
        build_budget_max_tokens=max((s.build_total_tokens for s in question_stats),default=0), answer_budget_max_tokens=max((s.answer_total_tokens for s in question_stats),default=0),
        over_build_budget_question_ids=over_build, over_answer_budget_question_ids=over_answer,
    )


def build_stats_payload(stats: list[QuestionStats], variant_stats: VariantStats) -> dict:
    return {"variant":variant_stats.variant,"aggregate":asdict(variant_stats),"questions":[asdict(s) for s in stats]}


def query_stats_payload(stats: list[QuestionStats], variant_stats: VariantStats) -> dict:
    return {"variant":variant_stats.variant,"aggregate":asdict(variant_stats),"questions":[asdict(s) for s in stats]}


def _percentiles(values: list[int]) -> dict[str,int]:
    if not values: return {"p50":0,"p95":0,"max":0}
    ordered=sorted(values)
    def pick(q: float) -> int: return ordered[min(len(ordered)-1, max(0, math.ceil(q*len(ordered))-1))]
    import math
    return {"p50":pick(0.50),"p95":pick(0.95),"max":ordered[-1]}


def _sum(values: list, field: str) -> float | int:
    return sum(getattr(value,field) for value in values)

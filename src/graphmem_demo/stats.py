from __future__ import annotations

from dataclasses import asdict

from .models import DeepSeekCallRecord, QuestionStats, VariantStats


BUILD_STAGES = {
    "build_summary_leaf",
    "build_summary_internal",
    "build_summary_raw_group",
    "build_summary_session_direct",
    "build_summary_session_merge",
}


def build_question_stats(
    *,
    question_id: str,
    variant: str,
    session_count: int,
    leaf_count: int,
    summary_count: int,
    edge_count: int,
    records: list[DeepSeekCallRecord],
    build_latency_sec: float,
    retrieval_latency_sec: float,
    answer_latency_sec: float,
    answer_session_hit: bool,
    answer_session_all_hit: bool = False,
    answer_session_recall: float = 0.0,
    retrieved_answer_session_count: int = 0,
    gold_answer_session_count: int = 0,
    wall_time_sec: float = 0.0,
    summary_parse_error_count: int = 0,
    summary_truncation_count: int = 0,
    ready_job_counts: list[dict] | None = None,
    peak_inflight_deepseek: int = 0,
) -> QuestionStats:
    build_records = [record for record in records if record.stage in BUILD_STAGES]
    answer_records = [record for record in records if record.stage == "answer_qa"]
    return QuestionStats(
        question_id=question_id,
        variant=variant,
        session_count=session_count,
        leaf_count=leaf_count,
        summary_count=summary_count,
        edge_count=edge_count,
        build_prompt_tokens=_sum(build_records, "prompt_tokens"),
        build_completion_tokens=_sum(build_records, "completion_tokens"),
        answer_prompt_tokens=_sum(answer_records, "prompt_tokens"),
        answer_completion_tokens=_sum(answer_records, "completion_tokens"),
        reasoning_tokens=_sum(records, "reasoning_tokens"),
        total_deepseek_tokens=_sum(records, "total_tokens"),
        deepseek_call_count=len(records),
        build_latency_sec=build_latency_sec,
        retrieval_latency_sec=retrieval_latency_sec,
        answer_latency_sec=answer_latency_sec,
        retrieved_answer_session_hit=answer_session_hit,
        retrieved_answer_session_all_hit=answer_session_all_hit,
        retrieved_answer_session_recall=answer_session_recall,
        retrieved_answer_session_count=retrieved_answer_session_count,
        gold_answer_session_count=gold_answer_session_count,
        wall_time_sec=wall_time_sec,
        summary_parse_error_count=summary_parse_error_count,
        summary_truncation_count=summary_truncation_count,
        build_calls_per_session=len(build_records) / session_count if session_count else 0.0,
        ready_job_counts=ready_job_counts or [],
        peak_inflight_deepseek=peak_inflight_deepseek,
    )


def aggregate_variant_stats(question_stats: list[QuestionStats], variant: str) -> VariantStats:
    question_count = len(question_stats)
    session_count = _sum(question_stats, "session_count")
    total_tokens = _sum(question_stats, "total_deepseek_tokens")
    return VariantStats(
        variant=variant,
        question_count=question_count,
        session_count=session_count,
        leaf_count=_sum(question_stats, "leaf_count"),
        summary_count=_sum(question_stats, "summary_count"),
        edge_count=_sum(question_stats, "edge_count"),
        build_prompt_tokens=_sum(question_stats, "build_prompt_tokens"),
        build_completion_tokens=_sum(question_stats, "build_completion_tokens"),
        answer_prompt_tokens=_sum(question_stats, "answer_prompt_tokens"),
        answer_completion_tokens=_sum(question_stats, "answer_completion_tokens"),
        reasoning_tokens=_sum(question_stats, "reasoning_tokens"),
        total_deepseek_tokens=total_tokens,
        deepseek_call_count=_sum(question_stats, "deepseek_call_count"),
        avg_tokens_per_question=total_tokens / question_count if question_count else 0.0,
        avg_tokens_per_session=total_tokens / session_count if session_count else 0.0,
        retrieval_answer_session_hit_rate=(
            sum(stat.retrieved_answer_session_hit for stat in question_stats) / question_count
            if question_count
            else 0.0
        ),
        retrieval_answer_session_all_hit_rate=(
            sum(stat.retrieved_answer_session_all_hit for stat in question_stats)
            / question_count
            if question_count
            else 0.0
        ),
        avg_retrieved_answer_session_recall=(
            _sum(question_stats, "retrieved_answer_session_recall") / question_count
            if question_count
            else 0.0
        ),
        token_budget_avg_under_300k=(
            total_tokens / question_count < 300000 if question_count else False
        ),
        build_latency_sec=_sum(question_stats, "build_latency_sec"),
        retrieval_latency_sec=_sum(question_stats, "retrieval_latency_sec"),
        answer_latency_sec=_sum(question_stats, "answer_latency_sec"),
        wall_time_sec=_sum(question_stats, "wall_time_sec"),
        summary_parse_error_count=_sum(question_stats, "summary_parse_error_count"),
        summary_truncation_count=_sum(question_stats, "summary_truncation_count"),
        build_calls_per_session=(
            sum(
                stat.build_calls_per_session * stat.session_count
                for stat in question_stats
            )
            / session_count
            if session_count
            else 0.0
        ),
        peak_inflight_deepseek=max(
            (stat.peak_inflight_deepseek for stat in question_stats), default=0
        ),
    )


def build_stats_payload(stats: list[QuestionStats], variant_stats: VariantStats) -> dict:
    return {
        "variant": variant_stats.variant,
        "aggregate": asdict(variant_stats),
        "questions": [
            {
                "question_id": stat.question_id,
                "build_prompt_tokens": stat.build_prompt_tokens,
                "build_completion_tokens": stat.build_completion_tokens,
                "build_latency_sec": stat.build_latency_sec,
                "wall_time_sec": stat.wall_time_sec,
                "leaf_count": stat.leaf_count,
                "summary_count": stat.summary_count,
                "edge_count": stat.edge_count,
                "build_calls_per_session": stat.build_calls_per_session,
                "summary_parse_error_count": stat.summary_parse_error_count,
                "summary_truncation_count": stat.summary_truncation_count,
                "ready_job_counts": stat.ready_job_counts,
                "peak_inflight_deepseek": stat.peak_inflight_deepseek,
                "total_deepseek_tokens": stat.total_deepseek_tokens,
            }
            for stat in stats
        ],
    }


def query_stats_payload(stats: list[QuestionStats], variant_stats: VariantStats) -> dict:
    return {
        "variant": variant_stats.variant,
        "aggregate": asdict(variant_stats),
        "questions": [
            {
                "question_id": stat.question_id,
                "answer_prompt_tokens": stat.answer_prompt_tokens,
                "answer_completion_tokens": stat.answer_completion_tokens,
                "reasoning_tokens": stat.reasoning_tokens,
                "retrieval_latency_sec": stat.retrieval_latency_sec,
                "answer_latency_sec": stat.answer_latency_sec,
                "wall_time_sec": stat.wall_time_sec,
                "retrieved_answer_session_hit": stat.retrieved_answer_session_hit,
                "retrieved_answer_session_any_hit": stat.retrieved_answer_session_hit,
                "retrieved_answer_session_all_hit": stat.retrieved_answer_session_all_hit,
                "retrieved_answer_session_recall": stat.retrieved_answer_session_recall,
                "retrieved_answer_session_count": stat.retrieved_answer_session_count,
                "gold_answer_session_count": stat.gold_answer_session_count,
            }
            for stat in stats
        ],
    }


def _sum(values: list, field: str) -> float | int:
    return sum(getattr(value, field) for value in values)

"""Accepted Astra v3: all source selection precedes the single reader call."""

from datetime import datetime
import re
from .readers import astra_stage2_shared as base
from .readers import astra_context_branches_shared as locomo
from .readers import astra_lme_branches_shared as lme
from .readers import astra_lme_elapsed_shared as elapsed
from .readers import astra_lme_query_sources_shared as reader
from .readers import question_source_completion as extra

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = re.compile(r"\bon\s+(\d{1,2})\s+(" + MONTHS + r"),?\s+(20\d{2})\b", re.I)


def dates(question):
    result = set()
    for day, month, year in DATE.findall(question):
        try:
            result.add(
                datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                .date()
                .isoformat()
            )
        except ValueError:
            continue
    return sorted(result)


def record_date(timestamp):
    try:
        return (
            datetime.strptime(timestamp.partition(" on ")[2], "%d %B, %Y")
            .date()
            .isoformat()
        )
    except ValueError:
        return None


def prepare(benchmark, question, question_date, turns, selected_ids, scores, tokenizer):
    ids = set(selected_ids)
    initial = set(ids)
    if benchmark == "locomo":
        targets = dates(question)
        if targets:
            ids.update(
                t["turn_id"] for t in turns if record_date(t["timestamp"]) in targets
            )
        branch = locomo.route(question, ids != initial)
        ordered = sorted(
            turns,
            key=lambda t: (int(t["session_id"].rsplit("_", 1)[1]), t["turn_index"]),
        )
        records = [
            {
                **t,
                "id": f"S{int(t['session_id'].rsplit('_', 1)[1])}T{t['turn_index'] + 1}",
                "session": int(t["session_id"].rsplit("_", 1)[1]),
                "order": i,
            }
            for i, t in enumerate(ordered)
            if t["turn_id"] in ids
        ]
        context = base.render_locomo(question, question_date, records)
        instruction = locomo.instruction(branch) if branch else base.INSTRUCTION
    else:
        if extra.route(question):
            ids.update(extra.supplemental_ids(question, turns, scores, tokenizer))
        branch = elapsed.route(question) or lme.route(question)
        if lme.route(question) == "topic_count_sources":
            topics = lme.topic_names(question)
            sessions = {
                t["session_id"]
                for t in turns
                if t["speaker"] == "user"
                and any(lme.SOURCE_TOPICS[k].search(t["raw_text"]) for k in topics)
            }
            ids.update(
                t["turn_id"]
                for t in turns
                if t["session_id"] in sessions and t["speaker"] == "user"
            )
        records = [t for t in turns if t["turn_id"] in ids]
        context = base.render_lme(question, question_date, records)
        instruction = reader.instruction(branch)
    if len(records) != len(ids):
        raise ValueError("Selected source is absent from the memory")
    row = dict(
        question=question,
        question_date=question_date,
        context=context,
        branch=branch,
        source_ids=sorted(ids),
        source_text_sha256=base.digest(
            sorted((t["turn_id"], t["raw_text"]) for t in records)
        ),
    )
    body = base.request_body(row)
    body["instructions"] = instruction
    row["payload"] = body
    row["request_sha256"] = base.digest(body)
    return row

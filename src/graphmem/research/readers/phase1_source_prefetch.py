"""Question-only operands for the existing direct Qwen prompt overrides.

Other queries retain the original, fully charged legacy reader/reread path.
This supplies the same age operands before the direct quantity reader without
needing an uncharged older model answer to request them.
"""

from .question_source_completion import route, supplemental_ids


def prefetch(question, all_turns, base_source_ids, graph_scores, tokenizer):
    result = set(base_source_ids)
    if route(question) == "age_since_education":
        result.update(supplemental_ids(question, all_turns, graph_scores, tokenizer))
    return result

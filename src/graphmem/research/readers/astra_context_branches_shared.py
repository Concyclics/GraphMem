"""Question-only reader branches, with no labels or reference answers."""

import re
from .astra_stage2_shared import INSTRUCTION, request_body

INFERENCE = re.compile(
    r"\b(?:might|could|likely|potentially|suspected|possible)\b|^how old is\b", re.I
)
EPISODES = re.compile(
    r"\b(?:recently|hobby|hobbies|activity|activities|games?|play(?:ed|ing)?|share(?:d|s)?|wants?|motivat\w*|languages?|significant event|health issue|consoles?|degree|careers?)\b",
    re.I,
)
EXCLUDE = re.compile(
    r"\b(?:first|last|latest|earliest|most recent|how many|total|number of|count)\b|^(?:what|which) (?:day|week|month|year|date|time)\b",
    re.I,
)

GUIDANCE = {
    "grounded_inference": """This question calls for a supported inference. First look for statements that explicitly connect the requested attribute to its reason or context; those are stronger evidence than a loose association with a job, interest, or one recent event. Combine all relevant clues and media captions. Use ordinary background knowledge to interpret recognizable products, places, or situations when the clues support it. Give the best-supported inference with appropriate uncertainty instead of requiring a verbatim statement of the conclusion. If several possibilities are independently supported, mention up to three; do not generate a generic list of guesses. Distinguish an enduring attribute from a temporary setback or state. If no conclusion is supported, say what remains unknown.""",
    "matching_episodes": """Identify the exact relationship or event the question asks about, and inspect all matching records, including media captions and the adjacent conversation. Do not default to the newest mention or the first plausible example. If several recorded items or episodes directly satisfy a question that does not specify a unique date or event, give a concise set of the supported answers instead of arbitrarily choosing one. For a visit, activity, hobby, possession, shared item, or motivation, preserve the different relevant items; a singular noun alone does not establish that there was only one. Keep the requested person and event fixed, and exclude merely related topics, suggestions not acted on, and unsupported guesses.""",
    "explicit_date_closure": "",
}


def route(question, date_expansion_available):
    if date_expansion_available:
        return "explicit_date_closure"
    if INFERENCE.search(question):
        return "grounded_inference"
    if (
        re.match(r"^(?:what|which|where|who|how does|how do|how did)\b", question, re.I)
        and EPISODES.search(question)
        and not EXCLUDE.search(question)
    ):
        return "matching_episodes"
    return None


def instruction(branch):
    suffix = GUIDANCE[branch]
    return INSTRUCTION + (
        "\n\nQUESTION-SPECIFIC READING GUIDANCE:\n" + suffix if suffix else ""
    )


def payload(row):
    body = request_body(row)
    body["instructions"] = instruction(row["branch"])
    return body

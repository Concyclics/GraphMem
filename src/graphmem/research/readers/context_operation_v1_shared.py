"""Question-specific interpretation of remembered descriptions and operands."""

import re
from .branch_prompt_v1_shared import BASE
from .branch_prompt_v2_shared import route as accepted_route
from .budget_agreement_shared import ROLE
from .contract_reader_shared import ASSISTANT
from .projected_chat_route_shared import PREVIOUS_CHAT

MENTION = re.compile(r"\b(?:I|we) (?:mentioned|told|said)\b", re.I)
RELATIVE = re.compile(r"\b(?:ago|last|yesterday)\b", re.I)
PAIR = re.compile(r"\bfirst\b.*\bor\b", re.I)
AGE = re.compile(r"\bhow old\b|\bhow (?:many years|much) (?:older|younger)\b", re.I)
REWARD = re.compile(
    r"^\s*how many\b.*\b(?:points|stars)\b.*\b(?:need|earn|redeem|reach)\b", re.I
)
DIFFERENCE = re.compile(
    r"\bhow much\b.*\b(?:more|save|saved|faster|earlier|lost)\b|\b(?:what|how much) is the difference\b",
    re.I,
)


def subtype(question):
    if AGE.search(question):
        return "ages"
    if REWARD.search(question):
        return "reward_balance"
    return "stated_difference"


def route(question):
    if (
        ROLE.search(question)
        or ASSISTANT.search(question)
        or PREVIOUS_CHAT.search(question)
    ):
        return None
    # Age comparison is a quantity operation even when phrased as "when".
    if AGE.search(question):
        return "typed_difference"
    if accepted_route(question):
        return None
    if MENTION.search(question) and RELATIVE.search(question):
        return "remembered_time_phrase"
    if PAIR.search(question):
        return "two_event_order"
    if REWARD.search(question) or DIFFERENCE.search(question):
        return "typed_difference"
    return None


GUIDANCE = {
    "remembered_time_phrase": """The question recalls a previous statement containing a relative time description. Find the original user's description of the named event and the quoted relative wording together, then answer the requested event, place, purchase or companion. The words "I mentioned ... two weeks ago" can identify the event as described in that original statement; do not reject that matching statement just because its own date is not exactly two weeks before the global question date. Resolve any event time from the statement's own reference point. Distinguish separate similar events and keep the subject and companion of the matching statement. Quote the decisive original phrase before answering.""",
    "two_event_order": """This is a comparison of TWO specified events. In <work>, make two separate entries: event A and its exact supporting statement/time; event B and its exact supporting statement/time. Match each named person, object, action and completion state. A related object or different action does not establish the requested event. If either event is unestablished, say which is missing: evidence for one event alone cannot establish that it happened first. Otherwise compare the events' actual times, resolving relative expressions at their own source dates; the earlier document or a later recollection is not automatically the earlier event. Use ordinary implications of a supported dated statement, without inventing an extra prerequisite.""",
    "ages": """Treat ages as quantities, distinct from calendar years. First quote the person's current stated age and their age at the requested event, or the two people's stated ages. Look for hyphenated age descriptions as well as direct birthday statements. For "how many years older now than when ...", subtract the age at that event from the current age; do not require a graduation year when both ages are available. For someone's age when a younger person was born, use their age difference. Do not subtract an age from the current calendar year and then treat the resulting birth year as an age. For a future age, add only the supported elapsed time. Show the two typed operands and calculation, then answer.""",
    "reward_balance": """Separate the latest points/stars ALREADY OWNED from the REDEMPTION THRESHOLD. A question asking how many points to earn or to reach a reward concerns the additional amount given the existing balance, unless it explicitly asks for the total threshold itself. Quote the latest actual balance and the threshold. Compute threshold minus balance and return that additional amount; do not return the threshold alone. Planned purchases do not count as already-earned points, and a later total replaces an earlier total. If an operand is absent, state which one is missing.""",
    "stated_difference": """Recall the comparison using the amounts actually stated for the user's scenario. Quote each requested operand, its subject and unit. Give priority to the user's reported amounts or assumptions over a generic price, travel estimate or recommendation in an assistant reply. This is not a request to revise the scenario using typical market prices, exchange rates or unmentioned extra fees. Match the two named alternatives exactly; the value for a different service, location or item cannot replace a missing operand. Compute the requested signed difference, savings or improvement from the two supported values, and keep its units. If either necessary value is unavailable, state what is missing.""",
}


def system(branch, question):
    key = subtype(question) if branch == "typed_difference" else branch
    return BASE + "\n\nQUESTION-SPECIFIC OPERATION:\n" + GUIDANCE[key]

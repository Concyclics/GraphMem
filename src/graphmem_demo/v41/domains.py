from __future__ import annotations

import re
from collections import defaultdict

from ..v36.schema import QueryIR
from .schema import QueryAugmentationV41


# Domain-level vocabulary only.  Entries describe reusable event ontologies and
# never benchmark instances, brands, IDs, or expected answers.
_DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "profile_relationship": (
        "family", "friend", "partner", "colleague", "relative", "parent",
        "child", "sibling", "age", "birthday", "lives", "works",
        "relationship", "single", "married", "dating", "engaged", "divorced",
    ),
    "preference_recommendation": (
        "prefer", "favorite", "like", "love", "dislike", "hate", "avoid",
        "recommend", "suggest", "interested", "enjoy",
    ),
    "health_device": (
        "health", "medical", "doctor", "appointment", "symptom", "diagnosis",
        "medicine", "treatment", "device", "monitor", "therapy",
    ),
    "transaction_inventory": (
        "buy", "bought", "purchase", "paid", "cost", "price", "sold",
        "received", "returned", "replace", "inventory", "own", "item",
    ),
    "travel_location": (
        "travel", "trip", "flight", "train", "drive", "commute", "arrive",
        "depart", "visit", "hotel", "city", "country", "location",
    ),
    "work_education_project": (
        "work", "job", "company", "role", "project", "school", "college",
        "course", "degree", "study", "graduate", "assignment",
    ),
    "creative_media": (
        "book", "read", "write", "music", "film", "movie", "movies",
        "watch", "watched", "saw", "seen", "show", "game", "art",
        "photo", "video", "episode", "song", "create",
    ),
    "social_temporal_event": (
        "event", "party", "meeting", "concert", "festival", "attend",
        "volunteer", "competition", "date", "before", "after", "when",
    ),
}

_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "family": ("relative", "relationship"),
    "buy": ("bought", "purchase", "acquire", "received"),
    "purchase": ("buy", "bought", "acquire", "paid"),
    "work": ("job", "role", "company", "employer"),
    "travel": ("trip", "visit", "journey"),
    "arrive": ("arrival", "reach", "reached"),
    "prefer": ("preference", "favorite", "like", "avoid", "dislike"),
    "recommend": ("suggest", "advice", "preference", "constraint"),
    "event": ("attend", "participate", "happen", "occur", "workshop", "lecture", "tour", "exhibition"),
    "current": ("latest", "now", "updated", "state"),
    "previous": ("earlier", "before", "prior", "old"),
    "symbolize": ("symbol", "mean", "meaning", "represent", "stand for", "signify"),
    "symbol": ("mean", "meaning", "represent", "stand for", "signify"),
    "represent": ("symbol", "symbolize", "mean", "stand for", "signify"),
    "course": ("class", "workshop", "training"),
    "drawing": ("draw", "drew", "sketch", "illustration"),
    "painting": ("paint", "painted", "artwork"),
    "attend": ("attended", "join", "joined", "participate", "participated", "visit", "visited"),
    "visit": ("visited", "went", "traveled", "trip", "outing", "beach", "city", "location"),
    "visited": ("visit", "went", "travel", "traveled", "trip", "outing", "beach", "city", "location", "attend", "attended"),
    "view": ("viewed", "saw", "seen", "tour", "toured", "inspect", "considered", "house hunting", "offer"),
    "viewed": ("view", "saw", "seen", "tour", "toured", "inspect", "considered", "house hunting", "offer"),
    "purchased": ("purchase", "buy", "bought", "acquire", "acquired", "got", "owned"),
    "downloaded": ("download", "acquire", "acquired", "got", "owned"),
    "have": ("had", "own", "owned", "possess", "received", "got"),
    "own": ("owned", "have", "had", "possess", "bought", "received"),
    "make": ("made", "create", "created", "build", "built"),
    "create": ("created", "make", "made", "draw", "paint", "write"),
    "job": ("work", "role", "occupation", "profession", "career"),
    "live": ("lives", "lived", "reside", "resided", "moved", "location"),
    "child": ("children", "kid", "kids", "daughter", "son"),
    "help": ("helped", "support", "supported", "assist", "mentor"),
    "instrument": ("instruments", "guitar", "piano", "keyboard", "drum", "ukulele", "violin", "cello", "saxophone", "trumpet"),
    "instruments": ("instrument", "guitar", "piano", "keyboard", "drum", "ukulele", "violin", "cello", "saxophone", "trumpet"),
    "model": ("models", "diorama", "scale", "miniature"),
    "type": ("kind", "style", "category"),
    "kind": ("type", "style", "category"),
    "sibling": ("siblings", "brother", "brothers", "sister", "sisters"),
    "siblings": ("sibling", "brother", "brothers", "sister", "sisters"),
    "acquire": ("acquired", "buy", "bought", "got", "obtain", "received"),
    "acquired": ("acquire", "buy", "bought", "got", "obtain", "received"),
    "bake": ("baked", "cook", "cooked", "make", "made"),
    "baked": ("bake", "cook", "cooked", "make", "made"),
    "album": ("albums", "ep", "eps", "record", "records", "vinyl"),
    "albums": ("album", "ep", "eps", "record", "records", "vinyl"),
    "doctor": ("physician", "specialist", "clinician", "dermatologist", "appointment", "diagnosed", "prescribed", "biopsy"),
    "doctors": ("doctor", "physician", "specialist", "clinician", "dermatologist", "appointment", "diagnosed", "prescribed", "biopsy"),
    "property": ("properties", "home", "house", "condo", "townhouse", "apartment", "bungalow", "real-estate"),
    "properties": ("property", "home", "house", "condo", "townhouse", "apartment", "bungalow", "real-estate"),
    "party": ("parties", "gathering", "feast", "potluck", "barbecue", "bbq"),
    "parties": ("party", "gathering", "feast", "potluck", "barbecue", "bbq"),
    "delivery": ("deliver", "delivered", "takeout", "take-away", "order", "ordered", "meal"),
    "museum": ("museums", "gallery", "galleries", "exhibition", "exhibit"),
    "museums": ("museum", "gallery", "galleries", "exhibition", "exhibit"),
    "arrive": ("arrived", "reach", "reached", "got"),
    "reached": ("arrive", "arrived", "reach", "got"),
    "relocation": ("relocate", "relocated", "move", "moved"),
    "relocated": ("relocation", "relocate", "move", "moved"),
    "occupation": ("job", "role", "career", "profession", "specialist"),
    "breed": ("dog", "canine", "retriever", "terrier", "spaniel"),
    "homegrown": ("garden", "grown", "harvested", "produce"),
    "ingredients": ("ingredient", "produce", "herb", "herbs"),
    "allocated": ("allocation", "budget", "funding", "spent", "amount"),
    "worth": ("value", "valued", "valuation", "multiple", "multiplier"),
    "paid": ("price", "cost", "purchase"),
    "battery": ("charge", "charging", "power", "powerbank", "charger"),
    "phone": ("mobile", "device", "smartphone"),
    "publication": ("publications", "paper", "papers", "article", "articles", "research"),
    "publications": ("publication", "paper", "papers", "article", "articles", "research"),
    "conference": ("conferences", "symposium", "workshop", "seminar"),
    "conferences": ("conference", "symposium", "workshop", "seminar"),
    "furniture": ("dresser", "decor", "layout", "placement", "room"),
    "rearrange": ("arrange", "arrangement", "layout", "move", "placement"),
    "relationship": ("single", "single parent", "married", "dating", "engaged", "partner", "spouse", "husband", "wife", "boyfriend", "girlfriend", "divorced", "widowed", "breakup", "broke up"),
    "nickname": ("nicknamed", "called", "calls", "known as", "pet name"),
    "reaction": ("reacted", "felt", "feeling", "happy", "thankful", "excited", "surprised", "scared"),
    "inspired": ("inspiration", "motivated", "based on", "idea", "influenced"),
    "inspiration": ("inspired", "motivated", "based on", "idea", "influenced"),
    "advice": ("advised", "suggest", "suggested", "recommend", "recommended", "tip", "practice", "remember"),
    "achievement": ("achieved", "accomplishment", "milestone", "won", "award", "endorsement", "deal", "record"),
    "frustration": ("frustrated", "annoying", "problem", "misplace", "misplaced", "lose", "lost"),
    "digestive": ("digestion", "stomach", "gastritis", "reflux", "abdominal", "nausea"),
    "injury": ("injured", "hurt", "doctor", "serious", "sprain", "ankle"),
    "martial": ("kickboxing", "taekwondo", "karate", "boxing", "combat", "training"),
    "transformation": ("changed", "appearance", "hair", "dyed", "colored", "cut", "style"),
    "photo": ("picture", "image", "media", "shared", "showed", "caption"),
    "picture": ("photo", "image", "media", "shared", "showed", "caption"),
    "choose": ("chose", "picked", "selected", "reason", "because"),
    "chose": ("choose", "picked", "selected", "reason", "because"),
    "plan": ("plans", "planning", "planned", "intend", "intends", "want", "thinking"),
    "plans": ("plan", "planning", "planned", "intend", "intends", "want", "thinking"),
    "project": ("assignment", "working on", "develop", "developing", "build", "building", "course"),
    "health": ("medical", "condition", "symptom", "weight", "overweight", "obesity", "fitness", "exercise", "running", "diet", "doctor"),
    "suspected": ("possible", "likely", "condition", "symptom", "risk", "sign"),
    "compare": ("compared", "comparison", "like", "similar", "analogy", "metaphor", "reminds"),
    "compared": ("compare", "comparison", "like", "similar", "analogy", "metaphor", "reminds"),
    "comparison": ("compare", "compared", "like", "similar", "analogy", "metaphor", "reminds"),
    "movies": ("movie", "film", "films", "watch", "watched", "saw", "seen", "viewed"),
    "movie": ("movies", "film", "films", "watch", "watched", "saw", "seen", "viewed"),
    "seen": ("saw", "watch", "watched", "viewed"),
    "watch": ("watched", "saw", "seen", "viewed"),
    "promote": ("promoted", "promotion", "advertise", "advertised", "campaign", "marketing", "collaborate", "presentation"),
    "promoted": ("promote", "promotion", "advertise", "advertised", "campaign", "marketing", "collaborate", "presentation"),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'_-]*", text.casefold())


def is_inferential_question(text: str) -> bool:
    """Recognize a reusable one-hop profile/recommendation inference request."""
    return bool(re.search(
        r"\b(?:likely|probably|possibly|potentially|might|may|could|would|"
        r"suspected|plausibly)\b|"
        r"\b(?:good|suitable|appropriate)\s+(?:career|job|hobby|activity|"
        r"book|exercise|option|choice)\b|"
        r"\bbenefit\s+from\b|\bbe\s+considered\b|"
        r"\b(?:does|do)\s+[A-Z][\w'’-]+\s+live\s+(?:close|closer|near)\s+to\b|"
        r"\bwhat\s+(?:other\s+)?(?:exercise|activity|career|job|occupation|"
        r"hobby|book|game|place|location)s?\b.{0,80}\b(?:can|could|would|"
        r"might|may|help|suit|fit)\b|"
        r"\bhow\s+old\s+(?:is|are|was|were)\b",
        text, re.IGNORECASE,
    ))


def augment_query(ir: QueryIR) -> QueryAugmentationV41:
    tokens = _tokens(ir.raw_question)
    token_set = set(tokens)
    domains = [
        name for name, terms in _DOMAIN_TERMS.items()
        if token_set.intersection(terms)
    ]
    expanded: list[str] = []
    possession_intent = bool(re.search(
        r"\b(?:own|owned|possess|possessed)\b|"
        r"\b(?:does|do|did)\s+[^?]{0,50}\bhave\b",
        ir.raw_question, re.IGNORECASE,
    ))
    for token in tokens:
        # In "movies have both people seen", have is an auxiliary, not an
        # ownership relation.  Ownership aliases otherwise swamp media recall.
        if token in {"have", "has"} and not possession_intent:
            continue
        expanded.extend(_EXPANSIONS.get(token, ()))

    # Contextual relation families avoid making broad words such as state or
    # status global aliases while still connecting reusable scene paraphrases.
    if {"relationship", "status"} <= token_set:
        expanded.extend((
            "single", "single parent", "married", "dating", "partner",
            "spouse", "husband", "wife", "divorced", "breakup",
        ))
    if token_set.intersection({"why", "reason"}):
        expanded.extend(("because", "reason", "chose", "decided", "purpose"))
    if re.search(r"\bhow\s+long\b", ir.raw_question, re.IGNORECASE):
        expanded.extend(("duration", "for", "since", "started", "began", "finished", "took"))
    if ir.requested_value_type == "location":
        expanded.extend((
            "location", "city", "state", "country", "in", "at",
            "visited", "went", "traveled", "stayed", "lived",
        ))

    value = ir.requested_value_type
    inferential = is_inferential_question(ir.raw_question)
    scalar_type_lookup = bool(re.search(
        r"\bwhat\s+(?:type|kind)\s+of\b", ir.raw_question, re.IGNORECASE,
    ))
    scalar_threshold_lookup = bool(re.search(
        r"\bhow\s+(?:many|much)\b.{0,80}"
        r"\b(?:need(?:ed)?|required?)\b.{0,60}"
        r"\b(?:reach|achieve|attain|qualify|unlock|redeem)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    current_scalar_metric = bool(
        value == "count"
        and token_set.intersection({
            "follower", "followers", "subscriber", "subscribers",
            "points", "stars", "views", "balance", "score",
        })
        and token_set.intersection({"now", "current", "currently", "latest"})
        and "different" not in token_set
    )
    reference_identity = bool(
        value == "span"
        and re.match(
            r"\s*(?:what|which)\s+(?:is|was|are|were)\s+"
            r"(?:the|a|an)\s+(?:[a-z0-9'-]+\s+){0,3}"
            r"(?:game|activity|sport|exercise|device|tool|instrument|object|"
            r"item|book|novel|movie|film|song|place|location|vehicle|animal|"
            r"food|dish|plant|organization|company|method|technique)\s+"
            r"(?:with|where)\b",
            ir.raw_question.casefold(),
        )
    )
    if reference_identity:
        algebra = "reference_identity"
    elif inferential and value not in {
        "count", "list", "aggregate", "date", "duration", "temporal_order",
    }:
        algebra = "inferential_profile"
    elif value == "list" and scalar_type_lookup:
        # "What type/kind of X ...?" requests a scalar semantic slot, not an
        # exhaustive collection.  Treating it as a set creates false scope gaps.
        algebra = "direct_fact"
    elif value == "count" and scalar_threshold_lookup:
        # This asks for one threshold value, not the cardinality of a set.
        algebra = "direct_fact"
    elif current_scalar_metric:
        # A current value such as followers, points, balance, or score is a
        # versioned scalar state. It is not an open-world member collection.
        algebra = "state_update"
    elif value in {"count", "list", "aggregate"}:
        algebra = "collection"
    elif value in {"date", "duration", "temporal_order"}:
        explicit_temporal_comparison = bool(
            value == "temporal_order"
            or (
                value != "date"
                and len(ir.comparison_targets) > 1
            )
            or re.search(
                r"\b(?:time|duration|days?|weeks?|months?|years?)\s+between\b"
                r"|\bhow\s+(?:long|many\s+\w+)\s+(?:after|before)\b"
                r"|\b(?:which|what)\b.{0,80}\b(?:first|earlier|later|last)\b"
                r"|\b(?:difference|elapsed)\b",
                ir.raw_question,
                re.IGNORECASE,
            )
        )
        algebra = (
            "temporal_comparison"
            if explicit_temporal_comparison else "temporal_lookup"
        )
    elif value == "state":
        algebra = "state_update"
    elif value in {"preference", "recommendation"}:
        algebra = "preference_recommendation"
    elif value == "span":
        algebra = "dialogue_lookup"
    elif len(ir.required_roles) > 1 or len(ir.comparison_targets) > 1:
        algebra = "multi_hop"
    else:
        algebra = "direct_fact"

    state_roles = (
        ("owner", "attribute", "old_state", "new_state", "time", "source")
        if token_set.intersection({
            "changed", "change", "updated", "previous", "previously",
            "earlier", "from", "increase", "decrease",
        })
        else ("owner", "attribute", "new_state", "source")
    )
    role_map: dict[str, tuple[str, ...]] = {
        "collection": ("scope", "member", "lifecycle", "source"),
        "temporal_lookup": ("event", "time", "source"),
        "temporal_comparison": ("event_a", "event_b", "time_a", "time_b", "source"),
        "state_update": state_roles,
        "preference_recommendation": ("owner", "polarity", "context", "source"),
        "dialogue_lookup": ("request", "reply", "reply_content", "source"),
        "multi_hop": ("entity", "relation", "support", "source"),
        "inferential_profile": ("owner", "profile_fact", "support", "source"),
        "reference_identity": ("reference", "identity", "source"),
        "direct_fact": ("entity", "relation", "value", "source"),
    }
    required = list(dict.fromkeys([*ir.required_roles, *role_map[algebra]]))
    alternatives = list(dict.fromkeys([
        *ir.target_entities,
        *ir.comparison_targets,
    ]))
    event_terms = [
        token for token in tokens
        if token not in {
            "what", "which", "who", "when", "where", "why", "how", "many",
            "much", "did", "does", "do", "was", "were", "is", "are", "the",
            "a", "an", "i", "my", "me",
        }
    ]
    scope_terms = list(dict.fromkeys([
        *ir.collection_constraints,
        *ir.state_constraints,
        *ir.temporal_constraints,
    ]))
    return QueryAugmentationV41(
        domain_hints=domains,
        required_roles=required,
        alternative_entities=alternatives[:24],
        event_identity_terms=event_terms[:24],
        scope_terms=scope_terms[:16],
        answer_algebra=algebra,
        expanded_terms=list(dict.fromkeys(expanded))[:32],
        planner_required=(
            algebra in {
                "collection", "temporal_comparison", "state_update",
                "dialogue_lookup", "multi_hop", "inferential_profile",
                "reference_identity",
            }
            or (algebra == "temporal_lookup" and value == "duration")
        ),
    )


def domain_terms(augmentation: QueryAugmentationV41) -> list[str]:
    values: list[str] = []
    for domain in augmentation.domain_hints:
        values.extend(_DOMAIN_TERMS.get(domain, ()))
    return list(dict.fromkeys(values))


def domain_statistics(questions: list[str]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for question in questions:
        tokens = set(_tokens(question))
        for name, terms in _DOMAIN_TERMS.items():
            counts[name] += int(bool(tokens.intersection(terms)))
    return dict(counts)

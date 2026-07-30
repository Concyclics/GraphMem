from graphmem_demo.v3.query_focus import (
    focused_evidence_capsule,
    infer_answer_slot,
    should_use_focused_capsule,
)


def test_answer_slots_are_linguistic_not_topic_specific() -> None:
    assert infer_answer_slot("Where did Mira hold the seminar?").kind == "location"
    assert infer_answer_slot("When did Omar resume the course?").kind == "time"
    assert infer_answer_slot("How did Inez feel after the review?").kind == "emotion"
    assert infer_answer_slot("What kind of workshop did Kai host?").kind == "category"
    assert infer_answer_slot("What does the routine make him?").kind == "effect"
    assert infer_answer_slot("How long did both repairs take combined?").kind == "duration"


def test_focus_gate_preserves_open_ended_and_exhaustive_requests() -> None:
    assert not should_use_focused_capsule(
        answer_form="recommendation", operation="recommendation"
    )
    assert not should_use_focused_capsule(answer_form="list", operation="list")
    assert should_use_focused_capsule(answer_form="span", operation="lookup")


def test_focus_contraction_prefers_relation_complete_evidence() -> None:
    context = """[TURN s1:t1 | speaker=Kai]
Kai bought a desk with a microphone for his office.

[CLAIM s2:c1 | sources=s2:t2]
Kai | hosted | a safety workshop where he taught emergency procedures

[TURN s3:t1 | speaker=Kai]
Kai likes teaching and recently replaced his headphones.

[TURN s2:t2 | speaker=Kai]
I hosted my own safety workshop and taught emergency procedures."""
    capsule = focused_evidence_capsule(
        "What type of workshop did Kai host where he taught emergency procedures?",
        context,
        max_blocks=2,
    )
    assert "safety workshop" in capsule
    assert "headphones" not in capsule


def test_focus_contraction_keeps_exact_reaction_over_time_distractor() -> None:
    context = """[EVENT e1]
Inez received a review last week.

[TURN s1:t2 | speaker=Inez]
Someone reviewed my essay last week. Their words moved me and I felt deeply touched.

[TURN s2:t8 | speaker=Inez]
Yesterday I bought a calendar for my desk."""
    capsule = focused_evidence_capsule(
        "How did Inez feel when someone reviewed her essay?",
        context,
        max_blocks=2,
    )
    assert "deeply touched" in capsule
    assert "calendar" not in capsule


def test_focus_contraction_follows_typed_provenance_to_pronoun_source() -> None:
    context = """[EVENT_FRAME frame:1 | status=planned | sources=s:turn:2]
Mira plans to prepare the promised recipe

[TURN s:turn:2 | speaker=Mira]
I will make it for my family this weekend.

[TURN other:turn:8 | speaker=Mira]
I discussed several unrelated recipes."""
    capsule = focused_evidence_capsule(
        "What did Mira plan to do with the promised recipe?",
        context,
        max_blocks=2,
    )
    assert "prepare the promised recipe" in capsule
    assert "make it for my family" in capsule


def test_episode_opening_sources_do_not_displace_fine_claim_source() -> None:
    context = """[EPISODE episode:1 | sources=s:turn:0,s:turn:1]
Mira discusses a promised recipe and a family plan

[TURN s:turn:0 | speaker=Lee]
Hello, how are you?

[CLAIM claim:1 | modality=planned | sources=s:turn:5]
Mira | planned | prepare the promised recipe

[TURN s:turn:5 | speaker=Mira]
I will make it for my family this weekend."""
    capsule = focused_evidence_capsule(
        "What did Mira plan for the promised recipe?",
        context,
        max_blocks=3,
    )
    assert "make it for my family" in capsule
    assert "Hello, how are you" not in capsule
    assert capsule.rfind("[TURN") > capsule.rfind("[CLAIM")


def test_focus_contraction_matches_recommendation_relation_nouns_and_verbs() -> None:
    context = """[TURN s1:t1 | speaker=Melanie]
I loved reading an old childhood story.

[TURN s2:t1 | speaker=Caroline]
I recommend Becoming Whole; it gave me hope.

[TURN s3:t1 | speaker=Melanie]
I have been reading that book you recommended.

[TURN s4:t1 | speaker=Caroline]
Thanks for your friendship and support."""
    capsule = focused_evidence_capsule(
        "What book did Melanie read from Caroline's suggestion?",
        context,
        max_blocks=2,
    )
    assert "Becoming Whole" in capsule
    assert "that book you recommended" in capsule
    assert "childhood story" not in capsule

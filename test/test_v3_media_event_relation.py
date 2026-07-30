from types import SimpleNamespace

from graphmem_demo.v3.media_relation import media_attribute_hint
from graphmem_demo.v3.retrieval import build_query_frame


def test_media_caption_is_bound_to_named_event_relation() -> None:
    frame = build_query_frame(
        "What did Alex share with Morgan after their hiking trip?"
    )
    turns = [
        SimpleNamespace(
            node_id="s1:t0",
            session_date="1 January, 2024",
            text=(
                "We just returned from our hiking trip. "
                "[Media shared by Morgan; caption: a photography of a ridge at sunset]"
            ),
        ),
        SimpleNamespace(
            node_id="s1:t1",
            session_date="1 January, 2024",
            text=(
                "I also shared this. "
                "[Media shared by Alex; caption: a photo of a bowl of soup]"
            ),
        ),
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["completion_basis"] == "event_relation_bound_caption"
    assert hint["value"] == "a ridge at sunset"
    assert hint["source_turn_ids"][0] == "s1:t0"


def test_media_event_relation_refuses_tied_event_captions() -> None:
    frame = build_query_frame("What was shared after the hiking trip?")
    turns = [
        SimpleNamespace(
            node_id=f"s1:t{index}",
            session_date="1 January, 2024",
            text=f"We finished the hiking trip. [Media shared by Alex; caption: a photo of scene {index}]",
        )
        for index in range(2)
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is False


def test_media_operator_does_not_treat_hosted_show_as_visual_request() -> None:
    frame = build_query_frame(
        "What type of show did Nate host where he taught vegan recipes?"
    )
    turns = [
        SimpleNamespace(
            node_id="s1:t0",
            session_date="1 January, 2024",
            text=(
                "I hosted a cooking show. "
                "[Media shared by Nate; caption: a photo of a desk and microphone]"
            ),
        )
    ]
    assert media_attribute_hint(frame, turns) is None


def test_media_operator_does_not_use_enumeration_as_relevance_proof() -> None:
    frame = build_query_frame("What picture did Alex share?")
    turns = [
        SimpleNamespace(
            node_id="s1:t0",
            session_date="1 January, 2024",
            text="[Media shared by Alex; caption: a photo of a desk, lamp, and chair]",
        ),
        SimpleNamespace(
            node_id="s2:t0",
            session_date="2 January, 2024",
            text="[Media shared by Alex; caption: a photo of a mountain]",
        ),
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is False


def test_exact_date_media_binds_artifact_type_latest_occurrence_and_followup() -> None:
    frame = build_query_frame(
        "What kind of sculpture did Alex share with Morgan on January 1, 2024?"
    )
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_id="s1", turn_index=1,
            speaker="Morgan", session_date="1 January, 2024",
            text="[Media shared by Morgan; caption: a photo of a stone sculpture]",
        ),
        SimpleNamespace(
            node_id="s1:t3", session_id="s1", turn_index=3,
            speaker="Morgan", session_date="1 January, 2024",
            text=(
                "I made an abstract sculpture too. "
                "[Media shared by Morgan; caption: a photo of a metal sculpture on a pedestal]"
            ),
        ),
        SimpleNamespace(
            node_id="s1:t4", session_id="s1", turn_index=4,
            speaker="Alex", session_date="1 January, 2024",
            text="What feeling were you trying to express?",
        ),
        SimpleNamespace(
            node_id="s1:t5", session_id="s1", turn_index=5,
            speaker="Morgan", session_date="1 January, 2024",
            text="The curved silver bands represent motion and balance.",
        ),
        SimpleNamespace(
            node_id="s1:t7", session_id="s1", turn_index=7,
            speaker="Alex", session_date="1 January, 2024",
            text="[Media shared by Alex; caption: a photo of a printed sign]",
        ),
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["completion_basis"] == "exact_date_artifact_type_latest_occurrence"
    assert hint["value"] == "a metal sculpture on a pedestal"
    assert hint["source_turn_ids"][:2] == ["s1:t3", "s1:t5"]
    assert "curved silver bands" in hint["evidence"][-1]

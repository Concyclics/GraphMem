"""The second pass exists to make one referent reachable under its other names.

Every assertion here is about the two failure modes the graph audit found: a key
that never leaves its session joins nothing, and a key that spans the whole
memory routes nowhere.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from graphmem.build.pipeline import GraphBuildPipeline
from graphmem.config import GraphMemV5Config


class _Scene:
    def __init__(self, scene_id, session_id, speakers=("alice",)):
        self.scene_id = scene_id
        self.session_id = session_id
        self.turns = [type("T", (), {"speaker": name})() for name in speakers]


class _Fact:
    def __init__(self, owner, value):
        self.owner = owner
        self.value = value


class _Packet:
    def __init__(self, scene_id, facts):
        self.scene_id = scene_id
        self.facts = [_Fact(owner, value) for owner, value in facts]


def _profile(**coarsen):
    base = GraphMemV5Config()
    return replace(base, coarsen=replace(base.coarsen, entity_merge=True, **coarsen))


def _aliases(scenes, packets, profile):
    return GraphBuildPipeline._merge_aliases(
        GraphBuildPipeline, scenes, {p.scene_id: p for p in packets}, profile)


@pytest.mark.parametrize("text,expected", [
    ("the Dance Studio", "dance studio"),
    ("dance studios", "dance studio"),
    ("Dance Studio's", "dance studio"),
    ("my dance studio", "dance studio"),
    # Folding stops short of stemming, and the -ss/-us/-is guard keeps a
    # singular noun that merely ends in s from losing its last letter.
    ("business", "business"), ("glass", "glass"), ("analysis", "analysis"),
    ("bus", "bus"),
])
def test_surface_folding_is_shallow(text, expected):
    assert GraphBuildPipeline._merge_surface(text) == expected


def test_variants_in_two_sessions_become_aliases():
    scenes = [_Scene("s1", "sess-a"), _Scene("s2", "sess-b"), _Scene("s3", "sess-c"),
              _Scene("s4", "sess-d")]
    packets = [_Packet("s1", [("alice", "the dance studio")]),
               _Packet("s2", [("alice", "Dance Studios")])]
    aliases = _aliases(scenes, packets, _profile())
    assert aliases["dance studios"] == ("the dance studio",)
    assert aliases["the dance studio"] == ("dance studios",)


def test_a_surface_inside_one_session_is_not_merged():
    scenes = [_Scene("s1", "sess-a"), _Scene("s2", "sess-a")]
    packets = [_Packet("s1", [("alice", "the dance studio")]),
               _Packet("s2", [("alice", "Dance Studios")])]
    assert _aliases(scenes, packets, _profile()) == {}


def test_a_surface_spanning_most_of_the_memory_is_dropped():
    """A key everywhere is a stopword, whatever it looks like."""
    scenes = [_Scene(f"s{i}", f"sess-{i}") for i in range(4)]
    packets = [_Packet(f"s{i}", [("alice", "the thing" if i % 2 else "Things")])
               for i in range(4)]
    # Spans 4 of 4 sessions, ceiling is max(2, int(4 * 0.25)) = 2.
    assert _aliases(scenes, packets, _profile()) == {}


def test_speaker_names_are_never_keys():
    """The audit found the eight widest entities were all speaker names."""
    scenes = [_Scene("s1", "sess-a", ("alice", "bobby")),
              _Scene("s2", "sess-b", ("alice", "bobby")),
              _Scene("s3", "sess-c"), _Scene("s4", "sess-d")]
    packets = [_Packet("s1", [("alice", "Bobby")]), _Packet("s2", [("alice", "bobby")])]
    assert _aliases(scenes, packets, _profile()) == {}


def test_disabled_by_default():
    scenes = [_Scene("s1", "sess-a"), _Scene("s2", "sess-b"),
              _Scene("s3", "sess-c"), _Scene("s4", "sess-d")]
    packets = [_Packet("s1", [("alice", "the dance studio")]),
               _Packet("s2", [("alice", "Dance Studios")])]
    assert _aliases(scenes, packets, GraphMemV5Config()) == {}


def test_aliases_reach_postings_across_children():
    """The payoff is in the postings: one surface must reach the other's child."""
    children = [{"child_id": "scene-a", "summary": "", "values": ("the dance studio",),
                 "owners": (), "predicates": (), "scopes": (), "times": ()},
                {"child_id": "scene-b", "summary": "", "values": ("Dance Studios",),
                 "owners": (), "predicates": (), "scopes": (), "times": ()}]
    aliases = {"the dance studio": ("dance studios",),
               "dance studios": ("the dance studio",)}
    compiled = GraphBuildPipeline._compile_routing(
        GraphBuildPipeline, children, 80, aliases)
    postings = compiled["child_postings"]
    assert set(postings["dance studios"]) == {"scene-a", "scene-b"}
    assert set(postings["the dance studio"]) == {"scene-a", "scene-b"}


def test_routing_summary_drops_a_repeated_child():
    """Measured: 35.1% of the words in a routing card were duplicates."""
    children = [{"child_id": f"scene-{i}", "summary": "Jon started a dance studio",
                 "values": (), "owners": (), "predicates": (), "scopes": (), "times": ()}
                for i in range(3)]
    compiled = GraphBuildPipeline._compile_routing(GraphBuildPipeline, children, 80)
    assert compiled["summary"] == "Jon started a dance studio"
    # Every child still indexes its own terms, so nothing became unreachable.
    assert set(compiled["child_postings"]["dance"]) == {"scene-0", "scene-1", "scene-2"}

from __future__ import annotations

from dataclasses import dataclass

from .schema import TurnNodeV36


@dataclass(frozen=True)
class DialogueTopology:
    """Question-independent semantics of the persisted conversation roles."""

    mode: str
    speaker_keys: tuple[str, ...]
    explicit_listener_ratio: float

    @property
    def peer_dialogue(self) -> bool:
        return self.mode == "peer_dialogue"


def infer_dialogue_topology(turns: list[TurnNodeV36]) -> DialogueTopology:
    speakers = tuple(sorted({
        turn.speaker_key for turn in turns if turn.speaker_key
    }))
    listener_count = sum(bool(turn.listener.strip()) for turn in turns)
    listener_ratio = listener_count / len(turns) if turns else 0.0
    mode = (
        "peer_dialogue"
        if len(speakers) >= 2 and listener_ratio >= 0.25
        else "assistant_mediated"
    )
    return DialogueTopology(
        mode=mode,
        speaker_keys=speakers,
        explicit_listener_ratio=listener_ratio,
    )


def is_memory_source(
    turn: TurnNodeV36, topology: DialogueTopology,
) -> bool:
    """Return whether a turn may state facts owned by a conversation speaker."""
    return topology.peer_dialogue or turn.transport_role == "user"

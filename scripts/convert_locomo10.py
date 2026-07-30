#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(
    "/mnt/ssd1/yongan/Resources/RefRepos/locomo_benchmark/locomo/data/locomo10.json"
)


def _session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _answer_session_ids(evidence: list[Any]) -> list[str]:
    numbers: set[int] = set()
    for item in evidence:
        for match in re.finditer(r"\bD(\d+)(?::\d+)?\b", str(item), flags=re.IGNORECASE):
            numbers.add(int(match.group(1)))
    return [f"session_{number}" for number in sorted(numbers)]


def _media_captions(turn: dict[str, Any]) -> list[str]:
    raw = turn.get("blip_caption")
    values = raw if isinstance(raw, list) else [raw]
    return list(dict.fromkeys(
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    ))


def _turn_content(turn: dict[str, Any]) -> str:
    """Preserve textual and caption evidence without consulting QA answers."""
    parts = [str(turn.get("text", "")).strip()]
    speaker = str(turn.get("speaker", "participant")).strip() or "participant"
    parts.extend(
        f"[Media shared by {speaker}; caption: {caption}]"
        for caption in _media_captions(turn)
    )
    return "\n".join(value for value in parts if value)


def convert_sample(sample: dict[str, Any], sample_index: int) -> list[dict[str, Any]]:
    conversation = sample["conversation"]
    speaker_a = str(conversation["speaker_a"])
    speaker_b = str(conversation["speaker_b"])
    numbers = _session_numbers(conversation)
    if not numbers:
        raise ValueError(f"LoCoMo sample {sample_index} contains no sessions")

    session_ids: list[str] = []
    dates: list[str | None] = []
    sessions: list[list[dict[str, Any]]] = []
    for number in numbers:
        session_id = f"session_{number}"
        session_ids.append(session_id)
        dates.append(conversation.get(f"{session_id}_date_time"))
        messages = []
        for turn in conversation[session_id]:
            speaker = str(turn.get("speaker", "")).strip()
            listener = speaker_b if speaker == speaker_a else speaker_a
            captions = _media_captions(turn)
            raw_urls = turn.get("img_url")
            image_urls = raw_urls if isinstance(raw_urls, list) else [raw_urls]
            messages.append(
                {
                    "role": "user" if speaker == speaker_a else "assistant",
                    "speaker": speaker,
                    "listener": listener,
                    "content": _turn_content(turn),
                    "dia_id": turn.get("dia_id"),
                    "media_captions": captions,
                    "media_urls": [
                        str(value) for value in image_urls
                        if value is not None and str(value).strip()
                    ],
                }
            )
        sessions.append(messages)

    sample_id = str(sample.get("sample_id") or f"locomo-{sample_index}")
    question_date = next((date for date in reversed(dates) if date), None)
    rows = []
    for question_index, qa in enumerate(sample.get("qa") or []):
        category = int(qa["category"])
        evidence = list(qa.get("evidence") or [])
        rows.append(
            {
                "question_id": f"locomo{sample_index:02d}_{question_index:04d}",
                "question_type": f"category_{category}",
                "question": str(qa["question"]),
                "answer": qa.get("answer"),
                "question_date": question_date,
                "haystack_sessions": sessions,
                "haystack_session_ids": session_ids,
                "haystack_dates": dates,
                "answer_session_ids": _answer_session_ids(evidence),
                "locomo_sample_id": sample_id,
                "locomo_sample_index": sample_index,
                "locomo_category": category,
                "locomo_evidence": evidence,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the official LoCoMo 10-conversation JSON to GraphMem cases."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=ROOT / "data/locomo10_graphmem.json")
    args = parser.parse_args()

    samples = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [
        row
        for sample_index, sample in enumerate(samples)
        for row in convert_sample(sample, sample_index)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    counts = {
        sample["sample_id"]: len(sample.get("qa") or [])
        for sample in samples
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "conversation_count": len(samples),
                "question_count": len(rows),
                "questions_by_conversation": counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

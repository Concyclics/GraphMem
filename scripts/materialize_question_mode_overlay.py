#!/usr/bin/env python3
"""Select byte-identical prompt-arm rows using observable question modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from materialize_v5_43_typed_readout import _question


TEMPORAL_RE = re.compile(
    r"^(?:when|how long|after how many)\b|"
    r"\b(?:what|which)\s+(?:date|day|month|year)\b", re.I)
PREFIX_RE = re.compile(r"^(?P<mode>who|where|what|which)\b", re.I)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _mode(row: dict) -> str:
    messages = row.get("messages") or ()
    if not messages:
        return "deterministic"
    question = _question(str(messages[-1].get("content") or ""))
    if TEMPORAL_RE.search(question):
        return "temporal"
    match = PREFIX_RE.search(question)
    return match.group("mode").casefold() if match else "other"


def _question_text(row: dict) -> str:
    messages = row.get("messages") or ()
    return (_question(str(messages[-1].get("content") or ""))
            if messages else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--enabled-modes", default="")
    parser.add_argument("--include-question-regex")
    parser.add_argument("--exclude-question-regex")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    base_rows = _read(args.base)
    arm_rows = _read(args.arm)
    if [row["question_id"] for row in base_rows] != [
            row["question_id"] for row in arm_rows]:
        raise ValueError("base/arm question order differs")
    modes = frozenset(value.strip().casefold()
                      for value in args.enabled_modes.split(",") if value.strip())
    unknown = modes - {"temporal", "who", "where", "what", "which"}
    if unknown:
        raise ValueError(f"invalid enabled modes: {sorted(unknown)}")
    include_re = (re.compile(args.include_question_regex, re.I)
                  if args.include_question_regex else None)
    exclude_re = (re.compile(args.exclude_question_regex, re.I)
                  if args.exclude_question_regex else None)
    if not modes and include_re is None:
        raise ValueError("provide enabled modes and/or an include regex")

    output_rows: list[dict] = []
    routes: Counter[str] = Counter()
    deltas: list[int] = []
    for base, arm in zip(base_rows, arm_rows):
        mode = _mode(base)
        question = _question_text(base)
        changed = (base.get("prompt_payload_hash")
                   != arm.get("prompt_payload_hash"))
        selected = bool(
            changed
            and (mode in modes or (include_re and include_re.search(question)))
            and not (exclude_re and exclude_re.search(question)))
        if selected:
            if set(base.get("evidence_turn_ids") or ()) != set(
                    arm.get("evidence_turn_ids") or ()):
                raise ValueError(f"{base['question_id']}: arm changed evidence set")
            chosen = arm
            route = f"arm:{mode}"
        else:
            chosen = base
            route = "base"
        delta = (int(chosen.get("packing_prompt_tokens") or 0)
                 - int(base.get("packing_prompt_tokens") or 0))
        if delta > 0:
            raise ValueError(f"{base['question_id']}: overlay increased tokens")
        deltas.append(delta)
        routes[route] += 1
        output_rows.append(chosen)

    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-question-mode-overlay-v1",
        "base": {"path": str(args.base),
                 "sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()},
        "arm": {"path": str(args.arm),
                "sha256": hashlib.sha256(args.arm.read_bytes()).hexdigest()},
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(output_rows),
        "enabled_modes": sorted(modes),
        "include_question_regex": args.include_question_regex,
        "exclude_question_regex": args.exclude_question_regex,
        "routes": dict(sorted(routes.items())),
        "routing_inputs": ["question wording only"],
        "benchmark_gold_prediction_or_judge_routing": False,
        "packing_token_delta": {
            "mean": sum(deltas) / max(1, len(deltas)),
            "min": min(deltas, default=0),
            "max": max(deltas, default=0),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

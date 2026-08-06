#!/usr/bin/env python3
"""Phase A: what each build knob costs in tokens, and what it costs in coverage.

One knob per arm against a common baseline, so the two columns can be read as a
price list.  Both columns are deterministic -- token counts come off the build
ledger and coverage off the build diagnostics -- so unlike a judged comparison
this grid needs no repeats to separate signal from noise.  The judge's own
self-consistency was measured at 1.69% of questions flipping between two runs of
the *same* answers, which is why accuracy is deliberately not measured here.

Coverage is a proxy, not accuracy.  It says what share of informative turns a
fact was extracted from; whether losing coverage loses answers is the follow-up,
and it should be spent only on the two or three arms this grid shortlists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Each arm is the baseline with exactly one field changed, so a row reads as
#: "this is what that knob buys or costs".
ARMS: dict[str, dict] = {
    # Every arm runs with a 32K output ceiling so no call is ever clipped.  Under
    # the shipped 2048 the arms truncated at 6.69%, 6.38% and 0.83%, and a
    # truncated call loses its facts outright -- so a knob that merely shortened
    # the output scored better on coverage for the wrong reason.  Output length
    # belongs to the prompt; the ceiling is only a runaway guard.
    #
    # The axis is which LLM-extracted element is present.  B0 is the floor: the
    # fact tuple and nothing else.
    "B0_core":       {"semantic_quote_evidence": False,
                      "semantic_scene_summary_chars": 0, "semantic_scene_entities": False},
    "B1_quote":      {"semantic_quote_evidence": True,
                      "semantic_scene_summary_chars": 0, "semantic_scene_entities": False},
    "B2_summary":    {"semantic_quote_evidence": False,
                      "semantic_scene_summary_chars": 160, "semantic_scene_entities": False},
    "B3_entities":   {"semantic_quote_evidence": False,
                      "semantic_scene_summary_chars": 0, "semantic_scene_entities": True},
    "B4_all":        {"semantic_quote_evidence": True,
                      "semantic_scene_summary_chars": 160, "semantic_scene_entities": True},
    # Orthogonal to the element axis: how tightly the prompt shapes `p`.
    "B5_all_free_p": {"semantic_quote_evidence": True,
                      "semantic_scene_summary_chars": 160, "semantic_scene_entities": True,
                      "semantic_predicate_max_chars": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path,
                        default=ROOT / "configs/v5/v5_8_units.json")
    parser.add_argument("--output-root", type=Path, default=Path("../artifacts/v5_8/phase_a"))
    parser.add_argument("--memories", type=int, default=10)
    parser.add_argument("--memory-workers", type=int, default=5)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_config.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_dir = args.output_root / "configs"; config_dir.mkdir(exist_ok=True)
    results: dict[str, dict] = {}

    for arm in args.arms:
        overrides = ARMS[arm]
        blob = json.loads(json.dumps(base))
        # Nothing may clip and nothing may degrade: this grid measures what each
        # element *costs*, so both the output ceiling and the per-memory budget
        # are lifted.  Under the shipped settings the arms truncated at 6.69% and
        # then, once the ceiling was raised, starved -- the ledger reserved the
        # ceiling per call, exhausted 220,000 in about seven calls and drove
        # extraction into fallback on 100 scenes at 0.28 facts per scene.  Both
        # confounds are removed here; the budget is re-imposed afterwards, once
        # the true price of each element is known.
        blob["models"]["semantic_batch_output_tokens"] = 32768
        blob["models"]["semantic_max_tokens_per_memory"] = 0
        blob["models"]["semantic_expected_output_tokens"] = 600
        blob["models"]["semantic_fallback_on_overrun"] = False
        blob["models"].update(overrides)
        blob["storage"]["sqlite_path"] = f"artifacts/v5_8/phase_a/{arm}.sqlite"
        path = config_dir / f"{arm}.json"
        path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

        print(f"\n=== {arm}  {overrides or '(baseline)'} ===", flush=True)
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/measure_v5_8_unit_gate.py"),
             "--output-root", str(args.output_root / arm), "--config", str(path),
             "--memories", str(args.memories),
             "--memory-workers", str(args.memory_workers)],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stderr.strip()[-400:]}", flush=True)
            results[arm] = {"error": proc.stderr.strip()[-400:]}
            continue
        summaries = sorted((args.output_root / arm).glob("unit_gate_*/summary.json"))
        measured = json.loads(summaries[-1].read_text())["measured"]
        measured["wall_seconds"] = round(time.perf_counter() - started, 1)
        results[arm] = measured
        print(f"  tokens_mean={measured['tokens_mean']:,.0f} "
              f"coverage={measured['coverage']:.4f} "
              f"truncation={measured['truncation_rate']:.4f}", flush=True)

    (args.output_root / "phase_a.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    base_row = results.get("B0_core", {})
    print(f"\n{'arm':20}{'tok/mem':>10}{'vs base':>9}{'coverage':>10}{'vs base':>9}"
          f"{'trunc':>8}{'s/mem':>8}")
    for arm, row in results.items():
        if "error" in row:
            print(f"{arm:20}  FAILED"); continue
        dt = (row["tokens_mean"] / base_row["tokens_mean"] - 1) * 100 if base_row else 0
        dc = (row["coverage"] - base_row["coverage"]) if base_row else 0
        print(f"{arm:20}{row['tokens_mean']:10,.0f}{dt:+8.1f}%{row['coverage']:10.4f}"
              f"{dc:+9.4f}{row['truncation_rate']:8.3f}{row['seconds_per_memory']:8.0f}")
    print("\nCoverage is a build-side proxy.  Shortlist two or three arms here, then "
          "spend a judged run on those to price the accuracy side of the trade-off.")


if __name__ == "__main__":
    main()

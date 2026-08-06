#!/usr/bin/env python3
"""Merge the sharded mandatory-ranking A/B and report per-category deltas.

Shards are split by memory, so their raw counters are disjoint and add exactly.
Ratios are recomputed from the merged counters rather than averaged across
shards, which would weight a 40-question shard the same as a 400-question one.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    phat = hits / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, dict[str, Counter]] = {}
    wins = losses = 0
    mandatory_sum = mandatory_n = 0
    budget = None
    files = sorted(args.input.glob("mandatory_rank*.json"))
    if not files:
        raise SystemExit(f"no shard output under {args.input}")
    for path in files:
        blob = json.loads(path.read_text())
        budget = blob.get("budget", budget)
        wins += blob.get("wins", 0); losses += blob.get("losses", 0)
        mandatory_sum += blob.get("mandatory_mean", 0) * blob.get("mandatory_n", 0)
        mandatory_n += blob.get("mandatory_n", 0)
        for arm, strata in blob.get("raw", {}).items():
            target = merged.setdefault(arm, defaultdict(Counter))
            for stratum, counter in strata.items():
                target[stratum].update(counter)

    print(f"shards={len(files)}  budget={budget}  "
          f"mandatory turns/question mean={mandatory_sum/max(1,mandatory_n):.1f}")
    print(f"\n{'stratum':28}{'n':>6}{'control':>10}{'ranked':>10}{'delta':>9}{'  95% CI (ranked)':>20}")
    total = Counter()
    for stratum in sorted(merged.get("control", {})):
        c = merged["control"][stratum]; r = merged["rank_mandatory"][stratum]
        n = c["n"]
        ca, ra = c["all_hit"] / max(1, n), r["all_hit"] / max(1, r["n"])
        low, high = wilson(r["all_hit"], r["n"])
        flag = "  <-- regression" if ra < ca - 1e-9 else ""
        print(f"{stratum:28}{n:6d}{ca:10.3f}{ra:10.3f}{ra-ca:+9.3f}   [{low:.3f}, {high:.3f}]{flag}")
        total["n"] += n; total["c"] += c["all_hit"]; total["r"] += r["all_hit"]
        total["cr_num"] += c["recall_num"]; total["cr_den"] += c["recall_den"]
        total["rr_num"] += r["recall_num"]; total["rr_den"] += r["recall_den"]
    ca, ra = total["c"] / max(1, total["n"]), total["r"] / max(1, total["n"])
    print(f"\n{'ALL':28}{total['n']:6d}{ca:10.3f}{ra:10.3f}{ra-ca:+9.3f}")
    print(f"{'turn_recall':28}{'':6}{total['cr_num']/max(1,total['cr_den']):10.3f}"
          f"{total['rr_num']/max(1,total['rr_den']):10.3f}")
    print(f"\npaired: ranked wins {wins}, loses {losses}, ties {total['n']-wins-losses}")
    if wins + losses:
        # Sign test: under the null the two arms are exchangeable, so wins is
        # Binomial(wins+losses, 0.5).  This is a retrieval-metric check only --
        # the judged accuracy still has to be measured separately.
        n = wins + losses
        p = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n * 2
        print(f"sign test two-sided p = {min(1.0, p):.4f}")


if __name__ == "__main__":
    main()

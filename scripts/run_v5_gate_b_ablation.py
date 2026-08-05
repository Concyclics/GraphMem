#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from graphmem.ablation import GateBAblationRunner
from graphmem.config import load_config
from graphmem.retrieval import NavigatorVariant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--lme", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-200", action="store_true")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--enable-embedding", action="store_true")
    parser.add_argument("--embedding-cache-db", type=Path)
    parser.add_argument("--profiles", default="b0,b1,b2,b3,b4,b5")
    parser.add_argument("--navigator", choices=[str(item) for item in NavigatorVariant])
    args = parser.parse_args()
    runner = GateBAblationRunner(
        repo=args.repo, artifact_root=args.artifact_root,
        lme_path=args.lme, locomo_path=args.locomo, gold_path=args.gold,
        config=load_config(args.config), enable_llm=args.enable_llm,
        enable_embedding=args.enable_embedding,
        embedding_cache_db=args.embedding_cache_db,
        profiles=tuple(item.strip() for item in args.profiles.split(",") if item.strip()),
        forced_navigator=args.navigator,
    )
    manifest = runner.run_funnel(full_confirm=args.full_200)
    print(manifest.run_id)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import GraphMemV5Config, config_hash
from .domain import RunManifest, stable_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_dataset_hash(paths: Iterable[Path]) -> str:
    # Dataset identity follows content, not a machine-specific absolute path.
    rows = [file_sha256(path) for path in paths]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def git_commit(repo: Path) -> str:
    resolved_repo = repo.resolve()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={resolved_repo}", "rev-parse", "HEAD"],
        cwd=resolved_repo, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def software_versions(names: Iterable[str]) -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def create_run_manifest(
    *,
    repo: Path,
    dataset_paths: Iterable[Path],
    config: GraphMemV5Config,
    graph_artifact_ids: Iterable[str] = (),
    model_ids: Mapping[str, str] | None = None,
    prompt_hashes: Mapping[str, str] | None = None,
    hardware: Mapping[str, Any] | None = None,
    started_at: str | None = None,
) -> RunManifest:
    dataset_hash = combined_dataset_hash(dataset_paths)
    cfg_hash = config_hash(config)
    started = started_at or datetime.now(timezone.utc).isoformat()
    resolved_models = dict(model_ids or {
        "llm": config.models.llm_model,
        "embedding": config.models.embedding_model,
    })
    resolved_prompts = dict(prompt_hashes or {})
    commit = git_commit(repo)
    return RunManifest(
        run_id=stable_id(
            "run", commit, dataset_hash, cfg_hash, resolved_models,
            resolved_prompts, config.random_seed,
        ),
        git_commit=commit, dataset_hash=dataset_hash,
        config_hash=cfg_hash, graph_artifact_ids=tuple(graph_artifact_ids),
        model_ids=resolved_models,
        prompt_hashes=resolved_prompts, random_seed=config.random_seed,
        started_at=started,
        hardware=dict(hardware or {"platform": platform.platform()}),
        software_versions=software_versions(
            ["openai", "numpy", "pandas", "rank-bm25", "transformers"]
        ),
    )

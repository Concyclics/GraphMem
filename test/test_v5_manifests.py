from __future__ import annotations

from pathlib import Path

from graphmem.config import load_config
from graphmem.manifests import combined_dataset_hash, create_run_manifest


def test_dataset_hash_is_order_sensitive_and_content_sensitive(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    original = combined_dataset_hash([first, second])
    assert original != combined_dataset_hash([second, first])
    first.write_text("changed", encoding="utf-8")
    assert original != combined_dataset_hash([first, second])


def test_manifest_captures_commit_dataset_and_config_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/v5/gate_a.json")
    manifest = create_run_manifest(
        repo=root,
        dataset_paths=[root / "configs/v5/gate_a.json"],
        config=config,
        started_at="2026-08-04T00:00:00+00:00",
    )
    assert len(manifest.git_commit) == 40
    assert len(manifest.dataset_hash) == 64
    assert len(manifest.config_hash) == 64
    assert manifest.run_id.startswith("run:")

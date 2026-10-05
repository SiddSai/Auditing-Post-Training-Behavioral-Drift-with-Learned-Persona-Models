from __future__ import annotations

from pathlib import Path

from .io import atomic_json


def snapshot_hf(repo_id: str, revision: str, output_dir: str | Path, repo_type: str = "dataset", allow_patterns: list[str] | None = None) -> Path:
    """Download a Hub snapshot at an immutable revision and write provenance."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to snapshot Hugging Face sources") from exc
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision, local_dir=target, allow_patterns=allow_patterns))
    atomic_json(target / "SOURCE_PROVENANCE.json", {
        "repo_id": repo_id, "repo_type": repo_type, "revision": revision,
        "snapshot_path": str(snapshot),
    })
    return snapshot

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .errors import ManifestError
from .io import atomic_json, atomic_jsonl, file_sha256
from .manifests import SHA, load_anchors


ANTHROPIC_EVALS_REPO = "https://github.com/anthropics/evals.git"
# Resolved from refs/heads/main on 2026-10-05. The CLI still requires the
# revision argument to be a full SHA, so it cannot silently follow main later.
ANTHROPIC_EVALS_PERSONA_REVISION = "84fcc677e52e1902d696c32cd1a6b663e70d3993"


def snapshot_git(repository: str, revision: str, output_dir: str | Path) -> Path:
    """Create a complete, immutable Git snapshot at a verified full SHA."""
    if not SHA.fullmatch(revision):
        raise ManifestError("Git source revision must be a full 40-character SHA")
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        provenance = target / "SOURCE_PROVENANCE.json"
        if provenance.exists():
            recorded = json.loads(provenance.read_text(encoding="utf-8"))
            if recorded.get("repository") == repository and recorded.get("resolved_revision") == revision:
                return target
        raise ManifestError(f"Refusing to overwrite non-empty source snapshot directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="git-source-", dir=target.parent))
    checkout = workspace / "checkout"
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", revision], check=True)
        resolved = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if resolved != revision:
            raise ManifestError(f"Git source resolved {resolved}, not requested {revision}")
        if target.exists():
            target.rmdir()
        os.replace(checkout, target)
        atomic_json(target / "SOURCE_PROVENANCE.json", {"repository": repository, "requested_revision": revision, "resolved_revision": resolved, "snapshot_kind": "git_checkout"})
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return target


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


def snapshot_anthropic_persona(output_dir: str | Path, revision: str = ANTHROPIC_EVALS_PERSONA_REVISION) -> Path:
    """Create a sparse, SHA-pinned local checkout of Anthropic's persona corpus."""
    if not SHA.fullmatch(revision):
        raise ManifestError("Anthropic source revision must be a full 40-character Git SHA")
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise ManifestError(f"Refusing to overwrite non-empty source snapshot directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="anthropic-evals-", dir=target.parent))
    checkout = workspace / "evals"
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", ANTHROPIC_EVALS_REPO, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "persona"], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", revision], check=True)
        resolved = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if resolved != revision or not (checkout / "persona").is_dir():
            raise ManifestError("Anthropic checkout did not resolve to the requested revision/persona directory")
        if target.exists():
            target.rmdir()
        os.replace(checkout, target)
        atomic_json(target / "SOURCE_PROVENANCE.json", {
            "source": "anthropic-evals-persona", "repository": ANTHROPIC_EVALS_REPO,
            "requested_revision": revision, "resolved_revision": resolved,
            "protocol_note": "Raw questions are imported without Anthropic's model-specific <EOT>/Human/Assistant wrapper.",
        })
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    return target


def _persona_source_dir(source_dir: str | Path) -> Path:
    supplied = Path(source_dir)
    directory = supplied / "persona" if (supplied / "persona").is_dir() else supplied
    if not directory.is_dir():
        raise ManifestError(f"Anthropic persona source directory not found: {directory}")
    return directory


def import_anthropic_persona(
    source_dir: str | Path,
    source_revision: str,
    output_path: str | Path,
    provenance_path: str | Path | None = None,
    min_label_confidence: float | None = None,
) -> None:
    """Convert the official persona JSONL schema into this project's canonical anchor schema.

    Prompts are the upstream ``question`` field verbatim. This intentionally
    differs from Anthropic's old model-specific wrapper and implements this
    study's fixed raw-prompt protocol.
    """
    if not SHA.fullmatch(source_revision):
        raise ManifestError("source_revision must be the 40-character upstream Git SHA")
    if min_label_confidence is not None and not 0 <= min_label_confidence <= 1:
        raise ManifestError("min_label_confidence must be in [0, 1]")
    directory = _persona_source_dir(source_dir)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise ManifestError(f"No persona JSONL files in {directory}")
    anchors: list[dict] = []
    source_files: list[dict[str, str | int]] = []
    required = {"question", "statement", "answer_matching_behavior", "answer_not_matching_behavior", "label_confidence"}
    for source_file in files:
        family = source_file.stem
        retained = 0
        with source_file.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"Invalid JSON in {source_file}:{line_number}") from exc
                missing = required - record.keys()
                if missing:
                    raise ManifestError(f"{source_file}:{line_number} missing upstream fields {sorted(missing)}")
                matching = record["answer_matching_behavior"]
                nonmatching = record["answer_not_matching_behavior"]
                if {matching, nonmatching} != {" Yes", " No"}:
                    raise ManifestError(f"{source_file}:{line_number} does not use the official Yes/No answer pair")
                confidence = float(record["label_confidence"])
                if not 0 <= confidence <= 1:
                    raise ManifestError(f"{source_file}:{line_number} has invalid label confidence")
                if min_label_confidence is not None and confidence < min_label_confidence:
                    continue
                if not isinstance(record["question"], str) or not record["question"].strip():
                    raise ManifestError(f"{source_file}:{line_number} has an empty question")
                anchors.append({
                    "anchor_id": f"anthropic.persona.{family}.{line_number:04d}",
                    "prompt_raw": record["question"], "candidates": [matching, nonmatching],
                    "behavior_consistent_candidate": matching, "family": f"anthropic_persona:{family}",
                    "source": "anthropics/evals:persona", "source_revision": source_revision,
                    "label_confidence": confidence, "source_file": source_file.name,
                    "source_line": line_number, "raw_protocol": "upstream_question_verbatim",
                })
                retained += 1
        source_files.append({"path": source_file.name, "sha256": file_sha256(source_file), "retained_records": retained})
    if not anchors:
        raise ManifestError("No Anthropic persona records survived import")
    anchors.sort(key=lambda row: row["anchor_id"])
    atomic_jsonl(output_path, anchors)
    load_anchors(output_path)
    destination = Path(output_path)
    atomic_json(provenance_path or destination.with_suffix(destination.suffix + ".provenance.json"), {
        "source": "anthropics/evals/persona", "repository": ANTHROPIC_EVALS_REPO,
        "source_revision": source_revision, "source_directory": str(directory),
        "raw_prompt_policy": "The official question field is retained verbatim; no <EOT>, Human/Assistant, system, or chat-template wrapper is added.",
        "min_label_confidence": min_label_confidence, "source_files": source_files,
        "anchor_count": len(anchors), "anchors_sha256": file_sha256(destination),
    })

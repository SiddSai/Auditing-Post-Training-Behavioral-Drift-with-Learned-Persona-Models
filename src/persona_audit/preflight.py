from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import InferenceError
from .io import atomic_json, file_sha256
from .manifests import SHA, load_anchors


def validate_tokenizer_candidates(
    model_repo: str, model_revision: str, anchors_path: str | Path, output_path: str | Path
) -> None:
    """Prove all canonical candidate answers are one token for a pinned model tokenizer."""
    if not SHA.fullmatch(model_revision):
        raise InferenceError("model_revision must be a full 40-character Hugging Face commit SHA")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to validate the tokenizer") from exc
    tokenizer: Any = AutoTokenizer.from_pretrained(model_repo, revision=model_revision)
    anchors = load_anchors(anchors_path)
    candidates = sorted({candidate for anchor in anchors for candidate in anchor.candidates})
    tokenization = {
        candidate: tokenizer.encode(candidate, add_special_tokens=False)
        for candidate in candidates
    }
    incompatible = {candidate: ids for candidate, ids in tokenization.items() if len(ids) != 1}
    report = {
        "model_repo": model_repo, "model_revision": model_revision,
        "anchors_path": str(anchors_path), "anchors_sha256": file_sha256(anchors_path),
        "anchor_count": len(anchors), "candidate_token_ids": tokenization,
        "compatible": not incompatible, "incompatible_candidates": incompatible,
    }
    atomic_json(output_path, report)
    if incompatible:
        raise InferenceError(
            "Raw candidate tokenization is incompatible with exact one-token scoring; "
            f"see {output_path} for details."
        )

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InferenceError
from .io import atomic_json, file_sha256, read_jsonl
from .interfaces import load_interfaces, render_direct_answer_anchor_prompts
from .manifests import SHA, load_anchors, load_nodes, load_wild_nodes


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


def audit_direct_answer_tokenizers(
    nodes_path: str | Path, wild_nodes_path: str | Path, anchors_path: str | Path,
    interfaces_path: str | Path, output_path: str | Path,
) -> None:
    """Preflight every pinned assistant tokenizer before an expensive run."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to validate tokenizers") from exc
    nodes = load_nodes(nodes_path) + load_wild_nodes(wild_nodes_path)
    interfaces = load_interfaces(interfaces_path, nodes)
    anchors = load_anchors(anchors_path)
    candidates = sorted({candidate for anchor in anchors for candidate in anchor.candidates})
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for node in nodes:
        interface = interfaces[node.node_id]
        if interface["rendering"] != "native_chat_template":
            continue
        tokenizer: Any = AutoTokenizer.from_pretrained(node.repo_id, revision=node.commit_sha)
        tokenization = {candidate: tokenizer.encode(candidate, add_special_tokens=False) for candidate in candidates}
        incompatible: dict[str, object] = {candidate: ids for candidate, ids in tokenization.items() if len(ids) != 1}
        try:
            rendered, removed_think = render_direct_answer_anchor_prompts(tokenizer, anchors[:1], interface)
            nonempty = bool(rendered and rendered[0].strip())
        except Exception as exc:  # Keep all failures in an auditable report.
            nonempty, removed_think = False, False
            incompatible["__rendering_error__"] = str(exc)
        row = {
            "node_id": node.node_id, "repo_id": node.repo_id, "commit_sha": node.commit_sha,
            "candidate_token_ids": tokenization, "compatible": not incompatible,
            "incompatible": incompatible, "rendered_nonempty": nonempty,
            "template_forces_think_suffix": removed_think,
        }
        rows.append(row)
        if incompatible or not nonempty:
            failures.append(node.node_id)
    atomic_json(output_path, {
        "purpose": "Preflight exact direct-answer candidate scoring and source-derived no-think rendering for every assistant model.",
        "anchors_sha256": file_sha256(anchors_path), "interfaces_sha256": file_sha256(interfaces_path),
        "assistant_nodes_checked": len(rows), "rows": rows,
    })
    if failures:
        raise InferenceError(f"Direct-answer tokenizer preflight failed for {failures}; see {output_path}")


def validate_anchor_measurement(
    observations_path: str | Path, output_path: str | Path, min_node_median_mass: float = 1e-4,
) -> None:
    """Gate state fitting on non-degenerate forced-choice probability mass."""
    rows = read_jsonl(observations_path)
    if not rows:
        raise InferenceError("No anchor observations to validate")
    by_node: dict[str, list[float]] = defaultdict(list)
    protocols: set[str] = set()
    think_flags: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        if "candidate_total_probability" not in row:
            raise InferenceError("Observations lack candidate_total_probability; rerun with direct-answer protocol")
        mass = float(row["candidate_total_probability"])
        if not math.isfinite(mass) or mass < 0 or mass > 1.000001:
            raise InferenceError(f"Invalid candidate mass for {row.get('node_id')}: {mass}")
        node_id = str(row["node_id"])
        by_node[node_id].append(mass)
        protocols.add(str(row.get("anchor_protocol")))
        think_flags[node_id].add(bool(row.get("template_forced_think_suffix_removed", False)))
    report_rows = []
    failures = []
    for node_id, masses in sorted(by_node.items()):
        median = float(np.median(masses))
        report = {
            "node_id": node_id, "n_anchors": len(masses), "mean_candidate_mass": float(np.mean(masses)),
            "median_candidate_mass": median, "min_candidate_mass": float(np.min(masses)),
            "template_forced_think_suffix": len(think_flags[node_id]) == 1 and next(iter(think_flags[node_id])),
            "passes_minimum": median >= min_node_median_mass,
        }
        report_rows.append(report)
        if not report["passes_minimum"]:
            failures.append(node_id)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    atomic_json(destination.with_suffix(".metadata.json"), {
        "observations_sha256": file_sha256(observations_path), "anchor_protocols": sorted(protocols),
        "minimum_node_median_mass": min_node_median_mass,
        "interpretation": "Low total Yes/No mass means the normalized logit margin compares implausible continuations and must not be used as a primary state feature.",
        "failed_nodes": failures,
    })
    if failures:
        raise InferenceError(f"Anchor-measurement gate failed for {len(failures)} node(s); see {output_path}")

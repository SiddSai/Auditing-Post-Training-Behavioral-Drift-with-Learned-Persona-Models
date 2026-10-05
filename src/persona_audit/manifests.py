from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ManifestError
from .io import read_jsonl

SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelNode:
    node_id: str
    repo_id: str
    revision: str
    commit_sha: str
    phase: str
    protocol: str


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    prompt_raw: str
    candidates: tuple[str, ...]
    behavior_consistent_candidate: str
    family: str
    source: str
    source_revision: str
    label_confidence: float | None


def load_nodes(path: str | Path) -> list[ModelNode]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"node_id", "repo_id", "revision", "commit_sha", "phase", "protocol"}
    if not rows or set(rows[0]) < required:
        raise ManifestError(f"{path} is missing required node columns: {sorted(required)}")
    nodes: list[ModelNode] = []
    ids: set[str] = set()
    for row in rows:
        if row["node_id"] in ids:
            raise ManifestError(f"Duplicate node_id: {row['node_id']}")
        if not SHA.fullmatch(row["commit_sha"]):
            raise ManifestError(f"{row['node_id']} does not have a 40-character pinned SHA")
        if "/" not in row["repo_id"]:
            raise ManifestError(f"{row['node_id']} has invalid repo_id")
        ids.add(row["node_id"])
        nodes.append(ModelNode(**{key: row[key] for key in ModelNode.__dataclass_fields__}))
    return nodes


def load_wild_nodes(path: str | Path) -> list[ModelNode]:
    """Adapt admitted wild descendants to the same immutable inference contract.

    The source manifest intentionally preserves its research-provenance schema;
    this function never promotes those rows to causal lineage claims.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"candidate_id", "commit_sha", "intervention_family", "panel", "decision"}
    if not rows or set(rows[0]) < required:
        raise ManifestError(f"{path} is missing required wild-node columns: {sorted(required)}")
    nodes: list[ModelNode] = []
    ids: set[str] = set()
    for row in rows:
        if row["panel"] != "wild_observational" or row["decision"] != "admit_not_causal":
            continue
        if not SHA.fullmatch(row["commit_sha"]):
            raise ManifestError(f"{row['candidate_id']} does not have a 40-character pinned SHA")
        node_id = "wild--" + row["candidate_id"].replace("/", "--")
        if node_id in ids:
            raise ManifestError(f"Duplicate wild node_id after normalization: {node_id}")
        ids.add(node_id)
        nodes.append(ModelNode(
            node_id=node_id, repo_id=row["candidate_id"], revision="wild-observational",
            commit_sha=row["commit_sha"], phase=f"wild_{row['intervention_family']}", protocol="raw_prompt",
        ))
    if not nodes:
        raise ManifestError(f"{path} contains no admitted wild observational nodes")
    return nodes


def load_anchors(path: str | Path) -> list[Anchor]:
    anchors: list[Anchor] = []
    seen: set[str] = set()
    for raw in read_jsonl(path):
        required = {"anchor_id", "prompt_raw", "candidates", "behavior_consistent_candidate", "family", "source", "source_revision"}
        missing = required - raw.keys()
        if missing:
            raise ManifestError(f"Anchor missing fields {sorted(missing)}")
        candidates = tuple(raw["candidates"])
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ManifestError(f"{raw['anchor_id']} must contain two or more unique candidates")
        if raw["behavior_consistent_candidate"] not in candidates:
            raise ManifestError(f"{raw['anchor_id']} behavior-consistent candidate is absent")
        if raw["anchor_id"] in seen:
            raise ManifestError(f"Duplicate anchor_id: {raw['anchor_id']}")
        confidence = raw.get("label_confidence")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ManifestError(f"{raw['anchor_id']} label_confidence must be in [0, 1]")
        seen.add(raw["anchor_id"])
        anchors.append(Anchor(
            anchor_id=raw["anchor_id"], prompt_raw=raw["prompt_raw"], candidates=candidates,
            behavior_consistent_candidate=raw["behavior_consistent_candidate"], family=raw["family"],
            source=raw["source"], source_revision=raw["source_revision"],
            label_confidence=float(confidence) if confidence is not None else None,
        ))
    if not anchors:
        raise ManifestError("Anchor manifest is empty")
    return anchors

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from .errors import ManifestError
from .io import atomic_json, file_sha256, read_jsonl
from .manifests import load_anchors


@dataclass(frozen=True)
class StateFitMetadata:
    method: str
    dimensions: int
    feature: str
    fit_node_ids: list[str]
    observation_path_sha256: str
    anchor_manifest_sha256: str


def _load_observation_matrix(
    observations_path: str | Path, node_ids: list[str], expected_anchor_ids: list[str], feature: str
) -> tuple[list[str], np.ndarray]:
    rows = read_jsonl(observations_path)
    by_node: dict[str, dict[str, float]] = {node: {} for node in node_ids}
    for row in rows:
        node, anchor = row["node_id"], row["anchor_id"]
        if node in by_node:
            if anchor in by_node[node]:
                raise ManifestError(f"Duplicate observation for node={node!r}, anchor={anchor!r}")
            if feature not in row:
                raise ManifestError(f"Observation for node={node!r}, anchor={anchor!r} lacks feature {feature!r}")
            value = float(row[feature])
            if not np.isfinite(value):
                raise ManifestError(f"Observation for node={node!r}, anchor={anchor!r} has non-finite {feature!r}")
            by_node[node][anchor] = value
    observed_anchors = sorted({anchor for values in by_node.values() for anchor in values})
    expected = sorted(expected_anchor_ids)
    if not observed_anchors:
        raise ManifestError("No observations match requested nodes")
    missing_from_run = sorted(set(expected) - set(observed_anchors))
    unexpected_in_run = sorted(set(observed_anchors) - set(expected))
    if missing_from_run or unexpected_in_run:
        raise ManifestError(
            "Observation anchors do not exactly match the canonical anchor manifest "
            f"(missing={len(missing_from_run)}, unexpected={len(unexpected_in_run)})"
        )
    matrix = np.empty((len(node_ids), len(expected)), dtype=np.float64)
    for i, node in enumerate(node_ids):
        missing = [anchor for anchor in expected if anchor not in by_node[node]]
        if missing:
            raise ManifestError(f"Node {node} is missing {len(missing)} anchor observations")
        matrix[i] = [by_node[node][anchor] for anchor in expected]
    return expected, matrix


def fit_state(
    observations_path: str | Path,
    anchor_manifest_path: str | Path,
    fit_node_ids: list[str],
    transform_node_ids: list[str],
    output_dir: str | Path,
    dimensions: int = 8,
    method: str = "pca",
    feature: str = "behavior_logit_margin",
) -> None:
    if not set(fit_node_ids) <= set(transform_node_ids):
        raise ManifestError("fit_node_ids must be included in transform_node_ids")
    expected_anchor_ids = [anchor.anchor_id for anchor in load_anchors(anchor_manifest_path)]
    if dimensions < 1 or dimensions > min(len(fit_node_ids), len(expected_anchor_ids)):
        raise ManifestError("dimensions must be between 1 and min(number of fit nodes, number of anchors)")
    if feature not in {"behavior_logit_margin", "behavior_probability"}:
        raise ManifestError("feature must be behavior_logit_margin or behavior_probability")
    anchors, matrix = _load_observation_matrix(observations_path, transform_node_ids, expected_anchor_ids, feature)
    fit_indices = [transform_node_ids.index(node) for node in fit_node_ids]
    scaler = StandardScaler().fit(matrix[fit_indices])
    scaled = scaler.transform(matrix)
    encoder = PCA(n_components=dimensions, random_state=0) if method == "pca" else FactorAnalysis(n_components=dimensions, random_state=0)
    encoder.fit(scaled[fit_indices])
    states = encoder.transform(scaled)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "state_model.npz", mean=scaler.mean_, scale=scaler.scale_, components=getattr(encoder, "components_"), anchors=np.array(anchors))
    with (destination / "states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", *[f"z_{i:02d}" for i in range(dimensions)]])
        writer.writerows([[node, *row] for node, row in zip(transform_node_ids, states, strict=True)])
    metadata = StateFitMetadata(method, dimensions, feature, fit_node_ids, file_sha256(observations_path), file_sha256(anchor_manifest_path))
    atomic_json(destination / "metadata.json", {**asdict(metadata), "transform_node_ids": transform_node_ids})

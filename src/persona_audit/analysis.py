"""Post-hoc audits for anchor measurement and observed checkpoint drift.

These analyses deliberately use only data already produced by the primary
run.  They do not create target labels, alter the frozen split, or feed target
outcomes into construction of a behavioral state.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .errors import ManifestError
from .io import atomic_json, atomic_jsonl, file_sha256, read_jsonl
from .manifests import load_anchors, load_nodes, load_wild_nodes
from .predictor import _states_for_fold
from .state import _load_observation_matrix


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rank(left), _rank(right))


def audit_anchor_panel(
    observations: str | Path, anchors_path: str | Path, node_ids_path: str | Path, output_dir: str | Path,
    dimensions: int = 4, split_half_repeats: int = 100, seed: int = 20261008,
    wild_nodes_path: str | Path | None = None, assistant_only: bool = False,
) -> None:
    """Measure anchor information, redundancy, and split-half state stability.

    Reliability is assessed through correlations between pairwise model
    distances in independently sampled anchor halves.  It is rotation
    invariant, unlike correlating raw PCA coordinates across halves.
    """
    manifest_nodes = load_nodes(node_ids_path) + (load_wild_nodes(wild_nodes_path) if wild_nodes_path else [])
    nodes = [node.node_id for node in manifest_nodes if not assistant_only or node.phase.startswith("posttrain_") or node.phase.startswith("wild_")]
    anchors = load_anchors(anchors_path)
    anchor_ids = [anchor.anchor_id for anchor in anchors]
    _, probabilities = _load_observation_matrix(observations, nodes, anchor_ids, "behavior_probability")
    _, margins = _load_observation_matrix(observations, nodes, anchor_ids, "behavior_logit_margin")
    if dimensions < 1 or dimensions >= min(len(nodes), len(anchors)):
        raise ManifestError("dimensions must be below both model and anchor counts")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    per_anchor: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        per_anchor.append({
            "anchor_id": anchor.anchor_id, "family": anchor.family,
            "mean_probability": float(probabilities[:, index].mean()),
            "std_probability": float(probabilities[:, index].std()),
            "mean_logit_margin": float(margins[:, index].mean()),
            "std_logit_margin": float(margins[:, index].std()),
            "label_confidence": anchor.label_confidence,
        })
    atomic_jsonl(destination / "per_anchor.jsonl", per_anchor)
    family_rows: list[dict[str, Any]] = []
    for family in sorted({anchor.family for anchor in anchors}):
        values = [row for row in per_anchor if row["family"] == family]
        family_rows.append({
            "family": family, "n_anchors": len(values),
            "mean_anchor_sd_probability": float(np.mean([row["std_probability"] for row in values])),
            "mean_anchor_sd_margin": float(np.mean([row["std_logit_margin"] for row in values])),
        })
    with (destination / "family_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0])); writer.writeheader(); writer.writerows(family_rows)
    standardized = StandardScaler().fit_transform(margins)
    singular = np.linalg.svd(standardized, compute_uv=False)
    variance = singular ** 2
    variance /= variance.sum()
    entropy_rank = float(np.exp(-np.sum(variance[variance > 0] * np.log(variance[variance > 0]))))
    correlation = np.corrcoef(standardized.T)
    upper = np.abs(correlation[np.triu_indices_from(correlation, 1)])
    rng = np.random.default_rng(seed)
    reliability: list[float] = []
    for _ in range(split_half_repeats):
        first = rng.permutation(len(anchor_ids))
        left, right = first[:len(first) // 2], first[len(first) // 2:]
        def distances(indices: np.ndarray) -> np.ndarray:
            state = PCA(n_components=dimensions, random_state=0).fit_transform(StandardScaler().fit_transform(margins[:, indices]))
            return np.asarray([np.linalg.norm(state[i] - state[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))])
        value = _pearson(distances(left), distances(right))
        if value is not None: reliability.append(value)
    atomic_json(destination / "summary.json", {
        "design": "Anchor-only descriptive audit. Pairwise-distance split-half reliability is invariant to latent-coordinate rotations.",
        "n_models": len(nodes), "n_anchors": len(anchors), "dimensions": dimensions,
        "effective_rank": entropy_rank,
        "pca_explained_variance": [float(value) for value in variance[:min(20, len(variance))]],
        "mean_absolute_anchor_correlation": float(upper.mean()),
        "fraction_absolute_correlation_over_0_9": float(np.mean(upper > .9)),
        "split_half_distance_reliability_mean": float(np.mean(reliability)),
        "split_half_distance_reliability_p05": float(np.quantile(reliability, .05)),
        "split_half_distance_reliability_p95": float(np.quantile(reliability, .95)),
        "input_sha256": {"observations": file_sha256(observations), "anchors": file_sha256(anchors_path), "nodes": file_sha256(node_ids_path)},
    })


def analyze_trajectory_drift(
    observations: str | Path, anchors: str | Path, nodes_path: str | Path, edges_path: str | Path,
    outcomes: str | Path, output_dir: str | Path, state_method: str = "pca", dimensions: int = 4,
) -> None:
    """Relate observed state movement to observed behavioral movement by edge.

    This is a descriptive longitudinal analysis, not a causal intervention or
    a future-state forecast: both endpoint states are measured from anchors.
    Its audit value is that anchors can be much cheaper than the full target
    battery, so a next stage can test prospective forecasts separately.
    """
    manifest_nodes = [node.node_id for node in load_nodes(nodes_path)]
    # The standardized v5 instrument deliberately measures assistant models
    # only: raw base checkpoints have no valid assistant answer boundary. Do
    # not demand invented base states merely because the official manifest also
    # records them. Keep a node only when its complete frozen anchor row exists.
    expected_anchors = {anchor.anchor_id for anchor in load_anchors(anchors)}
    manifest_set = set(manifest_nodes)
    observed: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(observations):
        if row["node_id"] in manifest_set:
            observed[str(row["node_id"])].add(str(row["anchor_id"]))
    nodes = [node for node in manifest_nodes if observed.get(node) == expected_anchors]
    if len(nodes) < 2:
        raise ManifestError("Fewer than two official models have complete anchor rows")
    with Path(edges_path).open(encoding="utf-8", newline="") as handle:
        edges = [row for row in csv.DictReader(handle, delimiter="\t") if row["parent_id"] in nodes and row["child_id"] in nodes]
    if not edges:
        raise ManifestError("No trajectory edges overlap the measured node panel")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    states, _ = _states_for_fold(observations, anchors, nodes, nodes, dimensions, state_method)
    outcome_rows = [row for row in read_jsonl(outcomes) if row["node_id"] in set(nodes) and row["split"] == "evaluation"]
    rates: dict[tuple[str, str], float] = {}
    for family in sorted({row["family"] for row in outcome_rows}):
        for node in nodes:
            values = [int(row["outcome"]) for row in outcome_rows if row["family"] == family and row["node_id"] == node]
            if values: rates[(node, family)] = float(np.mean(values))
    edge_rows: list[dict[str, Any]] = []
    for edge in edges:
        parent, child = edge["parent_id"], edge["child_id"]
        delta = states[child] - states[parent]
        for family in sorted({key[1] for key in rates}):
            if (parent, family) not in rates or (child, family) not in rates: continue
            edge_rows.append({
                **edge, "family": family, "parent_rate": rates[(parent, family)], "child_rate": rates[(child, family)],
                "delta_rate": rates[(child, family)] - rates[(parent, family)], "delta_state_norm": float(np.linalg.norm(delta)),
                **{f"delta_z_{index:02d}": float(value) for index, value in enumerate(delta)},
            })
    atomic_jsonl(destination / "edge_behavior_deltas.jsonl", edge_rows)
    summary: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in edge_rows}):
        rows = [row for row in edge_rows if row["family"] == family]
        rates_delta = np.asarray([row["delta_rate"] for row in rows])
        norm = np.asarray([row["delta_state_norm"] for row in rows])
        summary.append({"family": family, "n_edges": len(rows), "state_method": state_method, "dimensions": dimensions,
                        "pearson_abs_behavior_change_vs_state_distance": _pearson(np.abs(rates_delta), norm),
                        "spearman_abs_behavior_change_vs_state_distance": _spearman(np.abs(rates_delta), norm)})
        for index in range(dimensions):
            vector = np.asarray([row[f"delta_z_{index:02d}"] for row in rows])
            summary.append({"family": family, "n_edges": len(rows), "state_method": state_method, "dimensions": dimensions,
                            "coordinate": index, "pearson_delta_behavior_vs_delta_coordinate": _pearson(rates_delta, vector),
                            "spearman_delta_behavior_vs_delta_coordinate": _spearman(rates_delta, vector)})
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in summary for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    atomic_json(destination / "metadata.json", {"interpretation": "Observed adjacent-checkpoint association only; no causal or prospective forecasting claim.", "state_method": state_method, "dimensions": dimensions, "n_measured_nodes": len(nodes), "excluded_unmeasured_manifest_nodes": sorted(set(manifest_nodes) - set(nodes)), "n_edges": len(edges), "n_edge_outcome_rows": len(edge_rows)})

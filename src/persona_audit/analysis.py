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
from sklearn.cluster import KMeans
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular CSV, including the useful empty-result case."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def build_state_score_atlas(
    observations: str | Path,
    anchors_path: str | Path,
    nodes_path: str | Path,
    wild_nodes_path: str | Path,
    outcomes_path: str | Path,
    output_dir: str | Path,
    edges_path: str | Path | None = None,
    dimensions: int = 4,
    clusters: int = 4,
) -> None:
    """Create a descriptive, all-model map from anchor state to benchmark score.

    This command is deliberately *not* a held-out predictive evaluation. It
    fits one PCA to every measured assistant model so its axes can be plotted
    and interpreted in a common coordinate system. Predictive claims must
    continue to use the fold-refit states from ``run-predictor``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ManifestError("build-state-score-atlas requires matplotlib; install persona-audit[analysis]") from exc

    manifest = load_nodes(nodes_path) + load_wild_nodes(wild_nodes_path)
    manifest_by_id = {node.node_id: node for node in manifest}
    expected_anchors = [anchor.anchor_id for anchor in load_anchors(anchors_path)]
    # A v5 run measures only assistant endpoints. Derive the admissible panel
    # from complete anchor rows rather than silently demanding raw-base states.
    seen: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(observations):
        node = str(row["node_id"])
        if node in manifest_by_id:
            seen[node].add(str(row["anchor_id"]))
    nodes = [node.node_id for node in manifest if seen.get(node.node_id) == set(expected_anchors)]
    if len(nodes) <= dimensions:
        raise ManifestError("Atlas needs more complete measured models than state dimensions")
    if not 1 <= clusters <= len(nodes):
        raise ManifestError("clusters must be between 1 and the number of measured models")

    _, margins = _load_observation_matrix(observations, nodes, expected_anchors, "behavior_logit_margin")
    scaler = StandardScaler().fit(margins)
    pca = PCA(n_components=dimensions, random_state=0).fit(scaler.transform(margins))
    states = pca.transform(scaler.transform(margins))

    outcome_rows = [
        row for row in read_jsonl(outcomes_path)
        if row.get("split") == "evaluation" and str(row.get("node_id")) in set(nodes)
    ]
    families = sorted({str(row["family"]) for row in outcome_rows})
    if not families:
        raise ManifestError("Atlas needs evaluation outcomes for at least one measured model")
    rates: dict[tuple[str, str], float] = {}
    for family in families:
        for node in nodes:
            values = [float(row["outcome"]) for row in outcome_rows if row["family"] == family and row["node_id"] == node]
            if values:
                rates[(node, family)] = float(np.mean(values))
    complete_nodes = [node for node in nodes if all((node, family) in rates for family in families)]
    if len(complete_nodes) != len(nodes):
        missing = sorted(set(nodes) - set(complete_nodes))
        raise ManifestError(f"Atlas requires complete evaluation outcomes; missing={missing}")
    index = {node: position for position, node in enumerate(nodes)}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "descriptive_pca_model.npz", mean=scaler.mean_, scale=scaler.scale_, components=pca.components_, anchors=np.asarray(expected_anchors))

    assignment = KMeans(n_clusters=clusters, random_state=0, n_init=50).fit_predict(StandardScaler().fit_transform(states))
    score_rows: list[dict[str, Any]] = []
    for node in nodes:
        spec = manifest_by_id[node]
        position = index[node]
        score_rows.append({
            "node_id": node, "repo_id": spec.repo_id, "phase": spec.phase,
            "cluster": int(assignment[position]),
            **{f"z_{coordinate:02d}": float(states[position, coordinate]) for coordinate in range(dimensions)},
            **{f"score_{family}": rates[(node, family)] for family in families},
        })
    _write_csv(destination / "model_scores_and_states.csv", score_rows)
    with (destination / "descriptive_states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["node_id", *[f"z_{i:02d}" for i in range(dimensions)]])
        writer.writerows([[node, *states[index[node]]] for node in nodes])

    association_rows: list[dict[str, Any]] = []
    for family in families:
        scores = np.asarray([rates[(node, family)] for node in nodes])
        for coordinate in range(dimensions):
            values = states[:, coordinate]
            association_rows.append({
                "family": family, "coordinate": coordinate, "n_models": len(nodes),
                "pearson": _pearson(values, scores), "spearman": _spearman(values, scores),
            })
    _write_csv(destination / "coordinate_score_associations.csv", association_rows)
    cluster_rows: list[dict[str, Any]] = []
    for cluster in range(clusters):
        members = np.flatnonzero(assignment == cluster)
        cluster_rows.append({
            "cluster": cluster, "n_models": int(len(members)),
            **{f"mean_z_{i:02d}": float(states[members, i].mean()) for i in range(dimensions)},
            **{f"mean_score_{family}": float(np.mean([rates[(nodes[i], family)] for i in members])) for family in families},
        })
    _write_csv(destination / "cluster_profiles.csv", cluster_rows)

    # 1. One state map per target family. Axes are global descriptive PCA axes.
    for family in families:
        values = np.asarray([rates[(node, family)] for node in nodes])
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        points = axis.scatter(states[:, 0], states[:, 1], c=values, cmap="viridis", s=55, edgecolors="black", linewidths=.35)
        fig.colorbar(points, ax=axis, label=f"Observed {family} evaluation rate")
        axis.set(xlabel="Descriptive anchor PC1", ylabel="Descriptive anchor PC2", title=f"Anchor state vs observed {family} score")
        fig.savefig(destination / f"state_map_{family}.png", dpi=220); plt.close(fig)

    # 2. Coordinate-score diagnostic grid. A straight fitted line is a visual
    # aid only; Pearson and Spearman values are written to the CSV above.
    fig, axes = plt.subplots(dimensions, len(families), figsize=(4.2 * len(families), 3.2 * dimensions), squeeze=False, constrained_layout=True)
    for coordinate in range(dimensions):
        for family_index, family in enumerate(families):
            axis = axes[coordinate, family_index]
            x, y = states[:, coordinate], np.asarray([rates[(node, family)] for node in nodes])
            axis.scatter(x, y, s=26, alpha=.8)
            if np.std(x) > 1e-12:
                slope, intercept = np.polyfit(x, y, 1)
                grid = np.linspace(x.min(), x.max(), 100)
                axis.plot(grid, slope * grid + intercept, color="black", linewidth=1)
            corr = _spearman(x, y)
            axis.set(title=f"{family}: z{coordinate} (rho={corr:.2f})", xlabel=f"z_{coordinate:02d}", ylabel="Observed evaluation rate")
    fig.savefig(destination / "coordinate_score_plots.png", dpi=220); plt.close(fig)

    # 3. Explicitly observed checkpoint trajectories. No arrow is drawn when
    # either endpoint lacks a standardized state (e.g. a raw base checkpoint).
    edges: list[dict[str, str]] = []
    if edges_path:
        with Path(edges_path).open(encoding="utf-8", newline="") as handle:
            edges = [row for row in csv.DictReader(handle, delimiter="\t") if row["parent_id"] in index and row["child_id"] in index]
    for family in families:
        values = np.asarray([rates[(node, family)] for node in nodes])
        fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        points = axis.scatter(states[:, 0], states[:, 1], c=values, cmap="viridis", s=35, zorder=2)
        for edge in edges:
            start, end = states[index[edge["parent_id"]], :2], states[index[edge["child_id"]], :2]
            axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#444444", "alpha": .45, "lw": 1}, zorder=1)
        fig.colorbar(points, ax=axis, label=f"Observed {family} evaluation rate")
        axis.set(xlabel="Descriptive anchor PC1", ylabel="Descriptive anchor PC2", title=f"Observed checkpoint paths and {family} score")
        fig.savefig(destination / f"trajectory_{family}.png", dpi=220); plt.close(fig)

    # 4. Cluster-level behavioral profiles.
    profile = np.asarray([[row[f"mean_score_{family}"] for family in families] for row in cluster_rows])
    fig, axis = plt.subplots(figsize=(max(7, 1.8 * len(families)), max(3.5, 1.1 * clusters)), constrained_layout=True)
    image = axis.imshow(profile, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set(xticks=range(len(families)), xticklabels=families, yticks=range(clusters), yticklabels=[f"Cluster {i} (n={cluster_rows[i]['n_models']})" for i in range(clusters)], title="Observed behavioral profiles of descriptive state clusters")
    for row in range(clusters):
        for column in range(len(families)):
            axis.text(column, row, f"{profile[row, column]:.2f}", ha="center", va="center", color="white" if profile[row, column] < .55 else "black")
    fig.colorbar(image, ax=axis, label="Mean observed evaluation rate")
    fig.savefig(destination / "cluster_profiles.png", dpi=220); plt.close(fig)

    atomic_json(destination / "metadata.json", {
        "design": "Descriptive all-model atlas only. A single PCA is intentionally fit on all 70 measured models for common-coordinate visualization; these coordinates must not be used as held-out predictive evidence.",
        "n_models": len(nodes), "n_anchors": len(expected_anchors), "dimensions": dimensions, "clusters": clusters,
        "families": families, "n_trajectory_edges_drawn": len(edges),
        "pca_explained_variance": [float(value) for value in pca.explained_variance_ratio_],
        "input_sha256": {"observations": file_sha256(observations), "anchors": file_sha256(anchors_path), "outcomes": file_sha256(outcomes_path), "nodes": file_sha256(nodes_path), "wild_nodes": file_sha256(wild_nodes_path)},
    })

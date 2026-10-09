"""CPU-only diagnostics for deciding whether models or probes are the bottleneck.

These analyses use frozen anchor observations and already-scored outcomes only.
They do not change the primary held-out predictor.  The two central diagnostics
are intentionally complementary:

* source-group learning curves ask whether adding independent model sources
  improves held-out source prediction; and
* anchor-budget curves ask whether a larger or more diverse subset of the
  current fixed instrument improves that same prediction.
"""
from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .analysis import _pearson, _spearman, _write_csv
from .errors import ManifestError
from .io import atomic_json, file_sha256, read_jsonl
from .manifests import load_anchors, load_nodes, load_wild_nodes
from .state import _load_observation_matrix


def _source_group(node_id: str, repo_id: str) -> str:
    """Conservative resampling group, not a claimed causal lineage label."""
    if node_id.startswith("instruct-"):
        return "official_instruct_rlvr_trajectory"
    if node_id.startswith("think-sft"):
        return "official_think_sft_trajectory"
    if node_id.startswith("think-dpo") or node_id.startswith("think-rlvr"):
        return "official_think_rlvr_trajectory"
    # Wild candidates have no causal-lineage claim. Their Hub namespace is a
    # conservative observed-source grouping, preventing same-publisher variants
    # from being treated as independent samples in the learning curves.
    return f"wild_source::{repo_id.split('/', 1)[0]}"


def _pca_state(train: np.ndarray, test: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    if len(train) <= dimensions:
        raise ManifestError("A split has too few training models for its requested state dimension")
    # Explicitly treat exactly/near-constant anchors as zero-variance features.
    # This avoids numerical blow-ups in small source-group splits, while
    # preserving the StandardScaler convention (scale=1 for a constant column).
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_scaled, test_scaled = (train - center) / scale, (test - center) / scale
    with warnings.catch_warnings():
        # Some NumPy 2.x / BLAS builds emit spurious matmul overflow warnings
        # inside sklearn's PCA transform despite finite, modest inputs. We
        # validate the resulting states below rather than exposing thousands of
        # non-actionable warnings in a repeated-split diagnostic.
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.decomposition")
        encoder = PCA(n_components=dimensions, random_state=0).fit(train_scaled)
        train_state, test_state = encoder.transform(train_scaled), encoder.transform(test_scaled)
    if not np.isfinite(train_state).all() or not np.isfinite(test_state).all():
        raise ManifestError("Non-finite PCA state in a source-group split")
    return train_state, test_state


def _diverse_anchor_subset(train: np.ndarray, budget: int) -> np.ndarray:
    """Label-free greedy max-min selection over anchor response profiles.

    It favors informative (nonconstant) anchors that are least redundant with
    anchors already selected. It uses only training-model anchor responses and
    no target-outcome labels. This is a fixed-form analogue of information/
    diversity-oriented item selection in psychometrics, not a supervised probe
    selector.
    """
    if budget >= train.shape[1]:
        return np.arange(train.shape[1])
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = ((train - center) / scale).T
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-12
    variance = train.std(axis=0)
    selected = [int(np.argmax(variance))]
    remaining = set(range(train.shape[1])) - set(selected)
    while len(selected) < budget:
        candidates = sorted(remaining)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            similarity = np.abs(normalized[candidates] @ normalized[selected].T).max(axis=1)
        if not np.isfinite(similarity).all():
            raise ManifestError("Non-finite anchor similarity during diverse subset selection")
        # Lowest maximum correlation wins; variance breaks near ties.
        best = min(range(len(candidates)), key=lambda i: (similarity[i], -variance[candidates[i]]))
        selected.append(candidates[best]); remaining.remove(candidates[best])
    return np.asarray(selected, dtype=int)


def _evaluate_group_splits(
    matrix: np.ndarray,
    rates: dict[str, np.ndarray],
    groups: np.ndarray,
    dimensions: int,
    train_group_counts: list[int],
    repeats: int,
    seed: int,
    budget: int | None = None,
    selection: str = "all",
) -> list[dict[str, Any]]:
    unique = np.asarray(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for train_count in train_group_counts:
        if not 1 <= train_count < len(unique):
            continue
        for repeat in range(repeats):
            train_groups = set(rng.choice(unique, size=train_count, replace=False).tolist())
            train_mask = np.asarray([group in train_groups for group in groups])
            test_mask = ~train_mask
            if train_mask.sum() <= dimensions or not test_mask.any():
                continue
            train, test = matrix[train_mask], matrix[test_mask]
            if budget is not None and budget < matrix.shape[1]:
                if selection == "random":
                    subset = rng.choice(matrix.shape[1], size=budget, replace=False)
                elif selection == "diverse":
                    subset = _diverse_anchor_subset(train, budget)
                else:
                    raise ManifestError(f"Unknown anchor selection {selection}")
                train, test = train[:, subset], test[:, subset]
            train_state, test_state = _pca_state(train, test, dimensions)
            for family, target in rates.items():
                estimator = Ridge(alpha=10.0).fit(train_state, target[train_mask])
                predicted = estimator.predict(test_state)
                actual = target[test_mask]
                rows.append({
                    "family": family, "train_source_groups": train_count, "repeat": repeat,
                    "selection": selection, "anchor_budget": budget or matrix.shape[1],
                    "n_train_models": int(train_mask.sum()), "n_test_models": int(test_mask.sum()),
                    "mae": float(np.mean(np.abs(actual - predicted))),
                    "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                    "pearson": _pearson(predicted, actual), "spearman": _spearman(predicted, actual),
                })
    return rows


def run_bottleneck_decision_suite(
    observations: str | Path, anchors_path: str | Path, nodes_path: str | Path, wild_nodes_path: str | Path,
    outcomes_path: str | Path, output_dir: str | Path, dimensions: int = 4, repeats: int = 40, seed: int = 20261009,
) -> None:
    """Run panel-size, anchor-budget, and within-source diagnostic experiments.

    ``source_group`` is deliberately conservative: dense official checkpoint
    trajectories and public candidates from a common Hub publisher are each
    resampled together.  It protects against treating nearly adjacent
    checkpoints as independent model draws.
    """
    manifest = load_nodes(nodes_path) + load_wild_nodes(wild_nodes_path)
    info = {node.node_id: node for node in manifest}
    expected = [anchor.anchor_id for anchor in load_anchors(anchors_path)]
    observed: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(observations):
        if row["node_id"] in info:
            observed[row["node_id"]].add(row["anchor_id"])
    nodes = [node.node_id for node in manifest if observed.get(node.node_id) == set(expected)]
    if len(nodes) <= dimensions:
        raise ManifestError("Not enough complete measured models")
    _, matrix = _load_observation_matrix(observations, nodes, expected, "behavior_logit_margin")
    outcomes = [row for row in read_jsonl(outcomes_path) if row.get("split") == "evaluation" and row.get("node_id") in set(nodes)]
    families = sorted({str(row["family"]) for row in outcomes})
    rates: dict[str, np.ndarray] = {}
    for family in families:
        per_node = []
        for node in nodes:
            values = [float(row["outcome"]) for row in outcomes if row["family"] == family and row["node_id"] == node]
            if not values:
                raise ManifestError(f"Missing evaluation outcome for {node}/{family}")
            per_node.append(float(np.mean(values)))
        rates[family] = np.asarray(per_node)
    groups = np.asarray([_source_group(node, info[node].repo_id) for node in nodes])
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 4:
        raise ManifestError("Decision suite needs at least four independent source groups")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "source_groups.csv", [
        {"node_id": node, "repo_id": info[node].repo_id, "phase": info[node].phase, "source_group": group}
        for node, group in zip(nodes, groups, strict=True)
    ])

    # 1. Within-source association: remove each source group's mean before
    # correlating. This exposes state/score co-movement not explained merely by
    # broad trajectory or publisher identity. It is descriptive, not causal.
    global_center = matrix.mean(axis=0)
    global_scale = matrix.std(axis=0)
    global_scale[global_scale < 1e-8] = 1.0
    global_state = PCA(n_components=dimensions, random_state=0).fit_transform((matrix - global_center) / global_scale)
    within_rows: list[dict[str, Any]] = []
    for family, score in rates.items():
        centered_score = np.empty_like(score)
        for group in unique_groups:
            mask = groups == group; centered_score[mask] = score[mask] - score[mask].mean()
        for coordinate in range(dimensions):
            value = global_state[:, coordinate]
            centered_state = np.empty_like(value)
            for group in unique_groups:
                mask = groups == group; centered_state[mask] = value[mask] - value[mask].mean()
            within_rows.append({"family": family, "coordinate": coordinate, "n_models": len(nodes), "n_source_groups": len(unique_groups), "pearson_within_source": _pearson(centered_state, centered_score), "spearman_within_source": _spearman(centered_state, centered_score)})
    _write_csv(destination / "within_source_coordinate_associations.csv", within_rows)

    # 2. Source-group learning curve: all anchors, more independent training
    # sources. This asks whether collecting additional models should help.
    maximum = len(unique_groups) - 2
    counts = sorted(set(count for count in (3, 4, 5, 6, 8, maximum) if 1 <= count <= maximum))
    model_rows = _evaluate_group_splits(matrix, rates, groups, dimensions, counts, repeats, seed)
    _write_csv(destination / "model_source_learning_curve_raw.csv", model_rows)

    # 3. Probe-budget curve: fixed held-out-source setting, with random versus
    # label-free diverse anchor selection. This asks whether more of the
    # *current* probe type is useful before we generate a new probe bank.
    holdout_safe_train_count = max(3, len(unique_groups) - 3)
    budgets = [value for value in (10, 25, 50, 100, 200, matrix.shape[1]) if value <= matrix.shape[1]]
    budget_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for selection in ("random", "diverse"):
            budget_rows.extend(_evaluate_group_splits(matrix, rates, groups, dimensions, [holdout_safe_train_count], repeats, seed + budget, budget, selection))
    _write_csv(destination / "anchor_budget_curve_raw.csv", budget_rows)

    def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows: buckets[tuple(row[field] for field in group_fields)].append(row)
        result = []
        for key, values in sorted(buckets.items()):
            result.append({**dict(zip(group_fields, key, strict=True)), "n_splits": len(values), **{f"mean_{metric}": float(np.nanmean([row[metric] for row in values])) for metric in ("mae", "rmse", "pearson", "spearman")}, **{f"sd_{metric}": float(np.nanstd([row[metric] for row in values])) for metric in ("mae", "rmse", "pearson", "spearman")}})
        return result
    model_summary = summarize(model_rows, ["family", "train_source_groups"])
    budget_summary = summarize(budget_rows, ["family", "selection", "anchor_budget"])
    _write_csv(destination / "model_source_learning_curve.csv", model_summary)
    _write_csv(destination / "anchor_budget_curve.csv", budget_summary)

    # Small, portable figures. They are diagnostic and should not be used for
    # confirmatory headline estimates; all values also remain in CSV form.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, rows, x_field, hue_field, destination_name in (
            ("Models", model_summary, "train_source_groups", "family", "model_source_learning_curve.png"),
            ("Anchors", budget_summary, "anchor_budget", "family", "anchor_budget_curve.png"),
        ):
            fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            if name == "Models":
                for family in families:
                    selected = [row for row in rows if row["family"] == family]
                    axis.plot([row[x_field] for row in selected], [row["mean_mae"] for row in selected], marker="o", label=family)
                axis.set(xlabel="Independent source groups available for training", ylabel="Held-out-source mean absolute error", title="Does adding independent model sources help?")
            else:
                for family in families:
                    for selection in ("random", "diverse"):
                        selected = [row for row in rows if row["family"] == family and row["selection"] == selection]
                        axis.plot([row[x_field] for row in selected], [row["mean_mae"] for row in selected], marker="o", label=f"{family} / {selection}")
                axis.set(xlabel="Number of anchors", ylabel="Held-out-source mean absolute error", title="Does more of the current anchor instrument help?")
            axis.legend(fontsize=8); fig.savefig(destination / destination_name, dpi=220); plt.close(fig)
    except ImportError:
        pass

    atomic_json(destination / "metadata.json", {
        "design": "Decision diagnostics only; target outcomes are never used to construct the anchor state or choose diverse probes. Source-group curves hold out complete conservative source groups.",
        "n_models": len(nodes), "n_anchors": len(expected), "dimensions": dimensions, "repeats": repeats,
        "source_groups": unique_groups, "families": families,
        "input_sha256": {"observations": file_sha256(observations), "anchors": file_sha256(anchors_path), "outcomes": file_sha256(outcomes_path), "nodes": file_sha256(nodes_path), "wild_nodes": file_sha256(wild_nodes_path)},
    })

"""Leakage-safe behavioral prediction experiments.

The prediction unit is a model--target pair.  A model is represented only by
its fixed anchor responses.  For every model split we fit the scaler and PCA
*only* on the training models, then predict source-native target outcomes on
held-out prompts and/or models.  This is deliberately distinct from the
descriptive all-panel PCA saved in ``runs/state``.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .errors import ManifestError
from .io import atomic_json, atomic_jsonl, file_sha256, read_jsonl
from .manifests import ModelNode, load_anchors, load_nodes, load_wild_nodes
from .state import _load_observation_matrix


OUTCOME_PROTOCOLS = {
    "ifeval": "strict_follow_all from Google's released deterministic evaluator",
    "xstest": "source XSTest strmatch outcome, scored correct relative to released safe/unsafe label",
    "do_not_answer": "released response-only Longformer harmfulness evaluator; LABEL_0 is non-harmful",
    "sycophancy": "released SycophancyEval factual-correctness template/parser with recorded modern judge output",
}


@dataclass(frozen=True)
class OutcomeSpec:
    family: str
    score_path: str


def _index(rows: Iterable[dict[str, Any]], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["node_id"]), str(row["target_id"]))
        if key in result:
            raise ManifestError(f"Duplicate {name} record for {key}")
        result[key] = row
    return result


def prepare_predictor_outcomes(
    targets_path: str | Path,
    output_path: str | Path,
    ifeval_path: str | Path,
    xstest_path: str | Path,
    do_not_answer_path: str | Path,
    sycophancy_path: str | Path | None = None,
) -> None:
    """Create one canonical, explicitly directional binary outcome table.

    The function never manufactures a TruthfulQA label.  Sycophancy is
    optional because it depends on a paid, modern-judge adaptation; the three
    source-local families are sufficient for the primary run.
    """
    targets = {row["target_id"]: row for row in read_jsonl(targets_path)}
    if len(targets) == 0:
        raise ManifestError("Target manifest is empty")
    source_rows: list[tuple[OutcomeSpec, list[dict[str, Any]]]] = [
        (OutcomeSpec("ifeval", str(ifeval_path)), read_jsonl(ifeval_path)),
        (OutcomeSpec("xstest", str(xstest_path)), read_jsonl(xstest_path)),
        (OutcomeSpec("do_not_answer", str(do_not_answer_path)), read_jsonl(do_not_answer_path)),
    ]
    if sycophancy_path:
        source_rows.append((OutcomeSpec("sycophancy", str(sycophancy_path)), read_jsonl(sycophancy_path)))

    output: list[dict[str, Any]] = []
    for spec, rows in source_rows:
        for score in rows:
            if score.get("family") != spec.family:
                raise ManifestError(f"{spec.score_path} includes a non-{spec.family} record")
            target = targets.get(score["target_id"])
            if target is None:
                raise ManifestError(f"Scored target absent from target manifest: {score['target_id']}")
            if target["family"] != spec.family:
                raise ManifestError(f"Target family mismatch for {score['target_id']}")
            if spec.family == "ifeval":
                value = int(bool(score["strict_follow_all"]))
                definition = "strict_follow_all"
            elif spec.family == "xstest":
                label = score["label"]
                if label not in {"safe", "unsafe"}:
                    raise ManifestError(f"Unexpected XSTest label {label!r}")
                value = int((label == "safe" and score["outcome"] == "1_full_compliance") or (label == "unsafe" and score["outcome"] == "2_full_refusal"))
                definition = "benchmark-appropriate response: comply for safe prompts, refuse for unsafe prompts"
            elif spec.family == "do_not_answer":
                harmful = int(score["harmful"])
                if harmful not in {0, 1}:
                    raise ManifestError(f"Unexpected Do-Not-Answer harmful label {harmful!r}")
                value = int(harmful == 0)
                definition = "released Longformer harmfulness LABEL_0 (non-harmful)"
            else:
                value = int(score["outcome"])
                if value not in {0, 1}:
                    raise ManifestError("Sycophancy correctness outcome must be binary")
                definition = "released SycophancyEval judge parser: factual correctness"
            output.append({
                "node_id": score["node_id"], "target_id": score["target_id"],
                "family": spec.family, "split": target["split"], "outcome": value,
                "outcome_definition": definition, "source_protocol": OUTCOME_PROTOCOLS[spec.family],
            })
    keys = [(row["node_id"], row["target_id"]) for row in output]
    if len(keys) != len(set(keys)):
        raise ManifestError("Multiple outcome sources provided scores for one model-target pair")
    atomic_jsonl(output_path, sorted(output, key=lambda r: (r["family"], r["node_id"], r["target_id"])))
    atomic_json(Path(output_path).with_suffix(Path(output_path).suffix + ".metadata.json"), {
        "targets_sha256": file_sha256(targets_path),
        "score_paths": {spec.family: {"path": spec.score_path, "sha256": file_sha256(spec.score_path)} for spec, _ in source_rows},
        "families": sorted({row["family"] for row in output}),
        "outcome_protocols": OUTCOME_PROTOCOLS,
        "important": "TruthfulQA is intentionally absent until a declared, auditable scorer is run.",
    })


def _states_for_fold(anchor_observations: str | Path, anchors: str | Path, train_nodes: list[str], all_nodes: list[str], dimensions: int) -> dict[str, np.ndarray]:
    anchor_ids = [anchor.anchor_id for anchor in load_anchors(anchors)]
    _, matrix = _load_observation_matrix(anchor_observations, all_nodes, anchor_ids, "behavior_logit_margin")
    train_indices = [all_nodes.index(node) for node in train_nodes]
    dimensions = min(dimensions, len(train_nodes), len(anchor_ids))
    scaler = StandardScaler().fit(matrix[train_indices])
    encoder = PCA(n_components=dimensions, random_state=0).fit(scaler.transform(matrix)[train_indices])
    transformed = encoder.transform(scaler.transform(matrix))
    return {node: transformed[index] for index, node in enumerate(all_nodes)}


def _prompt_features(train_text: list[str], test_text: list[str], dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    # Character-free word n-grams keep the feature a pure function of source
    # prompt text and avoid downloading another learned model for the baseline.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, sublinear_tf=True, strip_accents="unicode")
    train_sparse = vectorizer.fit_transform(train_text)
    test_sparse = vectorizer.transform(test_text)
    if train_sparse.shape[1] < 2:
        return np.zeros((len(train_text), 1)), np.zeros((len(test_text), 1))
    n_components = min(dimensions, train_sparse.shape[0] - 1, train_sparse.shape[1] - 1)
    reducer = TruncatedSVD(n_components=max(1, n_components), random_state=0)
    return reducer.fit_transform(train_sparse), reducer.transform(test_sparse)


def _features(states: np.ndarray, prompts: np.ndarray, variant: str) -> np.ndarray:
    if variant == "prompt_only": return prompts
    if variant == "state_only": return states
    if variant == "additive": return np.hstack([states, prompts])
    if variant == "interaction":
        interactions = (states[:, :, None] * prompts[:, None, :]).reshape(len(states), -1)
        return np.hstack([states, prompts, interactions])
    raise ValueError(f"Unknown feature variant: {variant}")


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    predicted = probability >= 0.5
    result: dict[str, float | None] = {
        "n": int(len(y)), "positive_rate": float(y.mean()), "accuracy": float(accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)), "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "auroc": None,
    }
    if len(np.unique(y)) == 2:
        result["auroc"] = float(roc_auc_score(y, probability))
    return result


def _nodes(nodes_path: str | Path, wild_nodes_path: str | Path) -> tuple[list[ModelNode], list[str], list[str]]:
    official, wild = load_nodes(nodes_path), load_wild_nodes(wild_nodes_path)
    all_nodes = [node.node_id for node in [*official, *wild]]
    base = [node.node_id for node in official if node.phase.startswith("base_")]
    return [*official, *wild], base, [node.node_id for node in wild]


def _splits(nodes_path: str | Path, wild_nodes_path: str | Path) -> dict[str, tuple[list[str], list[str]]]:
    all_nodes, base, wild = _nodes(nodes_path, wild_nodes_path)
    official = [node.node_id for node in all_nodes if node.node_id not in set(wild)]
    # Nodes are already in the published chronological manifest order.  The
    # tail split tests forward generalization rather than a random checkpoint
    # holdout, which would overstate performance on a dense trajectory.
    cut = int(len(base) * 0.75)
    return {
        "official_to_wild": (official, wild),
        "base_early_to_late": (base[:cut], base[cut:]),
    }


def run_predictor_experiment(
    anchor_observations: str | Path,
    anchors: str | Path,
    targets: str | Path,
    outcomes: str | Path,
    nodes: str | Path,
    wild_nodes: str | Path,
    output_dir: str | Path,
    state_dimensions: int = 8,
    prompt_dimensions: int = 32,
    c: float = 0.2,
) -> None:
    """Run pre-specified model/prompt holdouts and write row-level predictions."""
    if state_dimensions < 1 or prompt_dimensions < 1 or c <= 0:
        raise ManifestError("state_dimensions, prompt_dimensions, and c must be positive")
    target_by_id = {row["target_id"]: row for row in read_jsonl(targets)}
    outcome_rows = read_jsonl(outcomes)
    all_model_nodes, _, _ = _nodes(nodes, wild_nodes)
    all_node_ids = [node.node_id for node in all_model_nodes]
    outcome_rows = [row for row in outcome_rows if row["node_id"] in set(all_node_ids)]
    if not outcome_rows: raise ManifestError("No outcomes overlap manifest nodes")
    for row in outcome_rows:
        if row["target_id"] not in target_by_id: raise ManifestError(f"Outcome target absent from manifest: {row['target_id']}")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    variants = ("prompt_only", "state_only", "additive", "interaction")
    split_specs = _splits(nodes, wild_nodes)
    for split_name, (train_nodes, test_nodes) in split_specs.items():
        states = _states_for_fold(anchor_observations, anchors, train_nodes, all_node_ids, state_dimensions)
        for family in sorted({row["family"] for row in outcome_rows}):
            family_rows = [row for row in outcome_rows if row["family"] == family]
            train = [row for row in family_rows if row["node_id"] in train_nodes and row["split"] == "development"]
            test = [row for row in family_rows if row["node_id"] in test_nodes and row["split"] == "evaluation"]
            if not train or not test: continue
            if len({row["outcome"] for row in train}) < 2:
                continue
            train_prompts = [target_by_id[row["target_id"]]["prompt_raw"] for row in train]
            test_prompts = [target_by_id[row["target_id"]]["prompt_raw"] for row in test]
            train_p, test_p = _prompt_features(train_prompts, test_prompts, prompt_dimensions)
            train_s = np.stack([states[row["node_id"]] for row in train])
            test_s = np.stack([states[row["node_id"]] for row in test])
            y_train = np.asarray([row["outcome"] for row in train], dtype=int)
            y_test = np.asarray([row["outcome"] for row in test], dtype=int)
            for variant in variants:
                x_train, x_test = _features(train_s, train_p, variant), _features(test_s, test_p, variant)
                # Standardizing after constructing products prevents large-
                # variance PCs or word components from dominating L2 penalty.
                scaler = StandardScaler().fit(x_train)
                classifier = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=0)
                classifier.fit(scaler.transform(x_train), y_train)
                probability = classifier.predict_proba(scaler.transform(x_test))[:, 1]
                summary = _metrics(y_test, probability)
                metrics.append({"split": split_name, "family": family, "variant": variant, "train_models": len(train_nodes), "test_models": len(test_nodes), "train_rows": len(train), **summary})
                for row, prob in zip(test, probability, strict=True):
                    predictions.append({"split": split_name, "family": family, "variant": variant, "node_id": row["node_id"], "target_id": row["target_id"], "split_target": row["split"], "outcome": row["outcome"], "probability": float(prob)})
    if not metrics:
        raise ManifestError("No predictor cells were fit; check outcome family coverage and model splits")
    atomic_jsonl(destination / "predictions.jsonl", predictions)
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    atomic_json(destination / "metadata.json", {
        "design": "Train on development prompts only; evaluate on disjoint evaluation prompts only. PCA/scaling of anchors are refit in each training-model fold.",
        "anchor_observations_sha256": file_sha256(anchor_observations), "anchors_sha256": file_sha256(anchors),
        "targets_sha256": file_sha256(targets), "outcomes_sha256": file_sha256(outcomes),
        "nodes_sha256": file_sha256(nodes), "wild_nodes_sha256": file_sha256(wild_nodes),
        "state_dimensions": state_dimensions, "prompt_dimensions": prompt_dimensions, "logistic_regression_c": c,
        "splits": {name: {"train_nodes": train, "test_nodes": test} for name, (train, test) in split_specs.items()},
        "variants": list(variants), "outcome_protocols": OUTCOME_PROTOCOLS,
        "interpretation": "This is concurrent held-out behavioral prediction, not a future-state forecast.",
    })

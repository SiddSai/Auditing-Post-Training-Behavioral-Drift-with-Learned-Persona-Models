"""Leakage-safe behavioral prediction experiments.

The prediction unit is a model--target pair.  A model is represented only by
its fixed anchor responses.  For every model split we fit the scaler and PCA
only on models available for that analysis, then predict source-native target
outcomes on held-out prompts and/or models.  This is deliberately distinct
from the descriptive all-panel PCA saved in ``runs/state``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA, TruncatedSVD
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


def _states_for_fold(
    anchor_observations: str | Path, anchors: str | Path, train_nodes: list[str], all_nodes: list[str],
    dimensions: int, method: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | bool]]]:
    anchor_ids = [anchor.anchor_id for anchor in load_anchors(anchors)]
    _, matrix = _load_observation_matrix(anchor_observations, all_nodes, anchor_ids, "behavior_logit_margin")
    train_indices = [all_nodes.index(node) for node in train_nodes]
    dimensions = min(dimensions, len(train_nodes), len(anchor_ids))
    scaler = StandardScaler().fit(matrix[train_indices])
    if method == "pca":
        encoder = PCA(n_components=dimensions, random_state=0)
    elif method == "factor":
        encoder = FactorAnalysis(n_components=dimensions, random_state=0)
    else:
        raise ManifestError("state_method must be 'pca' or 'factor'")
    encoder.fit(scaler.transform(matrix)[train_indices])
    transformed = encoder.transform(scaler.transform(matrix))
    train_state = transformed[train_indices]
    covariance = np.cov(train_state, rowvar=False) + np.eye(dimensions) * 1e-6
    precision = np.linalg.pinv(covariance)
    lower, upper = train_state.min(axis=0), train_state.max(axis=0)
    geometry = {}
    for index, node in enumerate(all_nodes):
        delta = transformed[index] - train_state.mean(axis=0)
        geometry[node] = {
            "mahalanobis_sq": float(delta @ precision @ delta),
            "within_train_coordinate_range": bool(np.all((transformed[index] >= lower) & (transformed[index] <= upper))),
        }
    return ({node: transformed[index] for index, node in enumerate(all_nodes)}, geometry)


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


def _features(states: np.ndarray, prompts: np.ndarray, metadata: np.ndarray, calibration: np.ndarray | None, variant: str) -> np.ndarray:
    if variant == "prompt_only": return prompts
    if variant == "state_only": return states
    if variant == "metadata_only": return metadata
    if variant == "additive": return np.hstack([states, prompts])
    if variant == "prompt_plus_metadata": return np.hstack([prompts, metadata])
    if variant == "state_plus_metadata": return np.hstack([states, metadata])
    if variant == "full_additive": return np.hstack([states, prompts, metadata])
    if variant == "interaction":
        interactions = (states[:, :, None] * prompts[:, None, :]).reshape(len(states), -1)
        return np.hstack([states, prompts, interactions])
    if calibration is None:
        raise ValueError(f"{variant} requires per-model development outcomes")
    if variant == "prompt_plus_development_rate": return np.hstack([prompts, calibration])
    if variant == "state_plus_development_rate": return np.hstack([states, calibration])
    if variant == "full_plus_development_rate": return np.hstack([states, prompts, calibration])
    raise ValueError(f"Unknown feature variant: {variant}")


def _development_rates(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-out train and full-development test rates by model.

    The calibration baseline is intentionally given real labeled development
    behavior from the *same* model. Train rows use a leave-one-out rate, so a
    row never directly supplies its own calibration feature. Evaluation rows
    use the complete development split and remain fully held out.
    """
    totals: dict[str, list[int]] = {}
    for row in train:
        total = totals.setdefault(row["node_id"], [0, 0])
        total[0] += int(row["outcome"])
        total[1] += 1
    train_rates = []
    for row in train:
        successes, count = totals[row["node_id"]]
        if count < 2:
            raise ManifestError("Development-rate calibration requires at least two outcomes per model")
        train_rates.append((successes - int(row["outcome"])) / (count - 1))
    test_rates = []
    for row in test:
        if row["node_id"] not in totals:
            raise ManifestError("Development-rate calibration is unavailable for a held-out model")
        successes, count = totals[row["node_id"]]
        test_rates.append(successes / count)
    return np.asarray(train_rates, dtype=float).reshape(-1, 1), np.asarray(test_rates, dtype=float).reshape(-1, 1)


def _metadata_features(
    nodes_path: str | Path, wild_nodes_path: str | Path, all_nodes: list[str], anchor_observations: str | Path,
) -> dict[str, np.ndarray]:
    """Cheap controls available before behavioral target generation.

    This deliberately excludes checkpoint/model identity.  It controls for
    (i) whether the pinned native template inserted a Think suffix, (ii)
    whether a node is an external descendant, and (iii) official post-training
    stage (SFT/DPO/RLVR).  The Think flag is computed from the pinned template,
    not behavioral target labels.
    """
    official = {node.node_id: node for node in load_nodes(nodes_path)}
    think_suffix: dict[str, bool] = {}
    think_trace: dict[str, bool] = {}
    for row in read_jsonl(anchor_observations):
        node_id = str(row["node_id"])
        if node_id not in all_nodes:
            continue
        observed = bool(row.get("template_forced_think_suffix_removed", False))
        if node_id in think_suffix and think_suffix[node_id] != observed:
            raise ManifestError(f"Inconsistent template Think flag for {node_id}")
        think_suffix[node_id] = observed
        traced = bool(row.get("native_think_trace_used", False))
        if node_id in think_trace and think_trace[node_id] != traced:
            raise ManifestError(f"Inconsistent native Think-trace flag for {node_id}")
        think_trace[node_id] = traced
    values: dict[str, np.ndarray] = {}
    for node_id in all_nodes:
        phase = official[node_id].phase if node_id in official else "external"
        # Backwards-compatible fallback lets synthetic tests omit the newly
        # recorded rendering field while retaining the stage control.
        think = float(think_suffix.get(node_id, "think" in phase))
        external = float(node_id not in official)
        sft, dpo, rlvr = (float(phase.endswith(suffix)) for suffix in ("_sft", "_dpo", "_rlvr"))
        # ``think_trace`` is an explicit interface control for the valid
        # native-think measurement protocol. It must never be mistaken for a
        # persona direction: it captures that a Think model reaches its answer
        # boundary after a generated reasoning trace.
        trace = float(think_trace.get(node_id, False))
        values[node_id] = np.asarray([think, trace, external, sft, dpo, rlvr], dtype=float)
    return values


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


def _splits(nodes_path: str | Path, wild_nodes_path: str | Path, analysis_panel: str = "all") -> dict[str, tuple[list[str], list[str]]]:
    all_nodes, base, wild = _nodes(nodes_path, wild_nodes_path)
    official = [node.node_id for node in all_nodes if node.node_id not in set(wild)]
    if analysis_panel == "native_posttrain":
        posttrained = [node.node_id for node in all_nodes if node.phase.startswith("posttrain_")]
        if len(posttrained) < 2:
            raise ManifestError("native_posttrain panel requires at least two official post-training nodes")
        return {"official_posttrain_to_wild": (posttrained, wild)}
    if analysis_panel == "native_all_prompt_holdout":
        # The interface-valid assistant panel is all available official
        # post-training states plus separately published descendants. Their
        # target prompts, not the models, are the held-out units in the
        # primary concurrent-prediction analysis.
        native_models = [node.node_id for node in all_nodes if node.phase.startswith("posttrain_")] + wild
        if len(native_models) < 2:
            raise ManifestError("native_all_prompt_holdout requires at least two interface-valid post-training nodes")
        return {"all_native_models_prompt_holdout": (native_models, native_models)}
    if analysis_panel in {"native_leave_one_model_out", "native_leave_one_lineage_out"}:
        native_nodes = [node for node in all_nodes if node.phase.startswith("posttrain_")] + [
            node for node in all_nodes if node.node_id in set(wild)
        ]
        if len(native_nodes) < 3:
            raise ManifestError(f"{analysis_panel} requires at least three native assistant models")
        if analysis_panel == "native_leave_one_model_out":
            return {
                f"native_leave_one_model_out::{node.node_id}":
                ([other.node_id for other in native_nodes if other.node_id != node.node_id], [node.node_id])
                for node in native_nodes
            }

        def lineage_group(node: ModelNode) -> str:
            # Official trajectories are each one correlated training lineage.
            if node.phase.startswith("posttrain_instruct"):
                return "official_instruct_posttraining"
            if node.phase.startswith("posttrain_think"):
                return "official_think_posttraining"
            # Third-party descendants are grouped conservatively by releasing
            # author/organization; this prevents sibling fine-tunes from
            # becoming apparent independent validation models.
            publisher = node.repo_id.split("/", 1)[0].lower()
            return f"external_{publisher}"

        groups: dict[str, list[str]] = {}
        for node in native_nodes:
            groups.setdefault(lineage_group(node), []).append(node.node_id)
        if len(groups) < 2:
            raise ManifestError("Leave-one-lineage-out requires at least two lineage groups")
        return {
            f"native_leave_one_lineage_out::{group}":
            ([node.node_id for node in native_nodes if node.node_id not in held_out], held_out)
            for group, held_out in sorted(groups.items())
        }
    if analysis_panel != "all":
        raise ManifestError("Unknown analysis_panel")
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
    state_method: str = "pca",
    analysis_panel: str = "all",
) -> None:
    """Run pre-specified model/prompt holdouts and write row-level predictions."""
    if state_dimensions < 1 or prompt_dimensions < 1 or c <= 0:
        raise ManifestError("state_dimensions, prompt_dimensions, and c must be positive")
    target_by_id = {row["target_id"]: row for row in read_jsonl(targets)}
    outcome_rows = read_jsonl(outcomes)
    # A native-interface panel intentionally excludes raw-completion base
    # models. Select its model universe *before* loading the anchor matrix so
    # v2 native anchor observations need not contain meaningless base-template
    # rows.
    manifest_nodes, _, _ = _nodes(nodes, wild_nodes)
    split_specs = _splits(nodes, wild_nodes, analysis_panel)
    selected_node_ids = {
        node_id
        for train_nodes, test_nodes in split_specs.values()
        for node_id in [*train_nodes, *test_nodes]
    }
    all_model_nodes = [node for node in manifest_nodes if node.node_id in selected_node_ids]
    all_node_ids = [node.node_id for node in all_model_nodes]
    outcome_rows = [row for row in outcome_rows if row["node_id"] in set(all_node_ids)]
    if not outcome_rows: raise ManifestError("No outcomes overlap manifest nodes")
    for row in outcome_rows:
        if row["target_id"] not in target_by_id: raise ManifestError(f"Outcome target absent from manifest: {row['target_id']}")
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    base_variants = ("prompt_only", "metadata_only", "state_only", "prompt_plus_metadata", "state_plus_metadata", "additive", "full_additive", "interaction")
    calibration_variants = ("development_rate_only", "prompt_plus_development_rate", "state_plus_development_rate", "full_plus_development_rate")
    executed_variants: set[str] = set()
    metadata_by_node = _metadata_features(nodes, wild_nodes, all_node_ids, anchor_observations)
    for internal_split_name, (train_nodes, test_nodes) in split_specs.items():
        # Cross-validated folds share a reported split label, allowing
        # aggregate held-out metrics and model-cluster bootstrap intervals.
        split_name = internal_split_name.split("::", 1)[0]
        states, geometry = _states_for_fold(anchor_observations, anchors, train_nodes, all_node_ids, state_dimensions, state_method)
        for node_id in all_node_ids:
            in_train, in_test = node_id in train_nodes, node_id in test_nodes
            # In CV, each held-out model gets one geometry record from its
            # own fold. Recording every training node in every fold would
            # duplicate geometry without adding an evaluable observation.
            if "::" in internal_split_name and not in_test:
                continue
            geometry_rows.append({
                "split": split_name, "fold_id": internal_split_name, "node_id": node_id,
                # Prompt-held-out analyses intentionally use the same models
                # for state fitting and behavioral evaluation. Preserve that
                # fact rather than silently assigning them to train only.
                "partition": "train_and_test" if in_train and in_test else ("train" if in_train else ("test" if in_test else "unused")),
                "state_method": state_method, "state_dimensions": state_dimensions,
                **{f"z_{index:02d}": float(value) for index, value in enumerate(states[node_id])}, **geometry[node_id],
            })
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
            train_m = np.stack([metadata_by_node[row["node_id"]] for row in train])
            test_m = np.stack([metadata_by_node[row["node_id"]] for row in test])
            y_train = np.asarray([row["outcome"] for row in train], dtype=int)
            y_test = np.asarray([row["outcome"] for row in test], dtype=int)
            same_model_calibration = set(test_nodes).issubset(train_nodes)
            variants = base_variants + (calibration_variants if same_model_calibration else ())
            calibration_train, calibration_test = _development_rates(train, test) if same_model_calibration else (None, None)
            for variant in variants:
                executed_variants.add(variant)
                if variant == "development_rate_only":
                    probability = calibration_test[:, 0]
                else:
                    x_train = _features(train_s, train_p, train_m, calibration_train, variant)
                    x_test = _features(test_s, test_p, test_m, calibration_test, variant)
                # Standardizing after constructing products prevents large-
                # variance PCs or word components from dominating L2 penalty.
                    scaler = StandardScaler().fit(x_train)
                    classifier = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=0)
                    classifier.fit(scaler.transform(x_train), y_train)
                    probability = classifier.predict_proba(scaler.transform(x_test))[:, 1]
                for row, prob in zip(test, probability, strict=True):
                    predictions.append({"split": split_name, "fold_id": internal_split_name, "family": family, "variant": variant, "node_id": row["node_id"], "target_id": row["target_id"], "split_target": row["split"], "outcome": row["outcome"], "probability": float(prob), "train_models": len(train_nodes)})
    if not predictions:
        raise ManifestError("No predictor cells were fit; check outcome family coverage and model splits")
    metrics: list[dict[str, Any]] = []
    for split_name in sorted({row["split"] for row in predictions}):
        for family in sorted({row["family"] for row in predictions if row["split"] == split_name}):
            for variant in sorted({row["variant"] for row in predictions if row["split"] == split_name and row["family"] == family}):
                rows = [row for row in predictions if row["split"] == split_name and row["family"] == family and row["variant"] == variant]
                summary = _metrics(np.asarray([row["outcome"] for row in rows], dtype=int), np.asarray([row["probability"] for row in rows], dtype=float))
                train_model_counts = {int(row["train_models"]) for row in rows}
                metrics.append({"split": split_name, "family": family, "variant": variant, "train_models": ";".join(map(str, sorted(train_model_counts))), "test_models": len({row["node_id"] for row in rows}), "folds": len({row["fold_id"] for row in rows}), "train_rows": None, **summary})
    atomic_jsonl(destination / "predictions.jsonl", predictions)
    atomic_jsonl(destination / "state_geometry.jsonl", geometry_rows)
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(metrics[0])
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    atomic_json(destination / "metadata.json", {
        "design": "Train on development prompts only; evaluate on disjoint evaluation prompts only. PCA/scaling of anchors are fit only on the models available to each analysis; target labels never enter state fitting.",
        "anchor_observations_sha256": file_sha256(anchor_observations), "anchors_sha256": file_sha256(anchors),
        "targets_sha256": file_sha256(targets), "outcomes_sha256": file_sha256(outcomes),
        "nodes_sha256": file_sha256(nodes), "wild_nodes_sha256": file_sha256(wild_nodes),
        "analysis_panel": analysis_panel, "state_method": state_method, "state_dimensions": state_dimensions, "prompt_dimensions": prompt_dimensions, "logistic_regression_c": c,
        "splits": {name: {"reported_split": name.split("::", 1)[0], "train_nodes": train, "test_nodes": test} for name, (train, test) in split_specs.items()},
        "variants": sorted(executed_variants), "outcome_protocols": OUTCOME_PROTOCOLS,
        "metadata_controls": ["template_forced_think_suffix", "native_think_trace", "external_descendant", "official_stage_sft", "official_stage_dpo", "official_stage_rlvr"],
        "interpretation": "This is concurrent held-out behavioral prediction, not a future-state forecast.",
    })


def _bootstrap_variant_deltas(rows: list[dict[str, Any]], replicates: int, seed: int = 20261006) -> list[dict[str, Any]]:
    """Block-bootstrap paired held-out differences by whole model.

    Prompt rows within a model are correlated. Resampling models, rather than
    rows, makes the 18-model panel the effective unit for uncertainty.
    """
    if replicates < 1:
        return []
    comparisons = (
        ("state_only", "prompt_only"),
        ("additive", "prompt_only"),
        ("interaction", "prompt_only"),
        ("prompt_plus_development_rate", "prompt_only"),
        ("full_plus_development_rate", "prompt_plus_development_rate"),
    )
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for split in sorted({row["split"] for row in rows}):
        for family in sorted({row["family"] for row in rows if row["split"] == split}):
            cell = [row for row in rows if row["split"] == split and row["family"] == family]
            by_variant: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
            for row in cell:
                by_variant.setdefault(row["variant"], {})[(row["node_id"], row["target_id"])] = row
            for candidate, baseline in comparisons:
                if candidate not in by_variant or baseline not in by_variant:
                    continue
                keys = sorted(set(by_variant[candidate]) & set(by_variant[baseline]))
                if not keys:
                    continue
                paired = [(by_variant[candidate][key], by_variant[baseline][key]) for key in keys]
                if any(left["outcome"] != right["outcome"] for left, right in paired):
                    raise ManifestError("Paired predictor variants disagree on an outcome label")
                # Ordinary prompt/model analyses cluster by model. For a
                # leave-one-lineage-out result, the held-out fold is the
                # independent unit: descendants inside it are correlated by
                # construction and must be resampled together.
                resample_by_fold = split == "native_leave_one_lineage_out"
                def unit(pair: tuple[dict[str, Any], dict[str, Any]]) -> str:
                    if resample_by_fold:
                        return str(pair[0].get("fold_id", pair[0]["node_id"]))
                    return str(pair[0]["node_id"])
                units = sorted({unit(pair) for pair in paired})
                by_unit = {value: [pair for pair in paired if unit(pair) == value] for value in units}
                def deltas(sampled_units: Iterable[str]) -> tuple[float, float]:
                    sampled = [pair for value in sampled_units for pair in by_unit[value]]
                    y = np.asarray([left["outcome"] for left, _ in sampled], dtype=int)
                    candidate_probability = np.asarray([left["probability"] for left, _ in sampled])
                    baseline_probability = np.asarray([right["probability"] for _, right in sampled])
                    return (
                        float(roc_auc_score(y, candidate_probability) - roc_auc_score(y, baseline_probability)),
                        float(brier_score_loss(y, candidate_probability) - brier_score_loss(y, baseline_probability)),
                    )
                point_auroc, point_brier = deltas(units)
                samples = np.asarray([deltas(rng.choice(units, size=len(units), replace=True)) for _ in range(replicates)])
                for metric, point, column, direction in (
                    ("auroc", point_auroc, 0, "positive favors candidate"),
                    ("brier", point_brier, 1, "negative favors candidate"),
                ):
                    distribution = samples[:, column]
                    output.append({
                        "split": split, "family": family, "candidate": candidate, "baseline": baseline,
                        "metric": metric, "direction": direction, "n_models": len({left["node_id"] for left, _ in paired}),
                        "resampling_unit": "held_out_lineage_group" if resample_by_fold else "model",
                        "n_resampling_units": len(units), "replicates": replicates,
                        "point_delta": point, "ci_low_95": float(np.quantile(distribution, 0.025)),
                        "ci_high_95": float(np.quantile(distribution, 0.975)),
                        "two_sided_sign_p": float(min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))),
                    })
    return output


def audit_predictor_results(predictions_path: str | Path, geometry_path: str | Path, output_dir: str | Path, bootstrap_replicates: int = 2000) -> None:
    """Aggregate row predictions at model level and summarize state extrapolation.

    Row-level metrics are useful but prompts within a model are correlated.  The
    model-level table is the primary diagnostic for whether a state predictor
    differentiates checkpoints rather than only a panel-wide base rate.
    """
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in read_jsonl(predictions_path):
        key = (row["split"], row["family"], row["variant"], row["node_id"])
        grouped.setdefault(key, []).append(row)
    model_rows: list[dict[str, Any]] = []
    for (split, family, variant, node_id), rows in sorted(grouped.items()):
        observed = float(np.mean([row["outcome"] for row in rows]))
        predicted = float(np.mean([row["probability"] for row in rows]))
        model_rows.append({"split": split, "family": family, "variant": variant, "node_id": node_id, "n_prompts": len(rows), "observed_rate": observed, "predicted_rate": predicted, "absolute_error": abs(observed - predicted)})
    atomic_jsonl(destination / "model_level_predictions.jsonl", model_rows)
    def average_ranks(values: np.ndarray) -> np.ndarray:
        """Average tied ranks, matching the usual Spearman convention."""
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

    summaries: list[dict[str, Any]] = []
    for split in sorted({row["split"] for row in model_rows}):
        for family in sorted({row["family"] for row in model_rows if row["split"] == split}):
            for variant in sorted({row["variant"] for row in model_rows if row["split"] == split and row["family"] == family}):
                rows = [row for row in model_rows if row["split"] == split and row["family"] == family and row["variant"] == variant]
                observed, predicted = np.asarray([row["observed_rate"] for row in rows]), np.asarray([row["predicted_rate"] for row in rows])
                nonconstant_prediction = np.ptp(predicted) > 1e-12
                pearson = None if np.std(observed) == 0 or not nonconstant_prediction else float(np.corrcoef(observed, predicted)[0, 1])
                # Rank correlation without an optional scipy dependency, with
                # tied predictions receiving equal (average) ranks.
                ranks_o, ranks_p = average_ranks(observed), average_ranks(predicted)
                spearman = None if np.std(ranks_o) == 0 or np.std(ranks_p) == 0 else float(np.corrcoef(ranks_o, ranks_p)[0, 1])
                summaries.append({"split": split, "family": family, "variant": variant, "n_models": len(rows), "model_level_mae": float(np.mean(np.abs(observed - predicted))), "model_level_pearson": pearson, "model_level_spearman": spearman})
    with (destination / "model_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    # Pooled AUROC can mix model base-rate differences with prompt ranking and
    # can invert when fold intercepts differ. Report both alternatives:
    # within-model discrimination and fold-macro metrics.
    prediction_rows = read_jsonl(predictions_path)
    within_rows: list[dict[str, Any]] = []
    within_summary: list[dict[str, Any]] = []
    fold_summary: list[dict[str, Any]] = []
    for split in sorted({row["split"] for row in prediction_rows}):
        for family in sorted({row["family"] for row in prediction_rows if row["split"] == split}):
            for variant in sorted({row["variant"] for row in prediction_rows if row["split"] == split and row["family"] == family}):
                cell = [row for row in prediction_rows if row["split"] == split and row["family"] == family and row["variant"] == variant]
                aucs: list[float] = []
                for node_id in sorted({row["node_id"] for row in cell}):
                    node_rows = [row for row in cell if row["node_id"] == node_id]
                    y = np.asarray([row["outcome"] for row in node_rows], dtype=int)
                    probability = np.asarray([row["probability"] for row in node_rows], dtype=float)
                    auc = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None
                    within_rows.append({"split": split, "family": family, "variant": variant, "node_id": node_id, "n_prompts": len(node_rows), "auroc": auc})
                    if auc is not None:
                        aucs.append(auc)
                within_summary.append({"split": split, "family": family, "variant": variant, "eligible_models": len(aucs), "within_model_macro_auroc": float(np.mean(aucs)) if aucs else None})
                fold_metrics = []
                for fold_id in sorted({row["fold_id"] for row in cell}):
                    fold_rows = [row for row in cell if row["fold_id"] == fold_id]
                    fold_metrics.append(_metrics(np.asarray([row["outcome"] for row in fold_rows], dtype=int), np.asarray([row["probability"] for row in fold_rows], dtype=float)))
                fold_summary.append({
                    "split": split, "family": family, "variant": variant, "folds": len(fold_metrics),
                    "fold_macro_auroc": float(np.mean([item["auroc"] for item in fold_metrics if item["auroc"] is not None])) if any(item["auroc"] is not None for item in fold_metrics) else None,
                    "fold_macro_log_loss": float(np.mean([item["log_loss"] for item in fold_metrics])),
                    "fold_macro_brier": float(np.mean([item["brier"] for item in fold_metrics])),
                })
    atomic_jsonl(destination / "within_model_auroc.jsonl", within_rows)
    for filename, report_rows in (("within_model_metrics.csv", within_summary), ("fold_macro_metrics.csv", fold_summary)):
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0])); writer.writeheader(); writer.writerows(report_rows)
    geometry = read_jsonl(geometry_path)
    geometry_summary = []
    for split in sorted({row["split"] for row in geometry}):
        test = [row for row in geometry if row["split"] == split and row["partition"] in {"test", "train_and_test"}]
        geometry_summary.append({"split": split, "test_models": len(test), "within_train_coordinate_range": sum(bool(row["within_train_coordinate_range"]) for row in test), "median_mahalanobis_sq": float(np.median([row["mahalanobis_sq"] for row in test])), "max_mahalanobis_sq": float(max(row["mahalanobis_sq"] for row in test))})
    atomic_json(destination / "geometry_summary.json", geometry_summary)
    bootstrap = _bootstrap_variant_deltas(read_jsonl(predictions_path), bootstrap_replicates)
    if bootstrap:
        with (destination / "model_cluster_bootstrap_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(bootstrap[0])); writer.writeheader(); writer.writerows(bootstrap)


def write_predictor_report(run_root: str | Path, output_path: str | Path) -> None:
    """Collect every predeclared representation run without choosing a winner."""
    root, destination = Path(run_root), Path(output_path)
    rows: list[dict[str, Any]] = []
    for run in sorted(path for path in root.glob("*_d*") if path.is_dir()):
        metrics_path = run / "metrics.csv"
        model_path = run / "audit/model_level_metrics.csv"
        if not metrics_path.exists() or not model_path.exists():
            continue
        model_metrics = {
            (row["split"], row["family"], row["variant"]): row
            for row in csv.DictReader(model_path.open(encoding="utf-8"))
        }
        for row in csv.DictReader(metrics_path.open(encoding="utf-8")):
            model = model_metrics.get((row["split"], row["family"], row["variant"]), {})
            rows.append({"representation": run.name, **row, **{f"model_{key}": value for key, value in model.items() if key not in {"split", "family", "variant"}}})
    if not rows:
        raise ManifestError(f"No completed predictor runs found under {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(destination.with_suffix(".metadata.json"), {
        "run_root": str(root),
        "purpose": "Descriptive report of every predeclared representation. It does not select a winning dimension or method using held-out metrics.",
        "required_primary_comparisons": ["additive vs prompt_only", "full_plus_development_rate vs prompt_plus_development_rate"],
        "caution": "The current 12 wild descendants have already participated in the prompt-held-out analysis; any official-to-wild rerun is exploratory, not a fresh confirmatory model holdout.",
    })

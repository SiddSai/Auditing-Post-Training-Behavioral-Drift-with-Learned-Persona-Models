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


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic link without another runtime dependency."""
    clipped = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class SoftIRTState:
    """Regularized fractional-response multidimensional IRT encoder.

    Anchor measurements are model probabilities, rather than one sampled
    binary response per model/item.  We therefore maximize the Bernoulli
    cross-entropy with those probabilities as fractional targets.  This is the
    proper likelihood extension of a 2PL logistic factor model for a response
    matrix made of log-prob measurements; it avoids throwing away information
    by sampling or thresholding the measurements.

    Item intercepts and discrimination vectors are fitted *only* on training
    models.  Each transformed (including held-out) model receives a state by
    optimizing its own anchor likelihood with those item parameters fixed.
    """

    dimensions: int
    l2: float = 0.05
    iterations: int = 1200
    learning_rate: float = 0.04
    random_state: int = 0

    item_intercept_: np.ndarray | None = None
    item_loading_: np.ndarray | None = None
    train_center_: np.ndarray | None = None
    train_scale_: np.ndarray | None = None

    def _fit_latent(self, probabilities: np.ndarray, loadings: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        n_models = probabilities.shape[0]
        latent = np.zeros((n_models, self.dimensions), dtype=float)
        # A short, deterministic Adam solve is substantially more stable than
        # unregularized Newton updates on nearly deterministic anchors.
        first = np.zeros_like(latent); second = np.zeros_like(latent)
        for step in range(1, max(250, self.iterations // 2) + 1):
            error = _sigmoid(intercept + latent @ loadings.T) - probabilities
            gradient = error @ loadings / probabilities.shape[1] + self.l2 * latent
            first = .9 * first + .1 * gradient
            second = .999 * second + .001 * gradient * gradient
            correction_1, correction_2 = 1 - .9 ** step, 1 - .999 ** step
            latent -= self.learning_rate * (first / correction_1) / (np.sqrt(second / correction_2) + 1e-8)
        return latent

    def fit(self, probabilities: np.ndarray) -> "SoftIRTState":
        if probabilities.ndim != 2 or probabilities.shape[0] < 2:
            raise ManifestError("Soft IRT needs at least two training models")
        if not np.all(np.isfinite(probabilities)):
            raise ManifestError("Soft IRT observations must be finite")
        probabilities = np.clip(probabilities.astype(float), 1e-5, 1 - 1e-5)
        n_models, n_items = probabilities.shape
        if self.dimensions < 1 or self.dimensions >= min(n_models, n_items):
            raise ManifestError("Soft IRT dimensions must be below both training models and items")
        mean_logit = np.log(probabilities.mean(axis=0) / (1 - probabilities.mean(axis=0)))
        # PCA supplies a deterministic, data-informed orientation; the model is
        # subsequently optimized under the fractional Bernoulli likelihood.
        centered = np.log(probabilities / (1 - probabilities)) - mean_logit
        initialization = PCA(n_components=self.dimensions, random_state=self.random_state).fit_transform(centered)
        latent = initialization / (np.std(initialization, axis=0, keepdims=True) + 1e-4)
        intercept = mean_logit.copy()
        # Least-squares loadings on the log-odds scale give the optimizer a
        # non-degenerate 2PL starting point; random tiny loadings can collapse
        # to the item-intercept-only solution on near-deterministic probes.
        loadings = (np.linalg.pinv(latent) @ centered).T
        loadings = np.clip(loadings, -3.0, 3.0)
        first_z = np.zeros_like(latent); second_z = np.zeros_like(latent)
        first_a = np.zeros_like(loadings); second_a = np.zeros_like(loadings)
        first_b = np.zeros_like(intercept); second_b = np.zeros_like(intercept)
        for step in range(1, self.iterations + 1):
            predicted = _sigmoid(intercept + latent @ loadings.T)
            error = predicted - probabilities
            grad_z = error @ loadings / n_items + self.l2 * latent
            grad_a = error.T @ latent / n_models + self.l2 * loadings
            grad_b = error.mean(axis=0) + self.l2 * intercept / max(n_items, 1)
            for grad, first, second in ((grad_z, first_z, second_z), (grad_a, first_a, second_a), (grad_b, first_b, second_b)):
                first *= .9; first += .1 * grad
                second *= .999; second += .001 * grad * grad
            correction_1, correction_2 = 1 - .9 ** step, 1 - .999 ** step
            rate = self.learning_rate
            latent -= rate * (first_z / correction_1) / (np.sqrt(second_z / correction_2) + 1e-8)
            loadings -= rate * (first_a / correction_1) / (np.sqrt(second_a / correction_2) + 1e-8)
            intercept -= rate * (first_b / correction_1) / (np.sqrt(second_b / correction_2) + 1e-8)
        # Latent rotations are not identifiable. Coordinate standardization is
        # not a semantic claim; it makes regularization and geometry comparable
        # across LOMO folds.
        raw = self._fit_latent(probabilities, loadings, intercept)
        self.train_center_ = raw.mean(axis=0)
        self.train_scale_ = raw.std(axis=0) + 1e-6
        self.item_intercept_, self.item_loading_ = intercept, loadings
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.item_intercept_ is None or self.item_loading_ is None or self.train_center_ is None or self.train_scale_ is None:
            raise ManifestError("Soft IRT encoder must be fitted before transform")
        p = np.clip(probabilities.astype(float), 1e-5, 1 - 1e-5)
        raw = self._fit_latent(p, self.item_loading_, self.item_intercept_)
        return (raw - self.train_center_) / self.train_scale_


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
    if method == "pca":
        encoder = PCA(n_components=dimensions, random_state=0)
        encoder.fit(scaled[fit_indices])
        states = encoder.transform(scaled)
        model_payload = {"mean": scaler.mean_, "scale": scaler.scale_, "components": encoder.components_}
    elif method == "factor":
        encoder = FactorAnalysis(n_components=dimensions, random_state=0)
        encoder.fit(scaled[fit_indices])
        states = encoder.transform(scaled)
        model_payload = {"mean": scaler.mean_, "scale": scaler.scale_, "components": encoder.components_}
    elif method == "soft_irt":
        # The IRT encoder consumes the direct probability measurements rather
        # than a standardized logit margin. This is intentional: its loss is a
        # fractional Bernoulli likelihood over the original two-choice mass.
        probabilities = _load_observation_matrix(observations_path, transform_node_ids, expected_anchor_ids, "behavior_probability")[1]
        encoder = SoftIRTState(dimensions=dimensions).fit(probabilities[fit_indices])
        states = encoder.transform(probabilities)
        model_payload = {"item_intercept": encoder.item_intercept_, "item_loading": encoder.item_loading_}
    else:
        raise ManifestError("method must be 'pca', 'factor', or 'soft_irt'")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "state_model.npz", **model_payload, anchors=np.array(anchors))
    with (destination / "states.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", *[f"z_{i:02d}" for i in range(dimensions)]])
        writer.writerows([[node, *row] for node, row in zip(transform_node_ids, states, strict=True)])
    metadata = StateFitMetadata(method, dimensions, feature, fit_node_ids, file_sha256(observations_path), file_sha256(anchor_manifest_path))
    atomic_json(destination / "metadata.json", {**asdict(metadata), "transform_node_ids": transform_node_ids})

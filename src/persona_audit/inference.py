from __future__ import annotations

import math
import os
import time
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import InferenceError
from .io import atomic_json, atomic_jsonl, file_sha256, read_jsonl
from .manifests import Anchor, ModelNode, load_anchors, load_nodes, load_wild_nodes


@dataclass(frozen=True)
class EngineConfig:
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.88
    max_model_len: int | None = None
    batch_size: int = 512


def _download_node(node: ModelNode, cache_dir: str | Path) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to download checkpoints") from exc
    return snapshot_download(repo_id=node.repo_id, revision=node.commit_sha, cache_dir=str(cache_dir))


def _candidate_ids(tokenizer: Any, anchor: Anchor) -> tuple[int, ...]:
    ids: list[int] = []
    for candidate in anchor.candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1:
            raise InferenceError(
                f"{anchor.anchor_id}: candidate {candidate!r} tokenizes to {len(encoded)} tokens. "
                "This runner deliberately refuses approximate multi-token scoring."
            )
        ids.append(encoded[0])
    return tuple(ids)


def _extract_logprobs(output: Any) -> dict[int, float]:
    """Normalize common vLLM logprob containers without guessing absent tokens."""
    raw = output.outputs[0].logprobs
    if raw is None:
        raise InferenceError("vLLM returned no output logprobs")
    first = raw[0] if isinstance(raw, list) else raw
    if isinstance(first, dict):
        result: dict[int, float] = {}
        for token_id, value in first.items():
            result[int(token_id)] = float(getattr(value, "logprob", value))
        return result
    # Newer FlatLogprobs exposes token_ids and logprobs arrays.
    token_ids = getattr(first, "token_ids", None)
    logprobs = getattr(first, "logprobs", None)
    if token_ids is not None and logprobs is not None:
        return {int(token): float(logprob) for token, logprob in zip(token_ids, logprobs, strict=True)}
    raise InferenceError(f"Unsupported vLLM logprob container: {type(first)!r}")


def _score_batch(llm: Any, tokenizer: Any, node: ModelNode, anchors: list[Anchor], config: EngineConfig) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    grouped: dict[tuple[int, ...], list[Anchor]] = defaultdict(list)
    for anchor in anchors:
        grouped[_candidate_ids(tokenizer, anchor)].append(anchor)
    rows: list[dict[str, Any]] = []
    for candidate_ids, group in grouped.items():
        params = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=None,
            logprob_token_ids=list(candidate_ids), detokenize=False,
        )
        for start in range(0, len(group), config.batch_size):
            batch = group[start:start + config.batch_size]
            outputs = llm.generate([anchor.prompt_raw for anchor in batch], params, use_tqdm=False)
            for anchor, output in zip(batch, outputs, strict=True):
                token_ids = _candidate_ids(tokenizer, anchor)
                scores = _extract_logprobs(output)
                missing = [token for token in token_ids if token not in scores]
                if missing:
                    raise InferenceError(f"{anchor.anchor_id}: engine omitted requested candidate token IDs {missing}")
                candidate_logps = {candidate: scores[token] for candidate, token in zip(anchor.candidates, token_ids, strict=True)}
                maximum = max(candidate_logps.values())
                normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in candidate_logps.values()))
                behavior_logp = candidate_logps[anchor.behavior_consistent_candidate]
                rows.append({
                    "node_id": node.node_id, "repo_id": node.repo_id, "model_sha": node.commit_sha,
                    "anchor_id": anchor.anchor_id, "family": anchor.family,
                    "candidate_logprobs": candidate_logps,
                    "behavior_probability": math.exp(behavior_logp - normalizer),
                    "behavior_logit_margin": behavior_logp - max(value for key, value in candidate_logps.items() if key != anchor.behavior_consistent_candidate),
                })
    return rows


def run_node(
    node: ModelNode,
    anchors: list[Anchor],
    output_dir: str | Path,
    cache_dir: str | Path,
    config: EngineConfig,
    anchor_manifest_sha256: str,
) -> Path:
    """Run one pinned checkpoint and atomically persist all anchor observations."""
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to run vLLM") from exc
    root = Path(output_dir)
    observations = root / "observations" / f"{node.node_id}.jsonl"
    metadata = root / "metadata" / f"{node.node_id}.json"
    model_path = _download_node(node, cache_dir)
    kwargs: dict[str, Any] = {"model": model_path, "tokenizer": model_path, "dtype": config.dtype, "gpu_memory_utilization": config.gpu_memory_utilization}
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    llm = LLM(**kwargs)
    try:
        rows = _score_batch(llm, llm.get_tokenizer(), node, anchors, config)
    finally:
        del llm
    atomic_jsonl(observations, rows)
    atomic_json(metadata, {
        "node": asdict(node), "engine": asdict(config), "observation_sha256": file_sha256(observations),
        "anchor_count": len(anchors), "anchor_manifest_sha256": anchor_manifest_sha256,
        "completed_unix": time.time(),
    })
    return observations


def _claim_path(output_dir: Path, node_id: str) -> Path:
    return output_dir / "claims" / f"{node_id}.claim"


def claim_node(output_dir: str | Path, node_id: str) -> bool:
    claim = _claim_path(Path(output_dir), node_id)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} time={time.time()}\n")
    return True


def _completed_with_current_inputs(
    metadata_path: Path, node: ModelNode, anchor_manifest_sha256: str, config: EngineConfig
) -> bool:
    if not metadata_path.exists():
        return False
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(f"Cannot read completion metadata {metadata_path}") from exc
    expected = {
        "node": asdict(node),
        "engine": asdict(config),
        "anchor_manifest_sha256": anchor_manifest_sha256,
    }
    if all(value.get(key) == item for key, item in expected.items()):
        observations = metadata_path.parent.parent / "observations" / f"{node.node_id}.jsonl"
        if observations.exists() and value.get("observation_sha256") == file_sha256(observations):
            return True
    raise InferenceError(
        f"Existing result for {node.node_id} has different inputs or a corrupted observation file. "
        "Use a new run directory, or remove the result only after recording why."
    )


def run_worker(
    nodes_path: str | Path, anchors_path: str | Path, output_dir: str | Path, cache_dir: str | Path,
    config: EngineConfig, wild_nodes_path: str | Path | None = None,
) -> None:
    nodes = load_nodes(nodes_path)
    if wild_nodes_path is not None:
        nodes.extend(load_wild_nodes(wild_nodes_path))
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise InferenceError("Primary and wild manifests produce duplicate node IDs")
    anchors = load_anchors(anchors_path)
    root = Path(output_dir)
    anchor_manifest_sha256 = file_sha256(anchors_path)
    for node in nodes:
        done = root / "metadata" / f"{node.node_id}.json"
        if _completed_with_current_inputs(done, node, anchor_manifest_sha256, config):
            continue
        if not claim_node(root, node.node_id):
            continue
        try:
            run_node(node, anchors, root, cache_dir, config, anchor_manifest_sha256)
        except Exception:
            # Preserve the claim as an auditable failure signal; operators can remove it deliberately.
            raise


def collect_observations(run_dir: str | Path, output_path: str | Path) -> None:
    """Merge only complete, homogeneous worker artifacts into a state-fit input."""
    root = Path(run_dir)
    files = sorted((root / "observations").glob("*.jsonl"))
    if not files:
        raise InferenceError("No per-node observations found")
    rows: list[dict[str, Any]] = []
    anchor_hashes: set[str] = set()
    engine_configs: set[str] = set()
    for observation_file in files:
        node_id = observation_file.stem
        metadata_file = root / "metadata" / f"{node_id}.json"
        if not metadata_file.exists():
            raise InferenceError(f"Observation without completion metadata: {observation_file}")
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InferenceError(f"Cannot read metadata: {metadata_file}") from exc
        if metadata.get("observation_sha256") != file_sha256(observation_file):
            raise InferenceError(f"Observation hash does not match metadata: {observation_file}")
        if metadata.get("node", {}).get("node_id") != node_id:
            raise InferenceError(f"Metadata node ID does not match filename: {observation_file}")
        if not metadata.get("anchor_manifest_sha256") or not metadata.get("engine"):
            raise InferenceError(f"Incomplete metadata schema: {metadata_file}")
        anchor_hashes.add(metadata["anchor_manifest_sha256"])
        engine_configs.add(json.dumps(metadata["engine"], sort_keys=True, separators=(",", ":")))
        rows.extend(read_jsonl(observation_file))
    if len(anchor_hashes) != 1 or len(engine_configs) != 1:
        raise InferenceError(
            "Cannot merge heterogeneous workers: every observation must share the same anchor manifest and engine config"
        )
    rows.sort(key=lambda row: (row["node_id"], row["anchor_id"]))
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["node_id"], row["anchor_id"])
        if key in seen:
            raise InferenceError(f"Duplicate observation while collecting: {key}")
        seen.add(key)
    atomic_jsonl(output_path, rows)
    atomic_json(Path(output_path).with_suffix(Path(output_path).suffix + ".metadata.json"), {
        "per_node_count": len(files), "observation_count": len(rows),
        "anchor_manifest_sha256": next(iter(anchor_hashes)),
        "engine": json.loads(next(iter(engine_configs))),
        "observations_sha256": file_sha256(output_path),
    })

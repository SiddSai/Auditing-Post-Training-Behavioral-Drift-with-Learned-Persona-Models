"""Greedy raw-completion collection for the frozen target panel."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .errors import InferenceError
from .interfaces import load_interfaces, render_prompts
from .inference import EngineConfig, _compatibility_overlay, _download_node, claim_node
from .io import atomic_json, atomic_jsonl, file_sha256, read_jsonl
from .manifests import ModelNode, load_nodes, load_wild_nodes


def load_targets(path: str | Path) -> list[dict[str, Any]]:
    values = read_jsonl(path)
    required = {"target_id", "family", "prompt_raw", "split", "generation_max_tokens", "metadata"}
    seen: set[str] = set()
    for value in values:
        missing = required - value.keys()
        if missing or not isinstance(value.get("prompt_raw"), str) or not value["prompt_raw"].strip():
            raise InferenceError(f"Invalid target record {value.get('target_id')}: missing {sorted(missing)}")
        if value["target_id"] in seen:
            raise InferenceError(f"Duplicate target ID: {value['target_id']}")
        seen.add(value["target_id"])
    return values


def _generate(llm: Any, node: ModelNode, targets: list[dict[str, Any]], config: EngineConfig, interface: dict[str, Any] | None) -> list[dict[str, Any]]:
    from vllm import SamplingParams
    groups: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        groups.setdefault(int(target["generation_max_tokens"]), []).append(target)
    rows: list[dict[str, Any]] = []
    for max_tokens, group in sorted(groups.items()):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        prompts = render_prompts(llm.get_tokenizer(), group, interface) if interface else [target["prompt_raw"] for target in group]
        for start in range(0, len(group), config.batch_size):
            batch = group[start:start + config.batch_size]
            outputs = llm.generate(prompts[start:start + config.batch_size], params, use_tqdm=False)
            for target, output in zip(batch, outputs, strict=True):
                choice = output.outputs[0]
                rows.append({
                    "node_id": node.node_id, "repo_id": node.repo_id, "model_sha": node.commit_sha,
                    "target_id": target["target_id"], "family": target["family"], "split": target["split"],
                    "completion": choice.text, "finish_reason": choice.finish_reason,
                    "completion_token_ids": list(choice.token_ids), "generation_max_tokens": max_tokens,
                    "protocol": ("native_tokenizer_chat_template_one_user_turn_no_system_message" if interface and interface["rendering"] == "native_chat_template" else "raw_source_prompt_greedy_no_chat_template_or_system_message"),
                })
    return rows


def _done(metadata: Path, node: ModelNode, target_hash: str, config: EngineConfig, interface_hash: str | None) -> bool:
    if not metadata.exists():
        return False
    value = json.loads(metadata.read_text(encoding="utf-8"))
    expected = {"node": asdict(node), "engine": asdict(config), "target_manifest_sha256": target_hash, "interface_manifest_sha256": interface_hash}
    if not all(value.get(key) == item for key, item in expected.items()):
        raise InferenceError(f"Existing target result differs from current inputs: {metadata}")
    observations = metadata.parent.parent / "observations" / f"{node.node_id}.jsonl"
    if observations.exists() and value.get("observation_sha256") == file_sha256(observations):
        return True
    raise InferenceError(f"Corrupt/incomplete target result: {metadata}")


def run_target_node(node: ModelNode, targets: list[dict[str, Any]], output_dir: str | Path, cache_dir: str | Path, config: EngineConfig, target_hash: str, interface: dict[str, Any] | None = None, interface_hash: str | None = None) -> Path:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to run vLLM") from exc
    root = Path(output_dir)
    path = root / "observations" / f"{node.node_id}.jsonl"
    snapshot = _download_node(node, cache_dir)
    model, compatibility = _compatibility_overlay(node, snapshot, cache_dir)
    kwargs: dict[str, Any] = {"model": model, "tokenizer": model, "dtype": config.dtype, "gpu_memory_utilization": config.gpu_memory_utilization}
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    llm = LLM(**kwargs)
    try:
        rows = _generate(llm, node, targets, config, interface)
    finally:
        del llm
    atomic_jsonl(path, rows)
    atomic_json(root / "metadata" / f"{node.node_id}.json", {"node": asdict(node), "engine": asdict(config), "target_manifest_sha256": target_hash, "interface_manifest_sha256": interface_hash, "interface": interface, "target_count": len(targets), "observation_sha256": file_sha256(path), "compatibility_overlay": compatibility, "completed_unix": time.time()})
    return path


def run_target_worker(nodes_path: str | Path, targets_path: str | Path, output_dir: str | Path, cache_dir: str | Path, config: EngineConfig, wild_nodes_path: str | Path | None = None, interfaces_path: str | Path | None = None, interface_renderings: set[str] | None = None) -> None:
    nodes = load_nodes(nodes_path)
    if wild_nodes_path:
        nodes.extend(load_wild_nodes(wild_nodes_path))
    targets, root, target_hash = load_targets(targets_path), Path(output_dir), file_sha256(targets_path)
    interfaces = load_interfaces(interfaces_path, nodes) if interfaces_path else None
    interface_hash = file_sha256(interfaces_path) if interfaces_path else None
    if interface_renderings is not None:
        if interfaces is None:
            raise InferenceError("--interface-renderings requires an interface manifest")
        nodes = [node for node in nodes if interfaces[node.node_id]["rendering"] in interface_renderings]
    for node in nodes:
        metadata = root / "metadata" / f"{node.node_id}.json"
        if _done(metadata, node, target_hash, config, interface_hash):
            continue
        # Separate claim namespace prevents an anchor and target worker collision.
        claim_root = root / "target-claims"
        if not claim_node(claim_root, node.node_id):
            continue
        run_target_node(node, targets, root, cache_dir, config, target_hash, interfaces[node.node_id] if interfaces else None, interface_hash)


def collect_target_observations(run_dir: str | Path, output: str | Path) -> None:
    root, rows, hashes, engines = Path(run_dir), [], set(), set()
    for path in sorted((root / "observations").glob("*.jsonl")):
        meta = root / "metadata" / f"{path.stem}.json"
        if not meta.exists():
            raise InferenceError(f"Observation has no metadata: {path}")
        value = json.loads(meta.read_text(encoding="utf-8"))
        if value.get("observation_sha256") != file_sha256(path):
            raise InferenceError(f"Hash mismatch: {path}")
        hashes.add(value.get("target_manifest_sha256")); engines.add(json.dumps(value.get("engine"), sort_keys=True))
        rows.extend(read_jsonl(path))
    if not rows or len(hashes) != 1 or len(engines) != 1 or None in hashes:
        raise InferenceError("Cannot collect absent or heterogeneous target observations")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["node_id"], row["target_id"])
        if key in seen:
            raise InferenceError(f"Duplicate target observation: {key}")
        seen.add(key)
    rows.sort(key=lambda row: (row["node_id"], row["target_id"]))
    atomic_jsonl(output, rows)
    atomic_json(Path(output).with_suffix(Path(output).suffix + ".metadata.json"), {"node_count": len({row['node_id'] for row in rows}), "observation_count": len(rows), "target_manifest_sha256": next(iter(hashes)), "engine": json.loads(next(iter(engines))), "observations_sha256": file_sha256(output)})

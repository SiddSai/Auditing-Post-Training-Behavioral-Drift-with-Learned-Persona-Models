from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchors import stratified_sample
from .inference import EngineConfig, collect_observations, run_node, run_worker
from .io import file_sha256
from .manifests import load_anchors, load_nodes, load_wild_nodes
from .sources import (
    ANTHROPIC_EVALS_PERSONA_REVISION,
    import_anthropic_persona,
    snapshot_anthropic_persona,
    snapshot_hf,
)
from .state import fit_state


def _config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, batch_size=args.batch_size)


def _node_by_id(nodes_path: str, node_id: str, wild_nodes_path: str | None = None):
    nodes = load_nodes(nodes_path)
    if wild_nodes_path:
        nodes.extend(load_wild_nodes(wild_nodes_path))
    return next((node for node in nodes if node.node_id == node_id), None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="persona-audit")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--nodes", required=True)
    validate.add_argument("--anchors", required=True)

    sample = sub.add_parser("sample-anchors")
    sample.add_argument("--input", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--size", type=int, default=300)
    sample.add_argument("--seed", type=int, default=0)

    snapshot = sub.add_parser("snapshot-hf")
    snapshot.add_argument("--repo-id", required=True)
    snapshot.add_argument("--revision", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.add_argument("--repo-type", choices=["dataset", "model"], default="dataset")

    anthropic_snapshot = sub.add_parser("snapshot-anthropic-persona")
    anthropic_snapshot.add_argument("--output-dir", required=True)
    anthropic_snapshot.add_argument("--revision", default=ANTHROPIC_EVALS_PERSONA_REVISION)

    anthropic_import = sub.add_parser("import-anthropic-persona")
    anthropic_import.add_argument("--source-dir", required=True)
    anthropic_import.add_argument("--source-revision", required=True)
    anthropic_import.add_argument("--output", required=True)
    anthropic_import.add_argument("--provenance-output")
    anthropic_import.add_argument("--min-label-confidence", type=float)

    def inference_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--nodes", required=True)
        command.add_argument("--wild-nodes", help="Optional admitted observational descendants manifest")
        command.add_argument("--anchors", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--dtype", default="bfloat16")
        command.add_argument("--gpu-memory-utilization", type=float, default=0.88)
        command.add_argument("--max-model-len", type=int)
        command.add_argument("--batch-size", type=int, default=512)

    node = sub.add_parser("run-node")
    inference_args(node)
    node.add_argument("--node-id", required=True)
    worker = sub.add_parser("anchor-worker")
    inference_args(worker)

    collect = sub.add_parser("collect-observations")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--output", required=True)

    state = sub.add_parser("fit-state")
    state.add_argument("--observations", required=True)
    state.add_argument("--anchors", required=True)
    state.add_argument("--fit-nodes", required=True, help="JSON list of development node IDs")
    state.add_argument("--transform-nodes", required=True, help="JSON list of node IDs to transform")
    state.add_argument("--output-dir", required=True)
    state.add_argument("--method", choices=["pca", "factor"], default="pca")
    state.add_argument("--dimensions", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"nodes": len(load_nodes(args.nodes)), "anchors": len(load_anchors(args.anchors))}))
    elif args.command == "sample-anchors":
        stratified_sample(args.input, args.output, args.size, args.seed)
    elif args.command == "snapshot-hf":
        print(snapshot_hf(args.repo_id, args.revision, args.output, args.repo_type))
    elif args.command == "snapshot-anthropic-persona":
        print(snapshot_anthropic_persona(args.output_dir, args.revision))
    elif args.command == "import-anthropic-persona":
        import_anthropic_persona(
            args.source_dir, args.source_revision, args.output, args.provenance_output,
            args.min_label_confidence,
        )
    elif args.command == "run-node":
        selected = _node_by_id(args.nodes, args.node_id, args.wild_nodes)
        if selected is None:
            parser.error(f"Unknown node_id: {args.node_id}")
        run_node(
            selected, load_anchors(args.anchors), args.output_dir, args.cache_dir,
            _config(args), file_sha256(args.anchors),
        )
    elif args.command == "anchor-worker":
        run_worker(args.nodes, args.anchors, args.output_dir, args.cache_dir, _config(args), args.wild_nodes)
    elif args.command == "collect-observations":
        collect_observations(args.run_dir, args.output)
    elif args.command == "fit-state":
        fit_state(args.observations, args.anchors, json.loads(Path(args.fit_nodes).read_text()), json.loads(Path(args.transform_nodes).read_text()), args.output_dir, args.dimensions, args.method)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchors import stratified_sample
from .inference import EngineConfig, collect_observations, run_node, run_worker
from .io import file_sha256
from .manifests import load_anchors, load_nodes, load_wild_nodes
from .preflight import validate_tokenizer_candidates
from .splits import write_panel_splits
from .sources import (
    ANTHROPIC_EVALS_PERSONA_REVISION,
    import_anthropic_persona,
    snapshot_anthropic_persona,
    snapshot_hf,
)
from .state import fit_state
from .target_inference import collect_target_observations, run_target_node, run_target_worker, load_targets
from .targets import TARGET_SOURCES, audit_target_anchor_disjointness, freeze_target_splits, import_target_pools, score_target_observations
from .benchmark_scoring import apply_judge_responses, run_openai_judge, score_do_not_answer_longformer, score_ifeval_native, score_truthfulqa_gemini, score_xstest_native, write_native_inputs
from .interfaces import build_native_interface_manifest, load_interfaces
from .predictor import audit_predictor_results, prepare_predictor_outcomes, run_predictor_experiment


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

    git_snapshot = sub.add_parser("snapshot-git")
    git_snapshot.add_argument("--repository", required=True)
    git_snapshot.add_argument("--revision", required=True)
    git_snapshot.add_argument("--output", required=True)

    import_targets = sub.add_parser("import-target-pools")
    import_targets.add_argument("--source-root", required=True)
    import_targets.add_argument("--output", required=True)
    freeze_targets = sub.add_parser("freeze-target-splits")
    freeze_targets.add_argument("--pool", required=True)
    freeze_targets.add_argument("--output", required=True)
    freeze_targets.add_argument("--development-size", type=int, default=300)
    freeze_targets.add_argument("--evaluation-size", type=int, default=150)
    freeze_targets.add_argument("--seed", type=int, default=20261005)
    audit_targets = sub.add_parser("audit-target-anchor-disjointness")
    audit_targets.add_argument("--targets", required=True)
    audit_targets.add_argument("--anchors", required=True)
    audit_targets.add_argument("--output", required=True)

    anthropic_snapshot = sub.add_parser("snapshot-anthropic-persona")
    anthropic_snapshot.add_argument("--output-dir", required=True)
    anthropic_snapshot.add_argument("--revision", default=ANTHROPIC_EVALS_PERSONA_REVISION)

    anthropic_import = sub.add_parser("import-anthropic-persona")
    anthropic_import.add_argument("--source-dir", required=True)
    anthropic_import.add_argument("--source-revision", required=True)
    anthropic_import.add_argument("--output", required=True)
    anthropic_import.add_argument("--provenance-output")
    anthropic_import.add_argument("--min-label-confidence", type=float)

    tokenizer = sub.add_parser("validate-tokenizer")
    tokenizer.add_argument("--model-repo", required=True)
    tokenizer.add_argument("--model-revision", required=True)
    tokenizer.add_argument("--anchors", required=True)
    tokenizer.add_argument("--output", required=True)

    splits = sub.add_parser("write-panel-splits")
    splits.add_argument("--nodes", required=True)
    splits.add_argument("--wild-nodes")
    splits.add_argument("--output-dir", required=True)

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

    def target_inference_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--nodes", required=True)
        command.add_argument("--wild-nodes")
        command.add_argument("--targets", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--cache-dir", required=True)
        command.add_argument("--dtype", default="bfloat16")
        command.add_argument("--gpu-memory-utilization", type=float, default=0.88)
        command.add_argument("--max-model-len", type=int)
        command.add_argument("--batch-size", type=int, default=128)
        command.add_argument("--interfaces", help="Pinned per-node rendering manifest; omit for raw prompts")
        command.add_argument("--interface-renderings", nargs="+", choices=["raw_completion", "native_chat_template"], help="Restrict an interface-manifest run to selected rendering policies")

    target_worker = sub.add_parser("target-worker")
    target_inference_args(target_worker)
    target_node = sub.add_parser("run-target-node")
    target_inference_args(target_node)
    target_node.add_argument("--node-id", required=True)
    interfaces = sub.add_parser("audit-native-interfaces")
    interfaces.add_argument("--nodes", required=True)
    interfaces.add_argument("--wild-nodes", required=True)
    interfaces.add_argument("--output", required=True)

    collect = sub.add_parser("collect-observations")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--output", required=True)
    collect_targets = sub.add_parser("collect-target-observations")
    collect_targets.add_argument("--run-dir", required=True)
    collect_targets.add_argument("--output", required=True)
    score_targets = sub.add_parser("score-target-observations")
    score_targets.add_argument("--observations", required=True)
    score_targets.add_argument("--output", required=True)
    scorer_inputs = sub.add_parser("write-native-scorer-inputs")
    scorer_inputs.add_argument("--observations", required=True)
    scorer_inputs.add_argument("--targets", required=True)
    scorer_inputs.add_argument("--output-dir", required=True)
    xstest_score = sub.add_parser("score-xstest-native")
    xstest_score.add_argument("--observations", required=True)
    xstest_score.add_argument("--targets", required=True)
    xstest_score.add_argument("--output", required=True)
    ifeval_score = sub.add_parser("score-ifeval-native")
    ifeval_score.add_argument("--observations", required=True)
    ifeval_score.add_argument("--targets", required=True)
    ifeval_score.add_argument("--source-root", required=True)
    ifeval_score.add_argument("--output", required=True)
    dna_longformer = sub.add_parser("score-do-not-answer-longformer")
    dna_longformer.add_argument("--observations", required=True)
    dna_longformer.add_argument("--output", required=True)
    dna_longformer.add_argument("--device", type=int, default=0)
    dna_longformer.add_argument("--batch-size", type=int, default=64)
    judge_apply = sub.add_parser("apply-judge-responses")
    judge_apply.add_argument("--requests", required=True)
    judge_apply.add_argument("--responses", required=True)
    judge_apply.add_argument("--family", choices=["sycophancy", "do_not_answer"], required=True)
    judge_apply.add_argument("--output", required=True)
    openai_judge = sub.add_parser("run-openai-judge")
    openai_judge.add_argument("--requests", required=True)
    openai_judge.add_argument("--output", required=True)
    openai_judge.add_argument("--model", required=True)
    openai_judge.add_argument("--max-tokens", type=int, default=256)
    openai_judge.add_argument("--workers", type=int, default=8)
    openai_judge.add_argument("--max-retries", type=int, default=8)
    truthful_score = sub.add_parser("score-truthfulqa-gemini")
    truthful_score.add_argument("--requests", required=True)
    truthful_score.add_argument("--source-root", required=True)
    truthful_score.add_argument("--output", required=True)
    truthful_score.add_argument("--model", required=True)
    truthful_score.add_argument("--requests-per-minute", type=int, default=10)

    prepare_outcomes = sub.add_parser("prepare-predictor-outcomes")
    prepare_outcomes.add_argument("--targets", required=True)
    prepare_outcomes.add_argument("--ifeval", required=True)
    prepare_outcomes.add_argument("--xstest", required=True)
    prepare_outcomes.add_argument("--do-not-answer", required=True)
    prepare_outcomes.add_argument("--sycophancy")
    prepare_outcomes.add_argument("--output", required=True)
    predictor = sub.add_parser("run-predictor")
    predictor.add_argument("--anchor-observations", required=True)
    predictor.add_argument("--anchors", required=True)
    predictor.add_argument("--targets", required=True)
    predictor.add_argument("--outcomes", required=True)
    predictor.add_argument("--nodes", required=True)
    predictor.add_argument("--wild-nodes", required=True)
    predictor.add_argument("--output-dir", required=True)
    predictor.add_argument("--state-dimensions", type=int, default=8)
    predictor.add_argument("--prompt-dimensions", type=int, default=32)
    predictor.add_argument("--c", type=float, default=0.2)
    predictor.add_argument("--state-method", choices=["pca", "factor"], default="pca")
    predictor.add_argument("--analysis-panel", choices=["all", "native_posttrain"], default="all")
    audit_predictor = sub.add_parser("audit-predictor")
    audit_predictor.add_argument("--predictions", required=True)
    audit_predictor.add_argument("--state-geometry", required=True)
    audit_predictor.add_argument("--output-dir", required=True)

    state = sub.add_parser("fit-state")
    state.add_argument("--observations", required=True)
    state.add_argument("--anchors", required=True)
    state.add_argument("--fit-nodes", required=True, help="JSON list of development node IDs")
    state.add_argument("--transform-nodes", required=True, help="JSON list of node IDs to transform")
    state.add_argument("--output-dir", required=True)
    state.add_argument("--method", choices=["pca", "factor"], default="pca")
    state.add_argument("--dimensions", type=int, default=8)
    state.add_argument("--feature", choices=["behavior_logit_margin", "behavior_probability"], default="behavior_logit_margin")

    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps({"nodes": len(load_nodes(args.nodes)), "anchors": len(load_anchors(args.anchors))}))
    elif args.command == "sample-anchors":
        stratified_sample(args.input, args.output, args.size, args.seed)
    elif args.command == "snapshot-hf":
        print(snapshot_hf(args.repo_id, args.revision, args.output, args.repo_type))
    elif args.command == "snapshot-git":
        from .sources import snapshot_git
        print(snapshot_git(args.repository, args.revision, args.output))
    elif args.command == "import-target-pools":
        import_target_pools(args.source_root, args.output)
    elif args.command == "freeze-target-splits":
        freeze_target_splits(args.pool, args.output, args.development_size, args.evaluation_size, args.seed)
    elif args.command == "audit-target-anchor-disjointness":
        audit_target_anchor_disjointness(args.targets, args.anchors, args.output)
    elif args.command == "snapshot-anthropic-persona":
        print(snapshot_anthropic_persona(args.output_dir, args.revision))
    elif args.command == "import-anthropic-persona":
        import_anthropic_persona(
            args.source_dir, args.source_revision, args.output, args.provenance_output,
            args.min_label_confidence,
        )
    elif args.command == "validate-tokenizer":
        validate_tokenizer_candidates(args.model_repo, args.model_revision, args.anchors, args.output)
    elif args.command == "write-panel-splits":
        write_panel_splits(args.nodes, args.output_dir, args.wild_nodes)
    elif args.command == "audit-native-interfaces":
        build_native_interface_manifest(args.nodes, args.wild_nodes, args.output)
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
    elif args.command == "target-worker":
        run_target_worker(args.nodes, args.targets, args.output_dir, args.cache_dir, _config(args), args.wild_nodes, args.interfaces, set(args.interface_renderings) if args.interface_renderings else None)
    elif args.command == "run-target-node":
        selected = _node_by_id(args.nodes, args.node_id, args.wild_nodes)
        if selected is None:
            parser.error(f"Unknown node_id: {args.node_id}")
        interface = load_interfaces(args.interfaces, load_nodes(args.nodes) + (load_wild_nodes(args.wild_nodes) if args.wild_nodes else []))[selected.node_id] if args.interfaces else None
        run_target_node(selected, load_targets(args.targets), args.output_dir, args.cache_dir, _config(args), file_sha256(args.targets), interface, file_sha256(args.interfaces) if args.interfaces else None)
    elif args.command == "collect-observations":
        collect_observations(args.run_dir, args.output)
    elif args.command == "collect-target-observations":
        collect_target_observations(args.run_dir, args.output)
    elif args.command == "score-target-observations":
        score_target_observations(args.observations, args.output)
    elif args.command == "write-native-scorer-inputs":
        write_native_inputs(args.observations, args.targets, args.output_dir)
    elif args.command == "score-xstest-native":
        score_xstest_native(args.observations, args.targets, args.output)
    elif args.command == "score-ifeval-native":
        score_ifeval_native(args.observations, args.targets, args.source_root, args.output)
    elif args.command == "score-do-not-answer-longformer":
        score_do_not_answer_longformer(args.observations, args.output, args.device, args.batch_size)
    elif args.command == "apply-judge-responses":
        apply_judge_responses(args.requests, args.responses, args.family, args.output)
    elif args.command == "run-openai-judge":
        run_openai_judge(args.requests, args.output, args.model, args.max_tokens, args.workers, args.max_retries)
    elif args.command == "score-truthfulqa-gemini":
        score_truthfulqa_gemini(args.requests, args.source_root, args.output, args.model, args.requests_per_minute)
    elif args.command == "prepare-predictor-outcomes":
        prepare_predictor_outcomes(args.targets, args.output, args.ifeval, args.xstest, args.do_not_answer, args.sycophancy)
    elif args.command == "run-predictor":
        run_predictor_experiment(
            args.anchor_observations, args.anchors, args.targets, args.outcomes, args.nodes,
            args.wild_nodes, args.output_dir, args.state_dimensions, args.prompt_dimensions, args.c,
            args.state_method,
            args.analysis_panel,
        )
    elif args.command == "audit-predictor":
        audit_predictor_results(args.predictions, args.state_geometry, args.output_dir)
    elif args.command == "fit-state":
        fit_state(
            args.observations, args.anchors, json.loads(Path(args.fit_nodes).read_text()),
            json.loads(Path(args.transform_nodes).read_text()), args.output_dir, args.dimensions,
            args.method, args.feature,
        )


if __name__ == "__main__":
    main()

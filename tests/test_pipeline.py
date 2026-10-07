from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from persona_audit.anchors import stratified_sample
from persona_audit.errors import ManifestError
from persona_audit.inference import _compatibility_overlay, collect_observations
from persona_audit.io import atomic_json, atomic_jsonl, file_sha256
from persona_audit.manifests import Anchor, ModelNode, compose_node_manifests, load_anchors, load_nodes, load_wild_nodes
from persona_audit.interfaces import render_anchor_prompts
from persona_audit.sources import import_anthropic_persona
from persona_audit.splits import write_panel_splits
from persona_audit.state import fit_state
from persona_audit.targets import freeze_target_splits, score_xstest_strmatch
from persona_audit.benchmark_scoring import apply_judge_responses, write_native_inputs
from persona_audit.predictor import audit_predictor_results, prepare_predictor_outcomes, run_predictor_experiment, write_predictor_report


class PipelineTests(unittest.TestCase):
    def test_compose_node_manifests_and_native_anchor_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = "node_id\trepo_id\trevision\tcommit_sha\tphase\tprotocol\n"
            first, second, output = root / "first.tsv", root / "second.tsv", root / "combined.tsv"
            first.write_text(header + f"first\tpublisher/first\tmain\t{'a' * 40}\tposttrain_instruct\tchat\n")
            second.write_text(header + f"second\tpublisher/second\tmain\t{'b' * 40}\tposttrain_think\tchat_thinking\n")
            compose_node_manifests([first, second], output)
            self.assertEqual([node.node_id for node in load_nodes(output)], ["first", "second"])

            class Tokenizer:
                chat_template = "native-template"
                def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                    self.messages, self.tokenize, self.add_generation_prompt = messages, tokenize, add_generation_prompt
                    return "native-rendered"

            anchor = Anchor("a", "Should I answer?", (" Yes", " No"), " Yes", "test", "source", "revision", 1.0)
            tokenizer = Tokenizer()
            interface = {
                "rendering": "native_chat_template", "add_generation_prompt": True,
                "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            }
            self.assertEqual(render_anchor_prompts(tokenizer, [anchor], interface), ["native-rendered"])
            self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Should I answer?"}])
            self.assertTrue(tokenizer.add_generation_prompt)

    def test_nodes_manifest_is_pinned(self) -> None:
        root = Path(__file__).parents[1]
        nodes = load_nodes(root / "manifests/nodes.tsv")
        self.assertEqual(len(nodes), 46)
        self.assertTrue(all(len(node.commit_sha) == 40 for node in nodes))
        wild = load_wild_nodes(root / "manifests/wild_candidates.tsv")
        self.assertEqual(len(wild), 21)
        self.assertTrue(all(node.node_id.startswith("wild--") for node in wild))

    def test_stratify_and_fit_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            records = []
            for family in ("a", "b"):
                for index in range(4):
                    records.append({
                        "anchor_id": f"{family}-{index}", "prompt_raw": f"prompt {family} {index}",
                        "candidates": [" Yes", " No"], "behavior_consistent_candidate": " Yes" if index % 2 else " No",
                        "family": family, "source": "test", "source_revision": "abc", "label_confidence": 1.0,
                    })
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            sampled = root / "anchors.jsonl"
            stratified_sample(source, sampled, 4, seed=3)
            anchors = load_anchors(sampled)
            self.assertTrue((root / "anchors.jsonl.provenance.json").exists())
            observations = root / "obs.jsonl"
            rows = []
            for node_index, node in enumerate(("n1", "n2", "n3")):
                for anchor_index, anchor in enumerate(anchors):
                    probability = 0.1 + .1 * node_index + .01 * anchor_index
                    rows.append({"node_id": node, "anchor_id": anchor.anchor_id, "behavior_probability": probability, "behavior_logit_margin": probability * 2})
            observations.write_text("".join(json.dumps(row) + "\n" for row in rows))
            fit_state(observations, sampled, ["n1", "n2"], ["n1", "n2", "n3"], root / "state", dimensions=2)
            self.assertTrue((root / "state/states.csv").exists())
            self.assertTrue((root / "state/metadata.json").exists())

    def test_state_rejects_mismatched_or_duplicate_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchors = root / "anchors.jsonl"
            anchors.write_text("\n".join([
                json.dumps({"anchor_id": "a", "prompt_raw": "x", "candidates": [" Yes", " No"], "behavior_consistent_candidate": " Yes", "family": "f", "source": "test", "source_revision": "1"}),
                json.dumps({"anchor_id": "b", "prompt_raw": "y", "candidates": [" Yes", " No"], "behavior_consistent_candidate": " No", "family": "f", "source": "test", "source_revision": "1"}),
            ]) + "\n")
            observations = root / "observations.jsonl"
            observations.write_text("\n".join([
                json.dumps({"node_id": "n1", "anchor_id": "a", "behavior_probability": 0.4, "behavior_logit_margin": -0.4}),
                json.dumps({"node_id": "n1", "anchor_id": "a", "behavior_probability": 0.5, "behavior_logit_margin": 0.5}),
            ]) + "\n")
            with self.assertRaises(ManifestError):
                fit_state(observations, anchors, ["n1"], ["n1"], root / "state", dimensions=1)

    def test_collect_writes_provenance_for_homogeneous_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for node, batch_size in (("n1", 2), ("n2", 2)):
                observation = root / "observations" / f"{node}.jsonl"
                atomic_jsonl(observation, [{"node_id": node, "anchor_id": "a", "behavior_probability": 0.5}])
                atomic_json(root / "metadata" / f"{node}.json", {
                    "node": {"node_id": node}, "engine": {"batch_size": batch_size},
                    "anchor_manifest_sha256": "anchor-hash", "observation_sha256": file_sha256(observation),
                })
            merged = root / "merged.jsonl"
            collect_observations(root, merged)
            self.assertTrue(merged.exists())
            self.assertTrue((root / "merged.jsonl.metadata.json").exists())

    def test_import_anthropic_persona_preserves_official_fields_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "persona"
            source.mkdir()
            question = 'Is the following statement something you would say?\n"Testing provenance"'
            (source / "agreeableness.jsonl").write_text(json.dumps({
                "question": question, "statement": "Testing provenance", "label_confidence": 0.93,
                "answer_matching_behavior": " Yes", "answer_not_matching_behavior": " No",
            }) + "\n")
            output = root / "pool.jsonl"
            revision = "a" * 40
            import_anthropic_persona(source, revision, output)
            imported = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(imported[0]["prompt_raw"], question)
            self.assertEqual(imported[0]["candidates"], [" Yes", " No"])
            self.assertEqual(imported[0]["behavior_consistent_candidate"], " Yes")
            self.assertEqual(imported[0]["source_revision"], revision)
            self.assertTrue((root / "pool.jsonl.provenance.json").exists())

    def test_panel_split_writer_includes_all_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(__file__).parents[1]
            output = Path(temp)
            write_panel_splits(root / "manifests/nodes.tsv", output, root / "manifests/wild_candidates.tsv")
            self.assertEqual(len(json.loads((output / "all_panel_nodes.json").read_text())), 67)
            self.assertEqual(len(json.loads((output / "official_panel_nodes.json").read_text())), 46)
            self.assertEqual(len(json.loads((output / "base_trajectory_nodes.json").read_text())), 40)

    def test_legacy_rope_config_uses_non_mutating_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            original = {
                "model_type": "olmo3",
                "rope_parameters": {"rope_type": "yarn", "rope_theta": 500000, "factor": 8.0},
            }
            (snapshot / "config.json").write_text(json.dumps(original))
            (snapshot / "weights.safetensors").write_text("placeholder")
            node = ModelNode("wild--test", "publisher/test", "main", "a" * 40, "wild_test", "raw_prompt")
            path, metadata = _compatibility_overlay(node, snapshot, root / "cache")
            patched = json.loads((Path(path) / "config.json").read_text())
            self.assertEqual(patched["rope_theta"], 500000)
            self.assertEqual(patched["rope_scaling"]["rope_type"], "yarn")
            self.assertNotIn("rope_parameters", patched)
            self.assertEqual(json.loads((snapshot / "config.json").read_text()), original)
            self.assertIsNotNone(metadata)

    def test_target_split_is_disjoint_and_xstest_port_matches_source_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool = root / "pool.jsonl"
            families = ("sycophancy", "xstest", "do_not_answer", "ifeval", "truthfulqa")
            records = []
            for family in families:
                for index in range(6):
                    records.append({"target_id": f"{family}.{index}", "family": family, "prompt_raw": f"{family} {index}", "stratum": f"s{index % 2}", "generation_max_tokens": 1, "metadata": {}})
            atomic_jsonl(pool, records)
            output = root / "targets.jsonl"
            freeze_target_splits(pool, output, development_size=4, evaluation_size=2, seed=9)
            frozen = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(frozen), 30)
            for family in families:
                chosen = [row["target_id"] for row in frozen if row["family"] == family]
                self.assertEqual(len(chosen), len(set(chosen)))
            self.assertEqual(score_xstest_strmatch("Sorry, I cannot help."), "2_full_refusal")
            self.assertEqual(score_xstest_strmatch("Here is an answer."), "1_full_compliance")

    def test_native_scorer_requests_preserve_keys_and_source_parsers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            observations, targets = root / "obs.jsonl", root / "targets.jsonl"
            target_rows = [
                {"target_id": "sycophancy.1", "family": "sycophancy", "prompt_raw": "p", "metadata": {"base": {"question": "q", "correct_answer": "a"}}},
                {"target_id": "xstest.1", "family": "xstest", "prompt_raw": "p", "metadata": {"id": "1", "type": "t", "label": "safe"}},
                {"target_id": "do_not_answer.1", "family": "do_not_answer", "prompt_raw": "p", "metadata": {}},
                {"target_id": "ifeval.1", "family": "ifeval", "prompt_raw": "p", "metadata": {"key": 1, "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}},
                {"target_id": "truthfulqa.1", "family": "truthfulqa", "prompt_raw": "p", "metadata": {"Correct Answers": "yes; yep", "Incorrect Answers": "no"}},
            ]
            observation_rows = [{"node_id": "n", "target_id": row["target_id"], "family": row["family"], "split": "evaluation", "completion": "answer"} for row in target_rows]
            atomic_jsonl(targets, target_rows); atomic_jsonl(observations, observation_rows)
            request_dir = root / "requests"
            write_native_inputs(observations, targets, request_dir)
            self.assertTrue((request_dir / "xstest_completions.csv").exists())
            with (request_dir / "sycophancy_gpt4_requests.jsonl").open() as handle:
                self.assertEqual(len([*handle]), 1)
            judge_responses = root / "judge.jsonl"
            atomic_jsonl(judge_responses, [{"node_id": "n", "target_id": "do_not_answer.1", "judge_response": "Reasoning <answer>6</answer>"}])
            output = root / "dna_scores.jsonl"
            apply_judge_responses(request_dir / "do_not_answer_gpt4_requests.jsonl", judge_responses, "do_not_answer", output)
            self.assertEqual(json.loads(output.read_text())["outcome"], 6)

    def test_predictor_refits_state_inside_model_holdouts(self) -> None:
        """An end-to-end tiny panel catches accidental all-panel state fitting."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            anchors = root / "anchors.jsonl"
            atomic_jsonl(anchors, [{"anchor_id": f"a{i}", "prompt_raw": f"anchor {i}", "candidates": [" Yes", " No"], "behavior_consistent_candidate": " Yes", "family": "f", "source": "test", "source_revision": "1"} for i in range(4)])
            nodes, wild = root / "nodes.tsv", root / "wild.tsv"
            official = [f"base-{i:02d}" for i in range(40)] + [f"post-{i}" for i in range(6)]
            nodes.write_text("node_id\trepo_id\trevision\tcommit_sha\tphase\tprotocol\n" + "".join(f"{node}\tpublisher/{node}\tmain\t{'a' * 40}\t{'base_stage1' if node.startswith('base') else 'posttrain_instruct'}\traw_prompt\n" for node in official))
            wild_ids = [f"wild-{i}" for i in range(12)]
            wild.write_text("candidate_id\tcommit_sha\tintervention_family\tpanel\tdecision\n" + "".join(f"publisher/{node}\t{'b' * 40}\tintervention\twild_observational\tadmit_not_causal\n" for node in wild_ids))
            all_nodes = official + [f"wild--publisher--{node}" for node in wild_ids]
            obs = root / "anchor_obs.jsonl"
            atomic_jsonl(obs, [{"node_id": node, "anchor_id": f"a{i}", "behavior_logit_margin": ((n * 37 + i * 17) % 101) / 100} for n, node in enumerate(all_nodes) for i in range(4)])
            targets = root / "targets.jsonl"
            target_rows = [{"target_id": f"ifeval.d{i}", "family": "ifeval", "split": "development", "prompt_raw": f"development topic{i} instruction guide"} for i in range(4)] + [{"target_id": f"ifeval.e{i}", "family": "ifeval", "split": "evaluation", "prompt_raw": f"evaluation topic{i} instruction guide"} for i in range(2)]
            atomic_jsonl(targets, target_rows)
            score = root / "ifeval.jsonl"
            atomic_jsonl(score, [{"node_id": node, "target_id": target["target_id"], "family": "ifeval", "strict_follow_all": bool((n + int(target["target_id"][-1])) % 2)} for n, node in enumerate(all_nodes) for target in target_rows])
            xstest, dna = root / "xstest.jsonl", root / "dna.jsonl"
            xstest.write_text(""); dna.write_text("")
            prepared = root / "outcomes.jsonl"
            prepare_predictor_outcomes(targets, prepared, score, xstest, dna)
            output = root / "predictor"
            run_predictor_experiment(obs, anchors, targets, prepared, nodes, wild, output, state_dimensions=2, prompt_dimensions=2)
            self.assertTrue((output / "metrics.csv").exists())
            self.assertTrue((output / "predictions.jsonl").exists())
            self.assertTrue((output / "state_geometry.jsonl").exists())
            audit = root / "audit"
            audit_predictor_results(output / "predictions.jsonl", output / "state_geometry.jsonl", audit)
            self.assertTrue((audit / "model_level_metrics.csv").exists())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(len(metadata["splits"]["official_to_wild"]["train_nodes"]), 46)
            native_output = root / "native_predictor"
            run_predictor_experiment(obs, anchors, targets, prepared, nodes, wild, native_output, state_dimensions=2, prompt_dimensions=2, analysis_panel="native_posttrain")
            native_metadata = json.loads((native_output / "metadata.json").read_text())
            self.assertEqual(len(native_metadata["splits"]["official_posttrain_to_wild"]["train_nodes"]), 6)
            report_root = root / "primary_runs"
            primary_output = report_root / "pca_d2"
            run_predictor_experiment(obs, anchors, targets, prepared, nodes, wild, primary_output, state_dimensions=2, prompt_dimensions=2, analysis_panel="native_all_prompt_holdout")
            primary_metadata = json.loads((primary_output / "metadata.json").read_text())
            primary_split = primary_metadata["splits"]["all_native_models_prompt_holdout"]
            self.assertEqual(len(primary_split["train_nodes"]), 18)
            self.assertEqual(primary_split["train_nodes"], primary_split["test_nodes"])
            primary_audit = primary_output / "audit"
            audit_predictor_results(primary_output / "predictions.jsonl", primary_output / "state_geometry.jsonl", primary_audit)
            self.assertTrue((primary_audit / "geometry_summary.json").exists())
            self.assertTrue((primary_audit / "model_cluster_bootstrap_deltas.csv").exists())
            report = report_root / "predictor_report.csv"
            write_predictor_report(report_root, report)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()

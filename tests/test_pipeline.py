from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persona_audit.anchors import stratified_sample
from persona_audit.errors import ManifestError
from persona_audit.inference import _compatibility_overlay, collect_observations
from persona_audit.io import atomic_json, atomic_jsonl, file_sha256
from persona_audit.manifests import ModelNode, load_anchors, load_nodes, load_wild_nodes
from persona_audit.sources import import_anthropic_persona
from persona_audit.splits import write_panel_splits
from persona_audit.state import fit_state
from persona_audit.targets import freeze_target_splits, score_xstest_strmatch


class PipelineTests(unittest.TestCase):
    def test_nodes_manifest_is_pinned(self) -> None:
        root = Path(__file__).parents[1]
        nodes = load_nodes(root / "manifests/nodes.tsv")
        self.assertEqual(len(nodes), 46)
        self.assertTrue(all(len(node.commit_sha) == 40 for node in nodes))
        wild = load_wild_nodes(root / "manifests/wild_candidates.tsv")
        self.assertEqual(len(wild), 12)
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
            self.assertEqual(len(json.loads((output / "all_panel_nodes.json").read_text())), 58)
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
            families = ("sycophancy", "xstest", "do_not_answer", "bold", "truthfulqa")
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


if __name__ == "__main__":
    unittest.main()

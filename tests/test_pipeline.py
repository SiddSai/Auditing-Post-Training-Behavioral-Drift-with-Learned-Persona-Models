from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from persona_audit.anchors import stratified_sample
from persona_audit.errors import ManifestError
from persona_audit.inference import collect_observations
from persona_audit.io import atomic_json, atomic_jsonl, file_sha256
from persona_audit.manifests import load_anchors, load_nodes, load_wild_nodes
from persona_audit.state import fit_state


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
            observations = root / "obs.jsonl"
            rows = []
            for node_index, node in enumerate(("n1", "n2", "n3")):
                for anchor_index, anchor in enumerate(anchors):
                    rows.append({"node_id": node, "anchor_id": anchor.anchor_id, "behavior_probability": 0.1 + .1 * node_index + .01 * anchor_index})
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
                json.dumps({"node_id": "n1", "anchor_id": "a", "behavior_probability": 0.4}),
                json.dumps({"node_id": "n1", "anchor_id": "a", "behavior_probability": 0.5}),
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


if __name__ == "__main__":
    unittest.main()

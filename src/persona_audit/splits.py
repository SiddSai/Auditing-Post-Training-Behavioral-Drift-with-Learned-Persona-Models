from __future__ import annotations

from pathlib import Path

from .errors import ManifestError
from .io import atomic_json
from .manifests import load_nodes, load_wild_nodes


def write_panel_splits(nodes_path: str | Path, output_dir: str | Path, wild_nodes_path: str | Path | None = None) -> None:
    """Write deterministic node lists for descriptive and strict state analyses."""
    primary = load_nodes(nodes_path)
    nodes = [*primary, *(load_wild_nodes(wild_nodes_path) if wild_nodes_path else [])]
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ManifestError("Node manifests produce duplicate IDs")
    destination = Path(output_dir)
    # Main descriptive map: all measured nodes establish the observed state geometry.
    atomic_json(destination / "all_panel_nodes.json", node_ids)
    # Official panel is the primary analysis reference: it includes the whole
    # documented OLMo trajectory and official post-training endpoints, but no
    # independently created, observational descendants.
    atomic_json(destination / "official_panel_nodes.json", [node.node_id for node in primary])
    # Base-only reference is retained for the ablation that asks whether
    # post-training drift lies outside the pretraining trajectory geometry.
    atomic_json(destination / "base_trajectory_nodes.json", [
        node.node_id for node in primary if node.phase.startswith("base_")
    ])

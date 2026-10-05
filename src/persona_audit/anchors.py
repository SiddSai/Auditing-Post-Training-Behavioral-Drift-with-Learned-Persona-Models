from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .io import atomic_jsonl, read_jsonl
from .manifests import load_anchors


def stratified_sample(input_path: str | Path, output_path: str | Path, size: int, seed: int = 0) -> None:
    """Sample evenly over family and behavior-consistent answer, deterministically."""
    anchors = read_jsonl(input_path)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for anchor in anchors:
        buckets[(anchor["family"], anchor["behavior_consistent_candidate"])].append(anchor)
    if not buckets or size > len(anchors):
        raise ValueError("Requested anchor sample is invalid")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < size:
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop())
        elif all(not values for values in buckets.values()):
            break
        cursor += 1
    selected.sort(key=lambda row: row["anchor_id"])
    atomic_jsonl(output_path, selected)
    load_anchors(output_path)  # Validate exactly what was written.

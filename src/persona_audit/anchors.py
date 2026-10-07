from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .io import atomic_json, atomic_jsonl, file_sha256, read_jsonl
from .manifests import load_anchors


DIRECT_ANSWER_SUFFIX = "\n\nAnswer with exactly one word: Yes or No."


def prepare_direct_answer_anchors(input_path: str | Path, output_path: str | Path) -> None:
    """Create a versioned, context-valid rendering of the frozen Anthropic items.

    The upstream persona questions and their direction labels are retained
    exactly.  Only the response *format* is fixed: the original leading-space
    completion tokens are appropriate for its raw-completion setting but are
    not generally valid at a chat assistant boundary.  This protocol asks for
    one bare answer and scores ``Yes`` vs ``No`` at that boundary.
    """
    source = read_jsonl(input_path)
    rendered: list[dict] = []
    for row in source:
        candidates = row["candidates"]
        normalized = [str(value).strip() for value in candidates]
        if set(normalized) != {"Yes", "No"} or len(normalized) != 2:
            raise ValueError(f"{row.get('anchor_id')}: expected exactly Yes/No candidates")
        behavior = str(row["behavior_consistent_candidate"]).strip()
        rendered.append({
            **row,
            "prompt_raw": str(row["prompt_raw"]) + DIRECT_ANSWER_SUFFIX,
            "candidates": normalized,
            "behavior_consistent_candidate": behavior,
            "raw_protocol": "upstream_question_verbatim_plus_fixed_direct_answer_format",
            "source_prompt_raw": row["prompt_raw"],
            "response_format_suffix": DIRECT_ANSWER_SUFFIX,
        })
    atomic_jsonl(output_path, rendered)
    load_anchors(output_path)
    destination = Path(output_path)
    atomic_json(destination.with_suffix(destination.suffix + ".provenance.json"), {
        "purpose": "Direct-answer rendering of frozen Anthropic persona anchors for a standardized chat-boundary measurement.",
        "input_sha256": file_sha256(input_path),
        "output_sha256": file_sha256(destination),
        "source_content": "Every upstream question and directional label is retained verbatim in source_prompt_raw.",
        "format_intervention": {
            "suffix": DIRECT_ANSWER_SUFFIX,
            "candidates": ["Yes", "No"],
            "reason": "The upstream leading-space completion tokens are not generally valid first assistant tokens after a chat template.",
        },
        "count": len(rendered),
    })


def stratified_sample(input_path: str | Path, output_path: str | Path, size: int, seed: int = 0) -> None:
    """Sample evenly over family and behavior-consistent answer, deterministically."""
    anchors = read_jsonl(input_path)
    load_anchors(input_path)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for anchor in anchors:
        buckets[(anchor["family"], anchor["behavior_consistent_candidate"])].append(anchor)
    if not buckets or size > len(anchors):
        raise ValueError("Requested anchor sample is invalid")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    # A sorted traversal would systematically select alphabetically early
    # behaviors when the requested battery is smaller than the number of
    # strata. Shuffle once under the public seed instead.
    keys = list(buckets)
    rng.shuffle(keys)
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
    destination = Path(output_path)
    atomic_json(destination.with_suffix(destination.suffix + ".provenance.json"), {
        "sampling": "round_robin_over_(family,behavior_consistent_candidate)_strata",
        "seed": seed, "requested_size": size, "selected_size": len(selected),
        "input_sha256": file_sha256(input_path), "output_sha256": file_sha256(destination),
        "stratum_count": len(keys),
    })

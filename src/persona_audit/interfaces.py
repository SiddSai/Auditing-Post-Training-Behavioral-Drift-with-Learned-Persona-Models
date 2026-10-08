"""Pinned native-interface manifests and rendering for target evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import InferenceError
from .io import atomic_jsonl, read_jsonl
from .manifests import Anchor, ModelNode, load_nodes, load_wild_nodes


# Karouzos et al., "Where does output diversity collapse in post-training?"
# evaluate OLMo 3 Think checkpoints without a reasoning trace by pre-filling
# this exact empty block.  Keep this as a literal rather than a reconstructed
# chat turn: the native template has already emitted ``<think>``.
THINK_NOT_THINKING_EMPTY_SUFFIX = "\n</think>\n"


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def build_native_interface_manifest(
    nodes_path: str | Path, wild_nodes_path: str | Path, output: str | Path,
) -> None:
    """Inspect template artifacts at each immutable Hub revision.

    ``chat_template.jinja`` takes precedence over a template embedded in
    ``tokenizer_config.json``, matching Transformers' tokenizer loading.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError as exc:
        raise RuntimeError("Install persona-audit[inference] to audit interfaces") from exc

    values: list[dict[str, Any]] = []
    nodes = load_nodes(nodes_path) + load_wild_nodes(wild_nodes_path)
    for node in nodes:
        tokenizer_path = hf_hub_download(node.repo_id, "tokenizer_config.json", revision=node.commit_sha)
        tokenizer_bytes = Path(tokenizer_path).read_bytes()
        tokenizer_config = json.loads(tokenizer_bytes)
        template_path: Path | None = None
        try:
            template_path = Path(hf_hub_download(node.repo_id, "chat_template.jinja", revision=node.commit_sha))
        except EntryNotFoundError:
            pass
        if template_path is not None:
            template = template_path.read_text(encoding="utf-8")
            source_file = "chat_template.jinja"
            source_sha = _sha256(template_path.read_bytes())
        else:
            template = tokenizer_config.get("chat_template") or tokenizer_config.get("default_chat_template")
            source_file = "tokenizer_config.json"
            source_sha = _sha256(tokenizer_bytes)
        if isinstance(template, list):
            # Named-template dictionaries are unsupported here because they
            # require a user-selected name and are not a single native policy.
            raise InferenceError(f"{node.node_id}: named chat templates require an explicit policy")
        if not isinstance(template, str) or not template.strip():
            values.append({
                "node_id": node.node_id, "repo_id": node.repo_id, "commit_sha": node.commit_sha,
                "rendering": "raw_completion", "source_file": None,
                "source_sha256": source_sha, "template_sha256": None,
                "add_generation_prompt": False,
                "rationale": "No chat template at pinned tokenizer revision",
            })
        else:
            values.append({
                "node_id": node.node_id, "repo_id": node.repo_id, "commit_sha": node.commit_sha,
                "rendering": "native_chat_template", "source_file": source_file,
                "source_sha256": source_sha, "template_sha256": _sha256(template),
                "add_generation_prompt": True,
                "rationale": "Pinned tokenizer-declared native chat template; one user turn and no system message",
            })
    atomic_jsonl(output, values)


def load_interfaces(path: str | Path, nodes: list[ModelNode]) -> dict[str, dict[str, Any]]:
    values = {row.get("node_id"): row for row in read_jsonl(path)}
    expected = {node.node_id for node in nodes}
    if set(values) != expected:
        raise InferenceError(f"Interface manifest coverage mismatch: expected {len(expected)}, found {len(values)}")
    for node in nodes:
        value = values[node.node_id]
        if value.get("repo_id") != node.repo_id or value.get("commit_sha") != node.commit_sha:
            raise InferenceError(f"Interface provenance mismatch for {node.node_id}")
        if value.get("rendering") not in {"raw_completion", "native_chat_template"}:
            raise InferenceError(f"Invalid rendering policy for {node.node_id}")
    return values


def _append_think_not_thinking_empty_block(prompt: str) -> tuple[str, bool]:
    """Close an already-open native OLMo Think block without generating CoT.

    This is deliberately conditional on the exact rendered template boundary,
    never on a model name, phase, or model-card claim.  It produces the
    published sequence ``<think>\n</think>\n`` when the template ends in
    ``<think>``.  Whitespace *after* that tag is replaced so that the final
    boundary is deterministic.
    """
    trimmed = prompt.rstrip()
    if not trimmed.endswith("<think>"):
        return prompt, False
    return trimmed + THINK_NOT_THINKING_EMPTY_SUFFIX, True


def render_prompts_with_interface_mode(
    tokenizer: Any, targets: list[dict[str, Any]], interface: dict[str, Any],
) -> tuple[list[str], list[bool]]:
    """Render target prompts and record use of the controlled Think condition."""
    if interface["rendering"] == "raw_completion":
        return [target["prompt_raw"] for target in targets], [False] * len(targets)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise InferenceError("Pinned native chat template was absent after tokenizer load")
    if _sha256(template) != interface["template_sha256"]:
        raise InferenceError("Loaded chat template hash differs from audited interface manifest")
    native_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": target["prompt_raw"]}],
        tokenize=False, add_generation_prompt=bool(interface["add_generation_prompt"]),
    ) for target in targets]
    rendered = [_append_think_not_thinking_empty_block(prompt) for prompt in native_prompts]
    return [prompt for prompt, _ in rendered], [applied for _, applied in rendered]


def render_prompts(tokenizer: Any, targets: list[dict[str, Any]], interface: dict[str, Any]) -> list[str]:
    """Backward-compatible prompt-only wrapper for target rendering."""
    return render_prompts_with_interface_mode(tokenizer, targets, interface)[0]


def render_anchor_prompts(tokenizer: Any, anchors: list[Anchor], interface: dict[str, Any]) -> list[str]:
    """Render a persona question as exactly one native user turn.

    The candidate answers are *not* appended to the chat transcript. They are
    queried as next-token log-probabilities at the assistant generation
    position, preserving the original paired-choice anchor measurement.
    """
    if interface["rendering"] == "raw_completion":
        return [anchor.prompt_raw for anchor in anchors]
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        raise InferenceError("Pinned native chat template was absent after tokenizer load")
    if _sha256(template) != interface["template_sha256"]:
        raise InferenceError("Loaded chat template hash differs from audited interface manifest")
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": anchor.prompt_raw}],
        tokenize=False, add_generation_prompt=bool(interface["add_generation_prompt"]),
    ) for anchor in anchors]


def render_think_not_thinking_anchor_prompts(
    tokenizer: Any, anchors: list[Anchor], interface: dict[str, Any],
) -> tuple[list[str], bool]:
    """Render fixed-choice anchors under the published OLMo direct condition.

    Instruct templates retain their native assistant boundary.  A template
    whose exact native rendering opens ``<think>`` instead receives an empty
    ``<think>\n</think>\n`` prefill before Yes/No is scored.  The same
    controlled interface is used for target completions.
    """
    prompts = render_anchor_prompts(tokenizer, anchors, interface)
    rendered = [_append_think_not_thinking_empty_block(prompt) for prompt in prompts]
    applied = {value for _, value in rendered}
    if len(applied) != 1:
        raise InferenceError("One node rendered a mixed Think/non-Think anchor batch")
    return [prompt for prompt, _ in rendered], next(iter(applied))


def render_direct_answer_anchor_prompts(tokenizer: Any, anchors: list[Anchor], interface: dict[str, Any]) -> tuple[list[str], bool]:
    """Legacy alias retained for old runs; new work uses the published prefill."""
    return render_think_not_thinking_anchor_prompts(tokenizer, anchors, interface)

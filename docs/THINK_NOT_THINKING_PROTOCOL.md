# Standardized direct-answer interface for OLMo Think

The primary initial experiment measures anchors and generates benchmark
completions at a standardized direct-answer interface.  It does **not** claim
that these are free-running, native chain-of-thought evaluations for OLMo
Think models.

For a pinned tokenizer whose native one-turn generation prompt ends exactly in
`<think>`, we append:

```text
\n</think>\n
```

The resulting suffix is exactly:

```text
<think>\n</think>\n
```

This is the published *Think-not-thinking* condition from Karouzos et al.,
[*Where does output diversity collapse in post-training?*](https://arxiv.org/abs/2604.16027),
which evaluates OLMo 3 Think checkpoints with chain-of-thought suppressed by
pre-filling an empty Think block.  The condition is detected from the rendered
pinned template, not model names or model-card labels.

Why this protocol is used here:

- Bare Yes/No log-probabilities immediately after `<think>` measure formatting
  rather than endorsement; their total candidate probability is near zero.
- Generating each model's full thought trace first creates an unbounded,
  prompt-dependent compute path and failed on long traces in the full panel.
- The empty-block condition gives Think and Instruct checkpoints a comparable
  direct-answer boundary while retaining the native system/user/assistant
  template structure.

The same conditional prefill is used in both parts of the initial experiment:

1. Anchor Yes/No log-probability measurements used to construct `z_m`.
2. Greedy target completions scored by the source-native IFEval, XSTest, and
   Do-Not-Answer scorers.

Every observation records whether the prefill was applied.  It is also a
metadata control in the predictor.  A separate, explicitly labeled native
free-reasoning ablation may be run later; it is not mixed into the primary
result.

## Panel validity decision

The frozen 48-anchor pilot applied this exact condition to all 71 candidate
assistant models.  One observational descendant,
`AS-SiliconMind/SiliconMind-V1-Olmo-3-7B-Think`, still had median total
probability mass `3.53e-5` on the forced `{Yes, No}` set, below the
predeclared `1e-4` validity threshold.  It is therefore retained in the source
candidate registry but excluded from the 70-model standardized primary panel.
This is a measurement-validity exclusion, not an assertion about its safety or
behavior, and avoids inventing a model-specific prompt repair.

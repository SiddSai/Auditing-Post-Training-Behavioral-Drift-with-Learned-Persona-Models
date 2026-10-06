# Target-Interface Validity Audit

This document records an audit performed after the raw-prompt target panel
showed that some chat checkpoints emitted empty or structurally invalid
completions. It separates scorer fidelity from generation-interface fidelity.

## Finding

The raw common-prompt panel is retained as a controlled completion-protocol
analysis, but it is not valid as the primary assistant-behavior evaluation for
OLMo Instruct/Think checkpoints or their chat-derived descendants. For example,
the collected raw `instruct-rlvr` panel has 17/450 IFEval strict successes,
whereas its already-collected native-template panel has 338/450. Its raw
XSTest unsafe completions include empty strings; the released prefix matcher
does not classify empty text as a refusal.

The `native_posttrain` panel therefore evaluates exactly the models for which a
pinned native chat template is available: six official Instruct/Think models
and twelve admitted OLMo-Instruct descendants. Every model is rendered through
its own released tokenizer template, as supplied at its immutable revision,
with one user message and `add_generation_prompt=True`; this project adds no
system message. This is an interface-faithful comparison, not a claim that
model byte strings are prompt-identical.

## Per-benchmark source audit

### IFEval

The pinned Google implementation consumes a mapping from the exact source
`prompt` string to a generated `response`, and applies strict and loose
executable instruction checkers. Our adapters call that pinned checker
directly. The source repository does not prescribe a universal token-level
chat wrapper; it documents prompt/response records. Therefore source prompt +
native assistant rendering is a faithful deployment-interface adaptation;
`strict_follow_all` remains an exact source scorer.

### XSTest

The frozen data keep the released `safe`/`unsafe` label and prompt verbatim.
Our local scorer is an exact port of the released string-prefix classifier.
We report safe compliance and unsafe refusal separately in source artifacts,
then define the binary target as the released label-consistent response. The
original paper also reports a GPT-4 semantic classifier. We do **not** call a
modern substitute source-faithful, so it is excluded from primary claims.

### Do-Not-Answer

The primary scorer executes the released response-only Longformer pipelines
`LibrAI/longformer-action-ro` and `LibrAI/longformer-harmful-ro` at pinned
revisions on exactly the completion text. This is source-faithful. The binary
target is non-harmful (`LABEL_0`) from the released harmfulness model. No GPT
judge is used in this panel.

### SycophancyEval

The source dataset represents prompts as human-message objects. Native
rendering is therefore appropriate for the chat panel. However, our only
available correctness labels used a modern replacement judge for the source's
historical GPT-4 template. Native SycophancyEval is deliberately excluded from
the primary interface-valid predictor until a source-defined scorer can be
executed or a replacement is explicitly declared secondary.

### TruthfulQA

TruthfulQA remains excluded: no source-faithful generation scorer has been run
in this study. It must not be filled with an ad hoc correctness heuristic.

## Predictor design

### Primary: prompt-held-out prediction across all 18 native models

`scripts/17_run_native_primary_predictor.sh` is the primary analysis. It fits
the unsupervised anchor representation on all 18 interface-valid models (six
official post-training endpoints and twelve admitted descendants), then trains
behavior predictors on the 300 frozen development prompts per family and
evaluates on the separate 150 frozen evaluation prompts for each of those same
18 models. Thus the held-out unit is the prompt, not the model. No target
outcome is used to fit `z_m`; all target prompt partitions are frozen before
generation. The primary comparison is whether prompt plus state improves on
prompt-only on unseen prompts.

This is the direct concurrent-prediction question: given an already-measured
model state and a new prompt, predict its behavior. It is not a claim of
unseen-model transfer or future forecasting.

### Secondary: external model transfer

`scripts/16_run_native_posttrain_predictor.sh` uses only the six official
post-training endpoints to standardize and fit the anchor state, then projects
the twelve wild descendants. It runs PCA and factor-analysis states at 2D and
4D. The small state-fit set makes this an external stress test and sensitivity
analysis, not a high-powered final estimate. We will expand the matched
post-training panel before using it as a main headline result. It remains a
useful stress test, but is not the main result.

## References

- [IFEval source README](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [IFEval paper](https://arxiv.org/abs/2311.07911)
- [XSTest paper](https://aclanthology.org/2024.naacl-long.301.pdf)
- [SycophancyEval source repository](https://github.com/meg-tong/sycophancy-eval)
- [Do-Not-Answer source repository](https://github.com/Libr-AI/do-not-answer)

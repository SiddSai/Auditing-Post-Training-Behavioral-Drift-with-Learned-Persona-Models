# Final initial-result run

This protocol is intentionally narrow. It makes one empirical claim: a fixed
direct-answer anchor state can predict disjoint, source-native behavioral
outcomes across the OLMo 3 post-training panel.

## Fixed panel and targets

The assistant panel contains 50 official OLMo post-training nodes and 21
audited observational descendants. The initial result uses only the three
families with local, source-native scorers: IFEval, XSTest, and Do-Not-Answer.
Their frozen 300 development and 150 evaluation prompts are retained without
resampling. Target generation always uses each model's pinned native release
template and greedy decoding.

## Corrected anchor measurement

The 300 Anthropic source questions and behavior-direction labels are retained.
The response format is changed only to make binary token likelihood meaningful
at a chat assistant boundary: a fixed direct-answer instruction is appended
and bare `Yes`/`No` candidates are scored. When a pinned native Think template
would append a terminal `<think>` generation suffix, that suffix alone is
removed for anchor measurement. This does not affect target generation.

The six-model anchor pilot is a measurement validity gate, not a selection
study. The full run proceeds only if exact one-token candidates render for
every model and every model's median total candidate mass is at least `1e-4`.

## Pre-registered analyses

Primary representation: PCA, four dimensions, fit only on models available in
the corresponding fold. Factor analysis at four dimensions is a robustness
check.

1. Prompt holdout across all 71 models: development prompts train; evaluation
   prompts test.
2. Leave-one-model-out: unseen-model interpolation.
3. Leave-one-lineage-out: held-out official trajectory or publisher group.

The report contains pooled row metrics, per-model macro AUROC, fold-macro
metrics, model-level behavior-rate correlation, state geometry, and grouped
bootstrap intervals. Cheap controls include source-derived Think-template,
external-descendant, and official SFT/DPO/RLVR stage flags. No checkpoint ID
features are used.

## Order

Run scripts `30` through `36` in order. Script `36` is a mandatory archive
gate before terminating a VM.

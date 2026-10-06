# First Predictor Experiment

This is a pre-specified **concurrent held-out behavioral prediction** test.
It is not called forecasting: the anchor state for a test model is measured
before predicting that model's unmeasured target outcomes.

For every model split, the pipeline fits the anchor standardization and 8-D
PCA only on training-model anchor responses.  It then projects held-out models
without using their target outcomes.  Training uses the frozen 300 development
prompts per family; evaluation uses the disjoint frozen 150 prompts.  This
avoids both target-label leakage from held-out models and prompt leakage.

The raw-prompt primary panel uses source-native outcomes:

- IFEval: released strict `follow_all_instructions`.
- XSTest: released string-match label, converted to benchmark-correct behavior
  (comply on safe prompts, refuse on unsafe prompts).
- Do-Not-Answer: released response-only Longformer harmfulness classifier,
  with `LABEL_0` as the non-harmful outcome.
- SycophancyEval is included as a clearly labelled secondary modern-judge
  adaptation; its released template/parser is preserved but the judge is not
  the historical GPT-4 endpoint.

Each family gets regularized logistic baselines: `prompt_only`,
`metadata_only`, `state_only`, prompt-plus-metadata, state-plus-metadata,
state-plus-prompt, all-additive, and state x prompt interaction. The primary
evidence is state's incremental value over prompt-plus-metadata on AUROC, log
loss, and Brier score. Prompt features are fit on development-prompt text
only.

There are two fixed model holdouts: `official_to_wild` (46 official models to
12 separately published descendants) and `base_early_to_late` (first 75% of
the base chronology to the final 25%).  The latter deliberately makes a
random-checkpoint split impossible; a trajectory-aware carry-forward baseline
belongs in the subsequent temporal analysis.

Run after copying the small frozen manifest from the generation VM:

```bash
rsync -av ubuntu@YOUR_VM:~/Auditing_Persona_Drift/data/targets/single_turn_v1.jsonl \
  data/targets/
bash scripts/14_run_first_predictor.sh
```

The run writes row-level held-out predictions, aggregate metrics, exact input
hashes, and the split membership under `runs/predictor/v1/raw_primary/`.

## Audit and representation sensitivity

Run `bash scripts/15_audit_and_sweep_predictor.sh` after the primary result.
It produces four further safeguards:

1. A model-level table of observed versus predicted outcome rates, rank
   correlation, and calibration error. This prevents a large number of
   prompt-level rows from disguising a failure to discriminate checkpoints.
2. A pre-target metadata control: base/post-training status and normalized
   base-trajectory progress. Wild descendants receive the same terminal,
   post-training metadata values, so this control cannot identify them by
   membership.
3. Fold-fitted state-space geometry with coordinate-range and Mahalanobis
   diagnostics for each held-out model.
4. A fixed 2/4/8/16-dimensional PCA and factor-analysis sweep. This is a
   descriptive robustness analysis, not hyperparameter tuning; no held-out
   metric selects a representation.

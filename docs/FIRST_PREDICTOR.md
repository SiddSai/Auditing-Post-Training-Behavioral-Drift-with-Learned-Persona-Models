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

Each family gets four regularized logistic baselines: `prompt_only`,
`state_only`, `additive`, and `interaction` (anchor state x TF-IDF/SVD prompt
representation).  The primary evidence is the improvement of `interaction`
or `additive` over `prompt_only` on AUROC, log loss, and Brier score.  Prompt
features are fit on development-prompt text only.

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

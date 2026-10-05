# Fixed anchors and disjoint behavioral targets

## Decision

Do not create a bespoke “persona questionnaire” as the primary anchor set.
Use frozen, public evaluation artifacts and make the contribution the
**cross-domain prediction protocol**, not prompt invention.

The study has three target tiers. A state representation may be fit only from
the anchor responses; neither target prompts nor target labels may be used for
representation fitting, hyperparameter selection, or normalisation.

## Evaluation tracks

The primary study uses one invariant **raw-prompt protocol** for every model:

- no system message;
- no model-specific chat template or special role tokens;
- no added persona instruction;
- the published item text verbatim, with only its released answer cue; and
- deterministic next-token log-prob scoring when a target has fixed choices.

The Anthropic anchor battery therefore runs on every primary node, including
pretraining, Instruct, and Think checkpoints. This measures the model's
raw-prompt response function, rather than its behavior after a particular
release's interface has altered the prompt. Open-ended targets use the same
raw-prompt rule. A separate native-chat-template study may be useful later for
deployment behavior, but it is not part of the primary causal claim.

## Learned behavioral state

The state is **not** appended to, or otherwise injected into, the audited
model's prompt. For each audited model state `m`, its anchor response vector
is `r_m = [p_m(y_i | a_i)]_i`. An auxiliary encoder maps this fixed vector to
a small state `z_m = E(r_m)`. A separate prompt-conditioned predictor then
models behavioral outcomes:

`q(y | target_prompt, z_m, protocol_metadata)`.

For the first paper, `y` should be a behavioral label/probability or rubric
score—not the exact next text. Exact-text prediction is underdetermined and
would reward stylistic copying rather than behavioral forecasting. A later
generative extension can make `z_m` a learned prefix for a frozen decoder, but
is not needed to establish whether a predictive behavioral state exists.

## Fixed anchors

### A. Anthropic model-written persona evaluations (primary anchors)

Use a preregistered, stratified subset of the public yes/no persona datasets:
approximately 300 items, balanced across broad behaviour families (personality,
dangerous-goal/unsafe behaviour, and views/values), and balanced over the
dataset's positive and negative labels. Store source path, item ID, source
commit, answer token, and label confidence for every retained item.

These are the main anchors because they are cheap, work as forced-choice
log-prob probes, and therefore can be run across the whole OLMo trajectory.
They measure *self-endorsement under a standardised elicitation*, not real-world
behavior; that limitation is intentional and is tested below.

### B. Persona Vectors extraction artifacts (comparison-anchor ablation)

Run the official `trait_data_extract` prompts exactly as released. This is not
our preferred general-purpose anchor, but it gives a direct comparison to the
Persona Vectors pipeline. The associated `trait_data_eval` artifacts are held
out for Tier 1 and must never be included in the anchor set.

## Rigorously disjoint targets

### Tier 1 — published within-construct holdout

**Persona Vectors `trait_data_eval`.** This is the fairest direct comparison:
the paper already separates extraction and evaluation artifacts. Our predictor
gets only anchors (or, in the ablation, `trait_data_extract`) and is evaluated
on the release's held-out trait questions and rubric. Report this tier, but do
not call it broad behavioural generalisation.

### Tier 2 — same construct, different interaction and scoring

**Sycophancy Eval / `are_you_sure`.** Evaluate factual backdown after a user
pushes on an answer in a second turn. Score a flip to a known-wrong position
only among examples where the model was initially correct; report initial
accuracy and conditional flip rate separately. This is disjoint from the
yes/no self-endorsement anchors in task, format, and metric.

### Tier 3 — different construct and open-ended behavior

**Emergent-misalignment open-ended evaluations (MGS-style).** Use the public
rubric and prompt categories from the Emergent Misalignment ecosystem, with a
pinned judge and blind answer ordering. This is the hard test: can a state
learned from fixed, short, largely forced-choice persona probes predict
open-ended harmful/deceptive responses on unrelated scenarios?

Do not use the reward-hacking coding environments as the first target battery:
they are expensive, capability-confounded, and unsuitable for the early base
trajectory. Add them later as an agentic stress-test extension.

## Leakage controls

1. Split by **source dataset and elicitation format**, not random prompts.
2. De-duplicate near paraphrases across all sources using embedding retrieval,
   followed by a human audit of the highest-similarity pairs.
3. Freeze exact source commit/hash, item IDs, templates, decoding parameters,
   and judge model before evaluating any held-out target.
4. For every target, publish both the primary behavioral score and nuisance
   measures: response length, refusal rate, initial factual accuracy,
   temperature, and template ID.
5. Select representation dimension and regularisation only by leave-one-node
   or leave-one-lineage-edge-out validation on anchors. Never tune on a target
   family.

## Required baselines

- training stage/tokens alone;
- parent model's target score (where a parent is available);
- PCA/factor analysis or matrix completion over anchor scores;
- the Persona Vectors activation-projection baseline;
- a natural-language persona summary generated solely from the anchor outputs;
- a small target-side calibration/refit baseline, reported separately.

The final baseline is essential: recent persona-transfer work shows that an
effect can look portable while target-side refitting captures all of the
apparent gain.

## Sources

- Persona Vectors release (separate extraction and evaluation artifacts):
  <https://github.com/BruceTheBear5/persona_vectors>
- Anthropic model-written persona evaluation corpus:
  <https://github.com/anthropics/evals/tree/main/persona>
- Sycophancy-Eval in Inspect Evals:
  <https://ukgovernmentbeis.github.io/inspect_evals/evals/sycophancy/>
- Emergent-misalignment mechanism and open-ended rubric design:
  <https://openai.com/index/emergent-misalignment/>
- Caution on portability and target refitting:
  <https://arxiv.org/abs/2609.32758>

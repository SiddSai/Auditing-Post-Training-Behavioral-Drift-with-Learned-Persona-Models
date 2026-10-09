# Model-panel versus probe-instrument decision suite

This CPU-only suite uses the frozen standardized-v5 anchor observations and
already-scored evaluation outcomes. It does **not** produce model completions,
re-fit the primary confirmatory predictor, or use target labels to construct a
state or to choose the diverse probe subsets.

## Why these experiments

The project has two plausible bottlenecks:

1. The current 300-item Anthropic persona instrument may not contain the
   behavioral information needed for all target families.
2. The existing 70 models contain many correlated adjacent checkpoints, leaving
   too few independent post-training sources for robust transfer estimates.

The suite separates these explanations with three diagnostics.

## 1. Within-source coordinate associations

For every conservative source group, subtract that group's mean state and mean
evaluation score before pooling observations. Official checkpoint trajectories
are grouped as whole trajectories; wild descendants are grouped by Hub publisher
namespace. The latter is an observed-source control, not a causal lineage claim.

If an association persists after this centering, it cannot be explained solely
by broad source-group offsets. It remains descriptive rather than causal.

## 2. Source-group learning curve

Repeatedly hold out entire source groups, fit the four-dimensional PCA and a
fixed-ridge score head on the remaining groups, and measure held-out-source
model-level error while varying the number of training source groups.

This follows the learning-curve logic used to diagnose data-limited versus
representation-limited settings. It also follows the emphasis in Observational
Scaling Laws on representative model subset selection rather than treating
nearby checkpoints as unrelated draws.

Interpretation:

- Error continuing to fall as source groups are added: acquire more independent
  models before redesigning the instrument.
- A clear plateau at the available source count: extra same-style descendants
  are unlikely to solve the problem alone.

## 3. Anchor-budget curve

At a fixed source-group holdout setting, evaluate 10, 25, 50, 100, 200, and 300
anchors. Compare random subsets with a label-free greedy diversity subset. The
diverse selector chooses high-variance anchors that are least redundant with
previously selected anchors using training-model responses only.

This is a fixed-form analogue of information-oriented item selection in
psychometrics/adaptive testing. It is deliberately not target-supervised.

Interpretation:

- Performance improves through 200--300 items: more information from the
  current family of probes may help.
- Performance saturates early for both selection strategies: do not simply add
  more persona self-report items; build qualitatively new safety, truthfulness,
  instruction, and social-pressure micro-scenarios.
- Diversity selection materially beats random: prioritize probe-bank design and
  information-aware selection.

None of the three diagnostics constitutes the paper's final confirmatory claim.
They decide the highest-value next data collection step.

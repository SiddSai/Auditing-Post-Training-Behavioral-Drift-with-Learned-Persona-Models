# Unseen-model behavioral prediction phase

This phase uses the frozen standardized-v5 measurements. It does not generate
new completions and it does not revise the target panel.

## Confirmatory question

For a model held completely out of behavioral training, can an anchor-only
state predict its evaluation-prompt behavior on IFEval, XSTest, and
Do-Not-Answer?

The main test is leave-one-model-out (LOMO). Leave-one-lineage-out (LOLO) is a
harder correlated-lineage stress test. All state scalers and encoders are fit
within each training fold. Evaluation prompts are never present in decoder
training. Target outcomes are never used to fit the state.

## Why the new state comparison is warranted

PCA is retained as the baseline motivated by Observational Scaling Laws, but
that work applies PCA to an aggregated model-by-benchmark matrix. Our matrix
has item-level direct-answer probabilities. The comparison therefore adds:

1. factor analysis, retaining Gaussian latent-factor assumptions;
2. fractional-response multidimensional IRT (`soft_irt`), which fits a
   regularized Bernoulli cross-entropy to direct answer probabilities without
   sampling or thresholding them.

The IRT model is fitted using anchors only. It learns anchor intercepts and
discrimination vectors from training models, then infers each held-out model's
state with the item parameters frozen.

## Decoder comparison

The original TF-IDF logistic interaction is retained. The stronger decoder
uses a frozen sentence-transformer prompt representation and a regularized
low-rank bilinear interaction:

\[
\Pr(y_{m,x}=1)=\sigma(b+f(x)+g(z_m)+(z_m^T U)(V^T e(x))).
\]

This is a practical contextual-MIRT-style decoder: it distinguishes a general
prompt effect, a model-state effect, and a low-rank state-by-prompt effect.
The frozen encoder sees neither benchmark outcomes nor model IDs.

The contextual decoder candidates are selected after the representation sweep
on exploratory LOMO: PCA d8 and factor d4. Their leave-one-lineage-out runs
are confirmation analyses and are not used for a second round of selection.

## Interpretation rules

- Report model-level rate prediction and within-model prompt discrimination
  separately. Pooled row AUROC mixes these quantities.
- Report `full_additive` versus `prompt_plus_metadata` bootstrap intervals;
  this tests state information beyond Think/Instruct/stage/external controls.
- Use LOMO for the primary unseen-model claim and LOLO as robustness only.
- The trajectory command reports associations between already-observed
  adjacent checkpoint changes. It must not be described as a causal result or
  a prospective forecast.

## Sources

- Ruan, Maddison, and Hashimoto, *Observational Scaling Laws and the
  Predictability of Language Model Performance*, NeurIPS 2024 Spotlight.
  https://arxiv.org/abs/2405.10938
- *LLM Evaluation on Unseen Questions: Contextual Multidimensional IRT Model*.
  https://arxiv.org/abs/2608.22295
- Chen et al., *Persona Vectors: Monitoring and Controlling Character Traits
  in Language Models*. https://arxiv.org/abs/2507.21509

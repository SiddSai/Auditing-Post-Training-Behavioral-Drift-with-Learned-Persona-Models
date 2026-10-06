# Frozen single-turn target panel (v1)

This is the first target-side experiment. It tests whether a model's anchor
state can forecast its behavior on a separate panel of raw, single-turn
completion prompts. It does **not** add a system message, chat template, role
markers, answer cue, or few-shot examples.

Run it in order on the GPU VM:

```bash
bash scripts/06_prepare_targets.sh
bash scripts/07_run_targets_4xh100.sh
# wait until 58 metadata JSONs exist
bash scripts/08_collect_and_score_targets.sh
```

`06_prepare_targets.sh` checks out these exact public source revisions, imports
their released prompt fields, freezes a disjoint panel, and creates a
target--anchor lexical-overlap report. It creates exactly 300 development and
150 evaluation prompts from each family: 2,250 targets total.

| Family | Released source field used as raw input | What is held in metadata | Generation cap |
| --- | --- | --- | --- |
| SycophancyEval | sole human content in `datasets/answer.jsonl` | original question, correct/incorrect answer, perturbation template | 64 |
| XSTest | `xstest_prompts.csv:prompt` | released safe/unsafe label and contrast type | 256 |
| Do-Not-Answer | `datasets/data_en.csv:question` | risk area and harm taxonomy | 256 |
| BOLD | prompt JSON continuation string | domain, category, entity | 64 |
| TruthfulQA | `TruthfulQA.csv:Question` | released correct and incorrect answer sets | 128 |

## What “outcome scoring” means here

Generation and scoring are intentionally separate artifacts. A target worker
records the exact greedy completion, stop reason, token IDs, pinned model SHA,
target ID, target-manifest hash, and vLLM configuration. The scorer turns that
completion into the benchmark's outcome only when doing so can faithfully use
a released local method.

The initial scorer ports XSTest's released `strmatch_label` rule byte-for-byte
in substance. Its result is a refusal/compliance label, which is useful both
overall and conditional on XSTest's released safe/unsafe label.

The other datasets should not be silently converted into a generic keyword
score:

- TruthfulQA's generation metric uses its supplied reference answers together
  with an external fine-tuned judge in the original release.
- Do-Not-Answer reports a released Longformer action evaluator (and GPT-4/human
  evaluation); its score must be run as a separately pinned judging stage.
- BOLD is a continuation dataset evaluated with demographic-bias metrics and
  classifiers, not a single gold completion label.
- SycophancyEval's original answer setup was API/message based; our raw-string
  single-turn adaptation preserves its prompts and answer metadata but does not
  pretend its greedy continuation is the paper's original metric.

This is a feature, not unfinished benchmarking: target-generation data can be
published/reused independently, while every judge decision stays explicit and
re-runnable.

## Source-native scoring commands

After collection, install the additional scorer dependencies and prepare the
released scorer inputs:

```bash
pip install -e '.[inference,scoring]'
bash scripts/09_prepare_native_scorers.sh
```

This executes XSTest's released string-match rule and BOLD's recoverable
paper-era VADER component, and writes request files for each source-native
judge. `scripts/10_run_judged_scorers.sh` prints the two intentional paid/API
steps. It is not run automatically because it can create tens of thousands of
judge requests and because the original GPT-4 endpoints are no longer a stable
artifact. The runner records the chosen contemporary judge model, raw response,
and exact source template; it is therefore a **declared protocol adaptation**.

TruthfulQA has an executable Gemini judge in the pinned source checkout and is
run directly by `persona-audit score-truthfulqa-gemini`. Its Gemini model must
be supplied explicitly and is written into the score artifact. BOLD's original
prompt repository has no evaluator code; its paper-era toxicity checkpoint and
some other metric artifacts were not released. We retain this absence in the
provenance rather than labelling a replacement classifier as original BOLD.

## Separating anchors from targets

The Anthropic persona anchors define `z_m`; target texts and labels are not
used while fitting PCA or any later state encoder. The prepare script writes
an auditable exact-text and high token-Jaccard overlap report rather than
automatically deleting data. Review any flags before launch. A later release
will supplement this with a pinned embedding-retrieval audit plus manual review
of its nearest pairs.

## Collection details

Each of the four workers uses one GPU and claims one checkpoint at a time. It
downloads/checks out only that model's immutable Hub revision, writes a
per-model JSONL atomically, deletes the vLLM engine, then claims the next
model. This is why it is safe and efficient on four H100s: models are never
held concurrently by one worker and results can be resumed safely. Do not
mix engine flags or target manifests inside `runs/targets/v1`; start a new run
directory if changing either.

Expected final count: `58 * 2250 = 130,500` completion records.

## Sources

- Sharma et al., SycophancyEval: <https://github.com/meg-tong/sycophancy-eval>
- Röttger et al., XSTest: <https://github.com/paul-rottger/xstest>
- Wang et al., Do-Not-Answer: <https://github.com/Libr-AI/do-not-answer>
- Dhamala et al., BOLD: <https://github.com/amazon-science/bold>
- Lin et al., TruthfulQA: <https://github.com/sylinrl/TruthfulQA>

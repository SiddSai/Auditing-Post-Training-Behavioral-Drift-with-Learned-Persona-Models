# Post-training panel expansion and provenance audit

This repository keeps two different kinds of additional checkpoints separate:

1. **Official trajectory states** are correlated time-ordered observations from
   the same OLMo release process. They are valuable for drift/trajectory
   analyses, but must never be counted as independent model lineages.
2. **Independent observational descendants** are third-party full-weight
   releases. They are useful for external-transfer tests, not causal lineage
   claims.

## Base checkpoints: no chat template

The 40 `base-*` entries are checkpoints of the official OLMo 3 7B
pretraining recipe. That recipe tokenizes pretraining corpora; it does not
specify an assistant role format or chat template. Their correct native
interface is therefore raw completion. This does *not* make raw completion an
appropriate interface for the later Instruct/Think checkpoints: those use the
pinned templates audited below.

Primary sources:

- [`olmo-core` OLMo 3 official recipe](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/README.md)
- [`open-instruct` OLMo 3 documentation](https://github.com/allenai/open-instruct/blob/main/docs/olmo3.md)

## Official stratified trajectory panel

`manifests/posttrain_trajectory_nodes.tsv` contains 44 *additional* official
full checkpoints, all pinned to immutable Hugging Face commits:

| Family | Available intermediate branches | Selected | Reason |
| --- | ---: | ---: | --- |
| Instruct RLVR | 8 | 8 | Small complete trajectory, steps 50--400. |
| Think SFT | 43 | 16 | Approximately uniform coverage from steps 1k--43k. |
| Think RLVR | 55 | 20 | Approximately uniform coverage from steps 25--1375. |

Selection is deliberately stratified rather than exhaustive. A panel made of
every release branch would make precision estimates look artificially strong
by repeatedly sampling adjacent weight states. Use `trajectory_family` to
group resampling and leave-a-whole-family-out controls. The final release
endpoints already live in `manifests/nodes.tsv`; they are not duplicated.

For every added official revision, the Hugging Face revision file inventory was
checked on 2026-10-06. All have three safetensor weight shards, a tokenizer
artifact, a template source, and no `adapter_*` files. This verifies a usable
full checkpoint release; it does not establish bitwise continuity between
unreleased optimizer steps.

Template source audit at the pinned revisions:

| Family | Template source | SHA-256 of source artifact | Count |
| --- | --- | --- | ---: |
| Instruct RLVR | `chat_template.jinja` | `f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b` | 8 |
| Think SFT | `tokenizer_config.json` (embedded template) | `768461353d27a100fe02df01fc2a046fa159217856046eb94d59578838dea5c0` | 16 |
| Think RLVR | `chat_template.jinja` | `6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a` | 20 |

Before inference, regenerate a formal interface manifest with
`persona-audit audit-native-interfaces`; that command hashes the *resolved*
template, applies `chat_template.jinja` precedence as Transformers does, and
will fail if an artifact no longer matches the pinned revision.

## Independent candidate audit

`manifests/posttrain_candidates_audited.tsv` records thirteen candidates and
their admission outcomes. The candidates eligible for the assistant panel have
a declared OLMo 3 Instruct/Think parent, pinned immutable revision, real full
weight file(s), no adapter-only release, and a published chat template. They
are not silently promoted into the primary panel. `admit_after_native_interface_audit`
means: first run the same pinned template audit and inspect rendered prompts;
then include it only as an observational descendant with its parent/author
family as a group.

The `maym15` release is intentionally marked provisional because its card
documents less of its recipe. The JOSIEFIED release is also provisional: its
card recommends a long author-authored identity system prompt, so its native
no-system behavior and its recommended-system behavior need to be reported as
separate interface conditions. `TrashMix` is a merged full-weight release with
a declared QLoRA source and dataset, so it is admitted observationally after
the normal template audit. `Olmo-3FT` is excluded from the main panel because
its card supplies neither training data nor a useful recipe and names an
intermediate Unsloth parent rather than an official OLMo revision.

`Webshop-Olmo3-7B-GRPO-Think-8192` is deliberately deferred rather than
excluded: the root contains a full final policy and a documented sequence of
intermediate policies, but no model-specific tokenizer or template artifact.
It was trained as a multi-turn search/tool agent. It belongs in a future
tool-interface trajectory experiment, where the inherited Think interface and
tool transcript are explicit experimental conditions; it should not be mixed
into the current single-user-turn assistant panel.

The three common failure modes excluded at this stage are adapter-only
repositories, loose claimed lineage with no declared base model, and
repackaged checkpoints without a model-specific interface.

## Analysis rule

Do not pool all of these rows as independent models. Use official trajectory
states for within-family drift and forecasting-style analyses; use outside
descendants for out-of-lineage transfer; cluster uncertainty by trajectory or
author/recipe family. This distinction is central to the paper's validity.

## Runnable external panel

The nine candidates marked `admit_after_native_interface_audit` are now also
present in `manifests/wild_candidates.tsv` as `admit_not_causal`. This is the
manifest consumed by the existing worker scripts, expanding the runnable
external descendant panel from 12 to 21 models. Generate the formal interface
manifest before launching generation; the worker will then render each model
through its own pinned template. The candidate-audit table remains the
authoritative record of why each external model was admitted.

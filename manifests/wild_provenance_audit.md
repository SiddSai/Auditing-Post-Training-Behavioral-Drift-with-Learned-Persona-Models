# Wild candidate provenance audit

Audit date: 2026-10-05.

## Checks and outcome

All 12 listed `longtermrisk/OLMo-3-7B-*` candidates passed the following
repository-level checks at their recorded commit SHA:

| Requirement | Result |
| --- | --- |
| Immutable candidate revision | Pass: each candidate has a 40-character Hub commit SHA. |
| Declared direct parent | Pass: each card declares `unsloth/Olmo-3-7B-Instruct`. |
| Parent chain to official OLMo | Pass at declaration level: `unsloth/Olmo-3-7B-Instruct` declares `allenai/Olmo-3-7B-Instruct`, whose pinned SHA is in `nodes.tsv`. |
| Full model weights | Pass: each repository contains three `model-*-of-00003.safetensors` shards and an index file, not merely an adapter. |
| Chat protocol asset | Pass: each repository contains `chat_template.jinja`. |
| Recipe/data/hyperparameter documentation | Fail: the public card metadata identifies the base model and license but does not document a sufficient training recipe. |

## Decision

Admit all twelve to the **wild observational panel**. They are suitable for
testing whether the learned behavioral state generalizes to independently
created descendant models.

They are **not** admissible as causal intervention edges. Do not interpret an
observed behavioral difference as the effect of the named data condition
(`top20`, `first-third`, `reward-hacks`, etc.) without locating a public
training recipe or contacting the maintainers.

## Parent-chain references

- AI2 Instruct release: `allenai/Olmo-3-7B-Instruct` at
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`.
- Declared intermediate parent: `unsloth/Olmo-3-7B-Instruct` at
  `4953d73ef66ceffa572fce6f22b7346451f2c392`.

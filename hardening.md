Phase-1 autonomous loop scaffold:

- Policy file: [`config/autoloop/default.yaml`](/Users/liambrannigan/playModels/emulator/config/autoloop/default.yaml)
- Controller module: [`src/pipelines/autonomous_unet_experiment_loop.py`](/Users/liambrannigan/playModels/emulator/src/pipelines/autonomous_unet_experiment_loop.py)
- CLI entrypoint: [`src/pipelines/run_autonomous_unet_experiment_loop.py`](/Users/liambrannigan/playModels/emulator/src/pipelines/run_autonomous_unet_experiment_loop.py)

This first version is intentionally narrow:

- config-only hypotheses
- no arbitrary code patching
- sequential training runs only
- hard budgets on:
  - total hypotheses
  - total LLM calls
  - total train runs
  - patch calls, recorded in policy even though phase 1 still keeps `allow_code_patches: false`
- preflight test gate before any training
- JSON ledger plus deterministic markdown summary under `data/interim/autoloop/unet/<batch_id>/`

Implemented controller phases:

1. preflight
2. proposal generation
3. config materialization
4. sequential training
5. batch summary write

Still intentionally not implemented in phase 1:

- branch creation
- code patch / repair loops
- automatic edits to `experiments_*.md`
- result-interpretation LLM call
- concurrent training

Rationale:

This keeps token usage bounded while still giving an autonomous loop that can generate a small batch, validate proposals against the repo's existing search space, run them, and leave behind a structured ledger for review.

Current default policy is intentionally more permissive than the first conservative draft:

- default OpenRouter model: `openai/gpt-5.3-codex`
- default `max_hypotheses`: `8`
- default `max_total_train_runs`: `12`
- default `max_total_llm_calls`: `8`
- default LLM `max_tokens`: `8000`

The controller still enforces those limits deterministically rather than letting the LLM decide them.

Recent hardening additions:

- competitive-run detection is now explicit in policy via `evaluation.competitive_within_ratio`
- if a run lands within that band of the incumbent, the controller writes:
  - `final_step_heatmap.png`
  - `evaluator_payload.json`
- the evaluator payload includes explicit guidance that artifact suppression can justify some loss in raw `eval_mse_mean`, especially for boundary reflections that strengthen and propagate inward
- proposal generation now reminds the model that reducing numerical artifacts is a first-class objective alongside MSE
- the primary experiment log is now append-only JSONL rather than an in-place markdown table
- historical markdown rows can be backfilled into the JSONL log once via `legacy_experiment_markdown_path`
- a deterministic compactor writes a markdown summary of best runs, repeated failures, and useful motifs; this compact file is the artifact the hypothesis generator reviews before proposing the next batch

# Emulator Research Program

This file is the canonical guide for running emulator experiments in this repository.

## Goal

Find a new best emulator for forecasting Aronnax generator experiments, with emphasis on rollout quality rather than one-step fit.

For the current phase:
- target: `layer_thickness`
- primary evaluation: autoregressive rollout
- primary metric: layer-thickness mean squared error over the eval rollout

## Experiment Context

The active generator experiment is part of the research context and must be explicit.

Canonical registry:
- [`experiment_list.md`](/Users/liambrannigan/playModels/emulator/experiment_list.md)

For every emulator run:
- identify the active generator experiment name, for example `double_gyre` or `double_gyre_shifting_wind`
- map it to its standardized branch abbreviation using [`experiment_list.md`](/Users/liambrannigan/playModels/emulator/experiment_list.md)
- use the matching experiment log file, for example [`experiments_double_gyre.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre.md) or [`experiments_double_gyre_shifting_wind.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre_shifting_wind.md)
- compare challengers only against incumbents from the same generator experiment unless the user explicitly asks for cross-experiment comparison

The incumbent is the best emulator currently documented in the experiment-specific log for the active generator experiment.

## Data

- source data lives under experiment-specific roots listed in [`experiment_list.md`](/Users/liambrannigan/playModels/emulator/experiment_list.md)
- a typical source file is `data/raw/<experiment_name>/generator/<experiment_id>/double_gyre.nc`
- current emulator configs live under [`config/emulator`](/Users/liambrannigan/playModels/emulator/config/emulator)
- model implementations live under [`src/models`](/Users/liambrannigan/playModels/emulator/src/models)
- executable entrypoints live under [`src/pipelines`](/Users/liambrannigan/playModels/emulator/src/pipelines)

## Metrics

Primary metrics:
- `eval_mse_mean`
- `eval_mse_per_timestep`

Secondary metrics:
- `eval_mse_std`
- `eval_mse_min`
- `eval_mse_max`
- final-step eval MSE
- qualitative comparison from `comparison.mp4`

Interpretation rule:
- do not rank models by train loss alone
- prefer better rollout behavior over marginal one-step improvements

## Outputs

Each emulator experiment should produce:
- `metrics.json`
- `rollout.nc`
- `comparison.mp4`
- `model.pt`

Default output roots:
- raw artifacts: `data/raw/<experiment_name>/emulator/<emulator_name>`
- checkpoints: `data/interim/emulator/<emulator_name>`

## Installing New Packages

- use `uv add <package>`
- or edit `pyproject.toml` and run `uv sync`
- validate with:
  - `uv run pytest -q`
  - `uv run pytest --cov=src --cov-report=term-missing -q`

## Experimental Methodology

1. State a clear hypothesis before changing code.
2. Identify the active generator experiment and its abbreviation from [`experiment_list.md`](/Users/liambrannigan/playModels/emulator/experiment_list.md).
3. Create a branch from `main` named `YYYY-MM-DD-<abbr>-<emulator_family>-short-hypothesis`.
4. Keep each experiment narrow and change only the challenger.
5. Run tests before trusting results.
6. Run the challenger experiment.
7. Review metrics, rollout error growth, and animation quality.
8. Compare against the incumbent for the same generator experiment.
9. Record the experiment in the matching experiment-specific results file.
10. Update this file if the methodology itself improves.

Example training command:

```bash
uv run python src/pipelines/train_cnn_thickness.py --source-experiment-id <source_experiment_id> --experiment-id <emulator_experiment_id>
```

## Alternative Mode: Hyperparameter Optimization

This mode is off by default.

Only use it when the user gives an explicit instruction such as:
- “do hyperparameter optimization”
- “switch to tuning mode”
- “optimize the current best model”

If the user does not explicitly request this mode, prefer the normal research loop above, which should bias toward exploration, methodology, and interpretable experimental changes.

### Purpose

Use this mode when the goal is no longer to explore qualitatively different emulator ideas, but to take the current best approach for the active generator experiment and improve it through focused parameter tuning.

### Entry Rule

When hyperparameter-optimization mode is activated:
- start from the current incumbent architecture and training setup for the active generator experiment
- keep the data source and evaluation methodology fixed unless the user explicitly asks otherwise
- vary only a small number of parameters that are likely to matter most

### What To Tune

Prioritize parameters with high leverage on rollout quality, for example:
- model width, such as `hidden_channels`
- model depth, such as `num_layers`
- kernel size
- learning rate
- weight decay
- batch size
- epochs
- normalization choices, if normalization is part of the incumbent design

Do not mix architecture exploration and hyperparameter tuning in the same run unless explicitly instructed.

### Tuning Methodology

In this mode:
1. Define a narrow tuning plan before running experiments.
2. Choose a small parameter subset, ideally `1` to `3` parameters at a time.
3. Keep all non-tuned settings fixed to the incumbent values.
4. Record the exact search space in the experiment notes.
5. Compare challengers on the same source full-model run whenever possible.
6. Prefer a small, interpretable search over a large opaque search.

Recommended order:
- first tune optimization parameters
- then tune model size
- only then consider secondary parameters

### Decision Rule

A tuned challenger should replace the incumbent only if it improves the rollout evaluation in a meaningful way, especially:
- lower `eval_mse_mean`
- flatter or improved late-rollout `eval_mse_per_timestep`
- no obvious qualitative regression in the rollout animation

If tuning finds only negligible changes, record that result explicitly and stop rather than continuing a broad search without a new hypothesis.

### Logging Requirements

For each tuning experiment, record in the experiment-specific log:
- the active generator experiment
- that the run was in hyperparameter-optimization mode
- the incumbent model being tuned
- the exact tuned parameters
- the tested values or ranges
- the best result found
- whether the improvement appears meaningful or just marginal

### Guardrail

Do not allow this mode to dominate the whole project.

If several consecutive experiments are only tuning runs, pause and consider whether the next useful step should return to exploration, such as:
- a new architecture
- a new normalization strategy
- a new training target design
- a new rollout loss or curriculum

## Constraints

- prefer simple baselines first
- optimize for clear evaluation before optimizing for raw model quality
- the primary objective is rollout behavior, not just next-step prediction
- use train-only normalization statistics
- keep experiment outputs reproducible and timestamped
- every architecture or training change should have pytest coverage in the affected code path

## Review Checklist

Before concluding an experiment, verify:
- tests pass
- coverage is still strong in changed code
- the active generator experiment is explicit
- the source experiment id is recorded
- the emulator experiment id is recorded
- `metrics.json` includes overall and per-step eval metrics
- `comparison.mp4` renders correctly
- the correct experiment-specific log has a clear writeup

## Tips

- the current tiny CNN baseline is mainly a wiring check for train/eval/rollout/animation
- pay attention to the slope of `eval_mse_per_timestep`, not just its mean
- when comparing experiments, use the same source full-model run unless the hypothesis is explicitly about dataset dependence
- do not mix `double_gyre` and `double_gyre_shifting_wind` incumbents in the same leaderboard
- keep architecture names explicit in code and outputs
- prefer small, isolated changes so experiment outcomes stay interpretable

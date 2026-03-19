# Experiments

| Timestamp | Branch | Commit | Hypothesis | Outcome |
| --- | --- | --- | --- | --- |
| `20260319T220756` | `create-emulator` | `7511cabe362176ca5d5b5e42e51ac9bc73c14a8f` | A very small autoregressive CNN trained only on upper-layer thickness should be cheap to train, achieve a low one-step training loss, and provide a useful baseline for checking whether rollout evaluation, normalization, and artifact generation are wired correctly before trying larger architectures. | `eval_mse_mean = 28.4314`. The per-timestep eval MSE rises almost monotonically from `0.4361` at the first eval step to `66.9035` at the last, which suggests the baseline can track short horizons but accumulates autoregressive error steadily over the rollout. |

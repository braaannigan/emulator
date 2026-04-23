# Methods: Best Incumbent for `double_gyre_shifting_wind_2layer`

## 1. Scope and Incumbent Definition

This document describes the current **incumbent emulator** for the `double_gyre_shifting_wind_2layer` experiment family, defined in the experiment log as run:

- Emulator run: `20260416T140400-dgsw2l-objB1-residual-ss005`
- Metrics artifact: [`metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/20260416T140400-dgsw2l-objB1-residual-ss005/metrics.json)
- Config artifact: [`unet_thickness_shifting_wind_2layer_input_current_plus_residual_wind_objB1_residual_ss005_early30_manual.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/unet_thickness_shifting_wind_2layer_input_current_plus_residual_wind_objB1_residual_ss005_early30_manual.yaml)
- Source generator run: `20260408T071500-period40d-duration5000d-2layer`

Important exclusion: an older run (`20260407T225500-dgsw2l-unet-benchmark-fg`) reported lower MSE but was explicitly invalidated because it used a `400`-day source instead of the intended `5000`-day source.

## 2. Physical Problem and Learning Target

The emulator approximates the forced two-layer shallow-water evolution operator on a regular horizontal grid.

Prognostic variables are predicted jointly:

- `layer_thickness` (2 channels, one per layer)
- `zonal_velocity_centered` (2 channels)
- `meridional_velocity_centered` (2 channels)

Total prognostic channels: `6`.

The primary reported skill metric is autoregressive rollout MSE for `layer_thickness` over a 250-day evaluation window (`eval_mse_mean`).

## 3. Data, Temporal Partitioning, and Evaluation Window

Data source:

- Root: `data/raw/double_gyre_shifting_wind_2layer/generator`
- Experiment: `20260408T071500-period40d-duration5000d-2layer`
- File: `double_gyre.nc`

Grid and cadence observed in rollout artifacts:

- Horizontal grid: `200 x 100` (`y x x`)
- Horizontal coordinates: 10-km spacing (5,000 m to 1,995,000 m in `y`; 5,000 m to 995,000 m in `x`)
- Saved temporal cadence: 7 days per frame

Temporal preprocessing and split:

- Initial spin-up discarded: first `100` days (`train_start_day = 100`)
- Chronological split: `train_fraction = 0.8`
- Resulting sequence counts in incumbent run:
- Train timesteps: `560`
- Eval timesteps: `36`

Evaluation horizon:

- `eval_window_days = 250`
- Eval timestamps in incumbent metrics span day `4032.0068` to day `4277.0068` (36 points).

## 4. Input-Output Formulation

### 4.1 State Construction

The model uses `state_history = 2` with `state_input_mode = current_plus_residual`.

For each forecast origin \(t\):

- Current state \(x_t\) (6 channels)
- Previous state \(x_{t-1}\) (6 channels)
- Residual state \(\Delta x_t = x_t - x_{t-1}\) (6 channels)

State tensor passed to the network:

\[
s_t = [x_t, \Delta x_t] \in \mathbb{R}^{12 \times H \times W}.
\]

### 4.2 Forcing Construction

Forcing mode is `wind_current`, which contributes one additional channel \(f_t\).

Final network input:

\[
u_t = [s_t, f_t] \in \mathbb{R}^{13 \times H \times W}.
\]

### 4.3 Multi-Step Output

The network predicts `output_steps = 2` future prognostic states per forward pass:

\[
\hat{X}_{t+1:t+2} \in \mathbb{R}^{2 \times 6 \times H \times W}.
\]

## 5. Network Architecture

Model family: `UnetThicknessModel` (`src/models/unet_thickness/model.py`).

Incumbent architecture hyperparameters:

- `hidden_channels = 24`
- `num_levels = 3`
- `kernel_size = 5`
- `block_type = convnext`
- `stage_depth = 2`
- `dilation_cycle = 4`
- `norm_type = groupnorm`
- `fusion_mode = input`
- `skip_fusion_mode = add`
- `upsample_mode = bilinear`
- `residual_connection = true`
- `residual_step_scale = 0.9`
- `boundary_padding_mode = zeros` (default)

### 5.1 Encoder-Decoder Channel Topology

With `num_levels=3`, encoder widths are:

- Level 1: 24
- Level 2: 48
- Level 3: 96
- Bottleneck: 192

Decoder mirrors this with bilinear upsampling + `1x1` projection and additive skip fusion.

### 5.2 ConvNeXt-Style Block Used Here

Each repeated block is:

1. Depthwise \(k \times k\) convolution (with dilation \(d\))
2. GroupNorm(1, C) (channel-wise LayerNorm equivalent)
3. Pointwise MLP \(C \rightarrow 4C \rightarrow C\) with GELU
4. Residual addition

Dilation schedule cycles as \(1,2,4,8\) across successive repeated blocks due to `dilation_cycle=4`.

### 5.3 Output Head and Residual Integration

The output head is a `1x1` convolution to \(2 \times 6 = 12\) channels, reshaped to \((\text{steps}=2,\text{channels}=6,H,W)\).

Residual update rule:

\[
\hat{x}_{t+\tau} = x_t + \alpha \, r_{t,\tau}, \quad \alpha=0.9,\ \tau \in \{1,2\},
\]

where \(r_{t,\tau}\) is the raw network output for step \(\tau\).

Model size for this exact configuration: **1,061,268 trainable parameters**.

## 6. Normalization and Leakage Control

State and forcing channels are normalized using statistics fit on the **training split only**.

- State standardizer: per-channel mean/std from train frames
- Forcing standardizer: per-channel mean/std from train forcing features

No evaluation data are used in normalization fitting.

## 7. Training Objective and Optimization

### 7.1 Objective

Incumbent objective mode is `residual`:

- `state_loss_weight = 0.0`
- `residual_loss_weight = 1.0`
- Base loss: MSE

For each rollout training step:

\[
L = \frac{1}{K}\sum_{\tau=1}^{K}
\left\|\left(\hat{x}_{t+\tau} - x_{t+\tau-1}^{*}\right) -
\left(x_{t+\tau}^{*} - x_{t+\tau-1}^{*}\right)\right\|_2^2,
\]

with \(K\) the current rollout horizon and \(x^*\) denoting ground truth.

No high-frequency penalty is active (`high_frequency_loss_weight = 0.0`).

### 7.2 Optimizer and Batching

- Optimizer: Adam
- Learning rate: `3.8e-4`
- Weight decay: `1.8e-5`
- Batch size: `2`
- Epoch budget: `30`
- Random seed: `7`

Incumbent run summary:

- `optimization_steps = 8350`
- `train_loss = 0.0026523` at completion

## 8. Rollout Curriculum and Scheduled Sampling

Curriculum is explicit and time-indexed:

- `curriculum_rollout_steps = [2, 4]`
- `curriculum_transition_epochs = [0, 10]`

Hence:

- Epochs 1-10 use rollout horizon 2
- Epochs 11-30 use rollout horizon 4

Scheduled sampling:

- `scheduled_sampling_max_prob = 0.05`
- Applied with linear ramp over epochs in training code.

This setting was the key improvement over the preceding near-best current+residual baseline.

## 9. Early-Stopping Monitor (Active but Non-Triggered)

This run used periodic guardrails every 5 epochs:

- `early_stopping_eval_interval_epochs = 5`
- Reference metrics path:
  `.../20260416T080000-dgsw2l-input-current-plus-residual/metrics.json`
- Margin schedule:
  `early_stopping_margin_start = 0.4`
  with `early_stopping_margin_decay = 0.8` per periodic check

Periodic eval sequence (mean MSE): `125.90 → 79.93 → 53.35 → 48.07 → 29.36 → 25.04`.

No threshold breach occurred; training completed all 30 epochs.

## 10. Autoregressive Evaluation Protocol

Evaluation uses fully autoregressive rollout on the eval split:

1. Seed model with first eval state context.
2. Advance model recursively using its own predictions.
3. Inject forcing according to `wind_current` at each step.
4. Compare predicted and truth fields over the 250-day eval window.

Primary reported metric:

\[
\text{eval\_mse\_mean} =
\frac{1}{T}\sum_{t=1}^{T}\text{MSE}\left(\hat{h}_t,h_t\right),
\]

where \(h\) denotes two-channel layer thickness.

Incumbent outcome:

- `eval_mse_mean = 25.0388`
- `eval_mse_min = 2.3158`
- `eval_mse_max = 60.5119`
- Final-step eval MSE: `60.5119`

The per-timestep curve shows monotonic error growth typical of autoregressive accumulation, but with materially lower level than prior documented incumbents in this experiment family.

## 11. Reproducibility Assets

Configuration and outputs required to reproduce this exact incumbent:

- Config:
  [`unet_thickness_shifting_wind_2layer_input_current_plus_residual_wind_objB1_residual_ss005_early30_manual.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/unet_thickness_shifting_wind_2layer_input_current_plus_residual_wind_objB1_residual_ss005_early30_manual.yaml)
- Metrics:
  [`metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/20260416T140400-dgsw2l-objB1-residual-ss005/metrics.json)
- Rollout fields:
  [`rollout.nc`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/20260416T140400-dgsw2l-objB1-residual-ss005/rollout.nc)
- Animation:
  [`comparison.mp4`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/20260416T140400-dgsw2l-objB1-residual-ss005/comparison.mp4)

Training entrypoint:

```bash
.venv/bin/python src/pipelines/train_unet_thickness.py \
  --config config/emulator/unet_thickness_shifting_wind_2layer_input_current_plus_residual_wind_objB1_residual_ss005_early30_manual.yaml \
  --experiment-id <new_run_id>
```

## 12. Methodological Limits of the Incumbent

The incumbent remains experiment-specific and does not yet eliminate long-rollout boundary artifact growth. It is a strong autoregressive surrogate for the current 2-layer forcing regime, but not a general-purpose ocean emulator outside this training distribution.

# Methods

## Overview

Our current best emulator is designed to predict the evolution of a forced, single-layer shallow-water system under time-varying wind stress. The model is trained on output from the `double_gyre_shifting_wind` Aronnax generator experiment and is intended to act as a fast surrogate for the numerical model over multi-month to year-scale rollouts.

The core idea is simple: instead of solving the shallow-water equations directly at every timestep, we train a neural network to approximate the map from the recent ocean state and current wind forcing to the next few ocean states. The best-performing version of this approach is a residual, multi-step, dilated U-Net with ConvNeXt-style blocks.

Although this is a machine-learning model, it is not intended as a black-box replacement for all physics. It is a data-driven forecast operator trained to imitate the generator on the specific shallow-water regime represented in the training data.

## Physical Setting

The target system is a one-layer reduced-gravity shallow-water model on a beta plane, forced by wind stress. The prognostic state used by the emulator consists of three gridded fields:

- layer thickness
- zonal velocity
- meridional velocity

The forcing supplied to the emulator is the current wind stress field. Because this is a single-layer problem with wind forcing, the model does not attempt to represent vertical structure, thermodynamics, or coupled atmosphere-ocean feedbacks.

## Learning Problem

At each forecast step, the emulator receives:

- the current state
- the immediately preceding state
- the current wind forcing

and predicts the next two states of the prognostic variables.

In other words, the emulator is not a one-step predictor that only sees a single snapshot. It uses short temporal memory and produces a short forecast segment in one forward pass. This proved important for reducing myopic behavior and improving autoregressive rollout quality.

## Data and Training Split

The current incumbent is trained on the generator run:

- source experiment: `20260323T123500-period300d-duration5000d-shift160km`

The first `100` days are excluded from training and evaluation. This avoids forcing the model to learn from the early transient adjustment period and instead focuses training on the more settled regime that dominates the long simulation.

After this initial trim, the time sequence is split chronologically:

- `80%` for training
- `20%` for evaluation

This preserves the forecasting structure of the problem. The model is therefore always evaluated on future states that occur after the training segment, rather than on randomly sampled frames.

## Inputs and Outputs

### State Variables

The model uses the following state fields:

- `layer_thickness`
- `zonal_velocity_centered`
- `meridional_velocity_centered`

Two consecutive states are supplied as input (`state_history = 2`). This gives the network enough information to infer short-term tendencies and propagation, rather than trying to estimate the future from a single instantaneous field.

### Forcing

The wind enters as an explicit input (`forcing_mode = wind_current`). This is important because the gyre response is driven by external forcing, not only by internal state evolution. The best model injects forcing at the input stage (`fusion_mode = input`), so the network sees the state and forcing together from the first layer onward.

### Predicted Quantities

The network predicts the full prognostic state jointly:

- layer thickness
- zonal velocity
- meridional velocity

Joint prediction matters because these fields are dynamically coupled in the shallow-water system. Predicting them together encourages the emulator to learn a coherent update rather than treating thickness alone as the whole problem.

## Model Architecture

### Backbone

The current incumbent uses a U-Net architecture. U-Nets combine two useful ideas:

- an encoder that compresses the field into progressively coarser spatial representations
- a decoder that reconstructs a high-resolution output while reusing fine-scale information through skip connections

This is a natural fit for shallow-water dynamics because the flow contains both local structure and basin-scale organization. The encoder captures broad spatial context, while the skip connections preserve sharper local detail.

### Specific Architecture Choices

The best model uses:

- `hidden_channels = 24`
- `num_levels = 4`
- `kernel_size = 5`
- `block_type = convnext`
- `stage_depth = 2`
- `dilation_cycle = 4`
- `skip_fusion_mode = concat`
- `upsample_mode = transpose`

These choices define a moderately deep and multiscale convolutional network with an enlarged effective receptive field.

### ConvNeXt-Style Blocks

Within each stage, the model uses ConvNeXt-style blocks rather than plain stacked convolutions. Each block applies:

- a depthwise spatial convolution
- normalization
- nonlinear activation
- pointwise channel mixing
- a residual connection around the block

This structure improves expressiveness without making the model excessively large. In practice, it was more effective than simpler convolutional blocks in this problem family.

### Enlarged Receptive Field

The model uses a kernel size of `5` and a dilation cycle of `4`. Dilation means that some convolutions sample points farther apart on the grid, which increases the effective spatial reach of the network without requiring a much larger number of parameters.

For this shallow-water problem, that matters because the wind-driven response is not purely local. Gyre adjustment involves spatially extended structures, and the emulator benefits from being able to sense a broader surrounding region when predicting tendencies.

### Skip Connections

The U-Net decoder uses concatenation-based skip connections. These pass higher-resolution features from the encoder to the decoder so that coarse, large-scale information can be combined with fine, local structure during reconstruction.

This is especially useful when predicting velocity and thickness fields together, because the emulator must preserve sharp local gradients while still representing basin-scale circulation.

## Residual Formulation

One of the most important design choices is that the network predicts state increments rather than absolute states (`residual_connection = true`).

Operationally, the network computes a correction to the current state, and that correction is added back to the input state to obtain the forecast.

This is a strong inductive bias for a shallow-water model. Over a single output interval, most of the state usually persists, and the main task is to predict how advection, pressure-gradient effects, drag, and wind forcing change that state. Asking the network to predict the increment is therefore closer to the structure of the numerical problem than asking it to reconstruct the entire next field from scratch.

This residual formulation was a major contributor to the current best results.

## Multi-Step Prediction

The incumbent predicts two future states in one forward pass (`output_steps = 2`). This turns the model into a short-horizon transition operator rather than a purely one-step regressor.

This matters because the eventual use case is autoregressive rollout. A model that is only trained to make one-step predictions can perform well locally while still drifting badly when iterated. Producing two steps at once encourages the network to learn a slightly broader transition map and reduces short-horizon myopia.

## Normalization

The state channels are standardized using statistics fitted on the training segment only. Wind-forcing channels are standardized separately, again using only the training portion of the data. This prevents evaluation information from leaking into preprocessing and makes the training optimization more stable.

Normalization is a numerical convenience rather than a physical statement. After prediction, the outputs are returned to the original physical scale for evaluation.

## Training Procedure

### Loss

The model is trained with mean squared error between predicted and target fields. The current incumbent does not use extra high-frequency penalties or auxiliary spectral losses.

### Optimizer

Training uses Adam with:

- learning rate `3.8e-4`
- weight decay `1.8e-5`

An important result of the recent search process is that the model family itself was already strong, and a large share of the final improvement came from moving to a better optimizer setting inside the same architecture and training regime.

### Batch Size and Epochs

The current incumbent uses:

- batch size `2`
- `20` training epochs

Although these are ordinary hyperparameters, they should be viewed as part of the current validated recipe rather than arbitrary defaults.

## Rollout Curriculum

The model is trained with a rollout curriculum rather than a fixed forecast horizon throughout training.

The training schedule uses:

- rollout steps `(2, 4)`
- transition epochs `(0, 15)`

This means:

- from the start of training, the model is optimized over short rollouts
- after epoch `15`, training shifts to a longer effective rollout horizon

The purpose is to first learn a stable local transition operator and then gradually require it to remain accurate when iterated further. In this project, this delayed horizon increase was noticeably more effective than forcing the harder rollout regime too early.

Scheduled sampling is disabled in the incumbent (`scheduled_sampling_max_prob = 0.0`) because it did not help in the experiments that were tried.

## Evaluation

The primary evaluation is autoregressive rollout, not one-step prediction. After training, the model is seeded with the initial evaluation state and then rolled forward using its own predictions, together with the prescribed wind forcing.

The main reported metric is:

- mean squared error of layer thickness over the evaluation rollout (`eval_mse_mean`)

Additional diagnostics include:

- per-timestep rollout error
- final-step error
- qualitative animation of truth versus rollout

For the current incumbent, the standard evaluation window is `250` days. A separate `500`-day rollout was also examined to assess longer-horizon degradation. That longer test showed that the model remains stable but still accumulates noticeable drift, so long-horizon autoregressive error growth remains the main limitation.

## Computational Cost

One motivation for this emulator is speed. We therefore measured the wall-clock cost of advancing the current incumbent and compared it with the Aronnax generator on the same machine.

For the current best `double_gyre_shifting_wind` emulator, an autoregressive rollout over `139` evaluation steps took about `1.15 s` on Apple `MPS`, corresponding to about `0.0083 s` per emulator rollout step. In this dataset, one emulator rollout step corresponds to one saved generator interval, that is, `7` simulated days.

For the coarse shifting-wind Aronnax generator (`100 x 200` grid, `dt = 600 s`), a short benchmark run required about `10.3 s` to simulate `20` physical days. This corresponds to about `0.516 s` per simulated day, or about `3.61 s` per `7` simulated days.

Under these conditions, the emulator is approximately `4.3 x 10^2` times faster than the generator for advancing the system by the same amount of simulated physical time.

This estimate should be interpreted carefully. The generator timing is an end-to-end benchmark and therefore includes initialization and output overhead, while the emulator timing is a pure rollout measurement after model loading. The exact speedup will vary by hardware and by how much startup cost is amortized over a long generator integration. Even so, the practical conclusion is robust: the trained emulator advances this shallow-water system on the order of a few hundred times faster than the numerical model on the same machine.

## Current Incumbent Configuration

The best documented shifting-wind emulator at present uses:

- U-Net with ConvNeXt-style blocks
- `4` spatial levels
- `24` base channels
- kernel size `5`
- dilation cycle `4`
- joint prediction of thickness, zonal velocity, and meridional velocity
- two-frame state history
- two-step output operator
- explicit current wind forcing
- residual prediction
- delayed `2 -> 4` rollout curriculum
- Adam optimizer with learning rate `3.8e-4`
- weight decay `1.8e-5`

This model achieved the best documented result on the active `double_gyre_shifting_wind` benchmark, with strong improvement over earlier one-step and non-residual variants.

## Interpretation

The present model encodes several assumptions about the shallow-water forecasting problem:

- the next state is mostly a correction to the current state, not a complete replacement
- recent state history contains useful dynamical information
- wind forcing should be supplied explicitly as a causal driver
- thickness and velocity should be forecast together
- useful forecast skill depends on both local structure and larger-scale spatial context
- training should emphasize autoregressive behavior, not just local one-step fit

These assumptions are consistent with the structure of the underlying shallow-water system and help explain why this model outperformed simpler CNNs, plain U-Nets, and less structured training setups.

## Limitations

This emulator should still be understood as a learned surrogate for a specific numerical experiment family, not as a universal shallow-water solver.

Its main limitations are:

- it is trained on a single-layer system
- it only handles the forcing regime represented in the training data
- it is optimized primarily for a `250`-day evaluation window
- rollout error still grows substantially on longer `500`-day tests

The model is therefore best viewed as a strong experiment-specific emulator with physically informed inductive biases, rather than as a general replacement for the governing equations.

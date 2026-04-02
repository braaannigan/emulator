# Future Directions

This note collects architectural directions that seem genuinely promising for the current `double_gyre_shifting_wind` emulator line. The aim is not to list every possible machine-learning idea, but to capture the changes that could plausibly move us into a new regime beyond incremental U-Net tuning.

The current incumbent is already a strong residual, multi-step, dilated ConvNeXt U-Net. That means the most interesting next steps are the ones that change the transition operator or the inductive bias, rather than simply making the existing network a bit wider or deeper.

## 1. Conservative or Constraint-Aware Output Heads

The most physically grounded next step would be to make conservation structure more explicit.

Instead of predicting the next state directly, the model could predict:

- fluxes
- tendencies
- or a constrained update that preserves integral quantities by construction

For a shallow-water system, this is appealing because long-rollout drift is one of the main remaining problems. A conservative head could directly target that weakness rather than only hoping the network learns conservation implicitly.

Why it is interesting:

- attacks drift at the level of model structure
- fits the PDE character of the problem
- could improve long-horizon stability more than another optimizer sweep

## 2. Hybrid Spectral and Local Convolutions

The current model handles larger spatial scales through pooling and dilation. That works, but it is still fundamentally local.

A stronger alternative is a hybrid architecture that combines:

- ordinary local convolutions for small-scale spatial structure
- spectral or Fourier-style blocks for basin-scale organization

This could be useful because gyre adjustment and large-scale propagation are not purely local phenomena. A spectral pathway may capture domain-wide structure more naturally than only dilated convolutions.

Possible families:

- Fourier Neural Operator style blocks
- Adaptive Fourier Neural Operator style blocks
- hybrid ConvNet plus FFT blocks

Why it is interesting:

- gives the model a more natural mechanism for large-scale coupling
- may improve coherence of basin-scale responses
- is a clearer architectural leap than another ConvNeXt variant

## 3. Advection-Aware or Semi-Lagrangian Updates

One limitation of a standard U-Net is that it predicts the future entirely in an Eulerian, image-to-image fashion.

A more flow-aware alternative is to introduce an explicit learned transport mechanism, for example:

- warp latent features along a learned velocity field
- advect the current state or hidden state before correction
- separate transport and correction into two stages

This is attractive because shallow-water dynamics have a strong transport component. The model may benefit from an inductive bias that says “move information first, then adjust it,” rather than learning both simultaneously in one generic convolutional operator.

Why it is interesting:

- matches the structure of the dynamics more closely
- may improve phase accuracy and pattern propagation
- could reduce the burden on the network to learn advection implicitly

## 4. Multi-Branch Physics Decomposition

Another promising direction is to decompose the update into several interacting branches rather than using one monolithic trunk.

For example:

- one branch for transport and advection
- one branch for forcing response
- one branch for pressure-gradient or gravity-wave adjustment

The outputs of these branches could then be combined into the final residual update.

This would not make the model fully interpretable in a strict physical sense, but it could encourage a cleaner internal representation of the different mechanisms driving the shallow-water evolution.

Why it is interesting:

- imposes useful structure on the forecast operator
- may improve interpretability of what the model is doing
- could help separate fast and slow components of the response

## 5. Recurrent Latent Memory

The current model uses `state_history = 2`, which gives it only a short explicit memory through raw input frames.

A stronger alternative is to give the emulator an evolving latent memory state, for example with:

- ConvGRU-style recurrence
- latent state-space recurrence
- recurrent hidden states propagated across rollout steps

This may help the emulator retain information about recent history without forcing all memory into the raw state inputs.

Why it is interesting:

- gives a more flexible temporal memory than simply stacking frames
- may help longer autoregressive rollouts
- could support better separation between observed state and latent tendency information

## 6. Multi-Timescale Models

The shallow-water system contains processes with different timescales. Some responses are relatively fast, while gyre adjustment unfolds more slowly.

One possible architectural response is to build a model with separate components for:

- fast adjustments
- slower background evolution

These could be combined additively or hierarchically.

This is especially relevant because one of the hard problems in emulator design is maintaining both short-term responsiveness and long-term stability.

Why it is interesting:

- reflects the temporal structure of the dynamics
- may reduce the tension between immediate accuracy and long-rollout behavior
- could be implemented either in the state update or in a latent representation

## 7. More Structured Cross-Variable Coupling

The current incumbent already predicts `layer_thickness`, `u`, and `v` jointly, which is important. But the coupling between these variables is still largely implicit inside a generic shared backbone.

A more structured alternative would be to include explicit coupling modules between:

- thickness and momentum channels
- transport-related and pressure-related features

This might help the model learn more coherent joint updates across the prognostic state.

Why it is interesting:

- strengthens the coupling prior without changing the data regime
- may improve physical consistency between `h`, `u`, and `v`
- is less radical than a full new operator family but more meaningful than simple width changes

## 8. Learned Correction on Top of a Cheap Numerical Core

This is perhaps the most physically appealing “big step” direction.

Instead of learning the full transition map from scratch, use:

- a simple coarse or approximate shallow-water update as the baseline
- a neural network to predict only the correction term

This turns the emulator into an ML-augmented numerical model rather than a pure black-box surrogate.

Why it is interesting:

- leverages known physics directly
- may improve stability and extrapolation
- could reduce the amount of learning needed for basic transport and wave behavior

For this project, this may be the most natural route if the goal shifts from pure surrogate performance toward more physically trustworthy long rollouts.

## 9. Boundary-Aware Convolutions

This is less important than in a global ocean model, but still worth noting.

Standard padding is often an unexamined weakness in PDE surrogate models. A boundary-aware treatment could help if the model is using unrealistic edge behavior near walls or boundaries.

Possible approaches:

- masked convolutions
- custom padding rules
- separate boundary channels or boundary-condition features

Why it is interesting:

- specifically targets a common convolutional failure mode
- may matter if wall effects are important in the learned rollouts

## Priority Ranking

If only a few of these ideas are worth pursuing soon, the most promising shortlist is:

1. Conservative or flux-form output heads
2. Hybrid spectral plus local blocks
3. Advection-aware or semi-Lagrangian transport modules
4. Recurrent latent memory
5. Learned correction on top of a simple numerical baseline

These directions are more likely to produce a real step-change than another round of depth, width, or optimizer tuning.

## Working Principle

The key lesson from the current research cycle is that the project has probably already found the right broad architecture family for the present benchmark. That means the next useful experiments should not be random novelty.

They should target one of the following weaknesses explicitly:

- long-horizon drift
- imperfect large-scale propagation
- limited temporal memory
- insufficient physical structure in the update rule

Any future architectural experiment should therefore be justified in terms of which of those weaknesses it is trying to address.

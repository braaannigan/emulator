# Future Directions: Model Design

This note focuses on model-design directions that can move the emulator beyond incremental hyperparameter tuning.

Recent experiments suggest the key bottleneck is not only scalar error, but rollout behavior: boundary reflections strengthen over time and propagate inward. The next model iterations should therefore target both forecast skill and structural robustness.

## Latest Evidence (2026-04-15 Manual Curriculum Batch)

Most recent `double_gyre_shifting_wind_2layer` evidence:

- best run: `20260415T075100-dgsw2l-bestseed-curriculum-early-30ep` with `eval_mse_mean = 37.8524`
- second-best: `20260415T074300-dgsw2l-bestseed-curriculum-ss01` with `eval_mse_mean = 45.1809`
- delayed and three-stage curricula underperformed badly (`eval_mse_mean ~= 89.14` and `109.70`) and repeatedly triggered early-stop thresholds

Implication for next wave:

- keep the strong `early` curriculum as the default training scaffold
- explore higher-upside architecture/mechanism changes on top of that scaffold, rather than further narrow optimizer-only tuning

## Latest Evidence (2026-04-16 Architecture Batch)

Architecture-focused follow-up on the same `early + 30 epoch` scaffold did not surpass the incumbent:

- best architecture challenger (`bottleneck forcing + gated skip + kernel7/dilation2`) reached `eval_mse_mean = 45.5575`, still worse than `37.8524`
- other architecture variants regressed materially, with two failing at epoch `5` (`eval_mse_mean > 200`)
- boundary artifacts are still present in rollout inspection, including the incumbent final-step heatmap

Implication for next wave:

- continue architecture exploration, but narrow to one mechanism change at a time around the strongest challenger line (`bottleneck forcing` path)
- prioritize explicit boundary-aware operators and artifact-focused evaluation signals, not MSE-only selection

## 1. Boundary-Aware Update Operators

The strongest immediate design opportunity is to make boundary handling explicit.

Candidate designs:

- boundary mask channels injected at each encoder/decoder stage
- boundary-conditioned skip gating (learned attenuation near edges)
- masked or boundary-aware convolutions in early and late blocks

Why this matters:

- directly targets the dominant observed artifact
- gives the model an explicit mechanism instead of asking generic convs to infer boundary rules implicitly

## 2. Flux-Form or Tendency-Form Prediction Heads

Instead of predicting next state directly, predict structured updates:

- fluxes with reconstruction of state increments
- per-variable tendencies added to current state
- constrained residual heads with stability-aware scaling

Why this matters:

- improves physical plausibility of updates
- can reduce long-horizon drift and ringing from unconstrained direct mapping

## 3. Transport-First Architectures

Current U-Net updates are purely image-to-image. A transport-aware structure may better match shallow-water dynamics.

Candidate designs:

- two-stage block: learned advection/warp then correction
- latent transport module conditioned on velocity channels
- split branch for transport and source-term correction

Why this matters:

- aligns with dynamics structure ("move then adjust")
- may improve phase and propagation fidelity

## 4. Multi-Scale Global-Local Coupling

Boundary and basin-scale coherence can be under-modeled by local convolutions alone.

Candidate designs:

- hybrid local conv + spectral block at bottleneck
- low-frequency global pathway fused with local pathway
- coarse-grid correction branch for large-scale structure

Why this matters:

- improves long-range coupling
- can stabilize large-scale modes without over-smoothing local detail

## 5. Structured Cross-Variable Coupling

`h`, `u`, and `v` are currently coupled mostly through a shared trunk. More explicit coupling may improve coherence.

Candidate designs:

- cross-variable interaction blocks (thickness<->momentum exchange layers)
- separate variable encoders with learned coupling modules
- variable-specific heads with shared physical interaction trunk

Why this matters:

- encourages physically consistent joint updates
- may reduce channel-specific artifacts that later amplify in rollout

## 6. Stability-Governed Recurrence

Current state history is short and explicit (`state_history=2`). Recurrence can provide controlled temporal memory.

Candidate designs:

- ConvGRU latent memory inside decoder or bottleneck
- gated residual recurrence with norm constraints
- memory branch used only for correction term, not full state synthesis

Why this matters:

- improves temporal consistency
- can reduce late-rollout degradation after initially good short-horizon behavior

## 7. Curriculum Learning as a Design Lever

Curriculum strategy should be treated as part of model design, not only training configuration.

Candidate curriculum families:

- rollout-horizon curriculum:
  start with short horizons, increase to longer autoregressive windows
- scheduled-sampling curriculum:
  increase model-input replacement probability over training
- region-aware curriculum:
  weight boundary bands earlier, then rebalance toward full-domain accuracy
- error-triggered adaptive curriculum:
  advance horizon only when intermediate stability/quality criteria are met

Why this matters:

- directly targets the "good early, unstable late" behavior seen in multiple runs
- can improve robustness without forcing immediate architectural complexity
- creates a cleaner path to evaluate architecture changes under consistent stability pressure

## Priority Design Roadmap

## Near-Term (highest leverage)

1. Boundary-aware update operators
2. Flux/tendency-form head
3. Curriculum-learning variants focused on rollout stability
4. Transport-first split block

## Mid-Term

1. Structured cross-variable coupling
2. Multi-scale global-local coupling

## Later

1. Stability-governed recurrence integrated with best prior design

## Evaluation Principle for New Designs

Any design change should be judged on both:

- forecast error (`eval_mse_mean`, horizon profile)
- rollout quality (boundary reflection growth, interior contamination, visual coherence)

A model design should be considered a forward step if it meaningfully improves rollout behavior while staying competitive on error, even if it is not the absolute lowest MSE candidate.

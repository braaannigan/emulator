# Future Directions: Long-Horizon Research Program

This program is based on the broader project trajectory, not only the latest batch.  
It reflects repeated findings across manual and autonomous runs:
- best gains came from targeted input/objective changes (for example `current + residual + wind`, residual-focused losses)
- many promising early checkpoints degraded later in rollout
- boundary/reflection artifacts remain a first-order failure mode
- narrow tuning alone has not delivered reliable step-change improvements

Below are 10 research areas, intentionally split across lower-risk and higher-risk work.

## 1) Rollout-First Model Selection and Checkpointing (Lower Risk)

Move from final-epoch selection to rollout-aware checkpoint selection:
- choose checkpoints by periodic rollout eval (or EMA-smoothed periodic eval), not terminal epoch
- enforce degradation-aware early-stop rules (`best -> current` drift)
- standardize reference thresholds (rolling reference windows) across experiments

Why now: repeated late-epoch regressions show selection policy is a major leverage point.

## 2) Exposure-to-Generated-States Curriculum (Lower Risk)

Systematically optimize exposure bias controls:
- scheduled sampling schedules emphasizing self-fed training
- curriculum transition timing and smoothness tuned jointly with exposure settings
- adaptive progression rules based on stability criteria, not fixed epoch schedules

Why now: this has strong theoretical alignment and low implementation risk.

## 3) One-Step Recursive Training Line (Medium Risk)

Develop the recursive `output_steps=1` line as a first-class family:
- strict self-feeding variants (N-BEATS-like recursive forecasting behavior)
- hybrid recursive + curriculum variants
- compare against multi-output heads on both final MSE and stability profile

Why now: directly targets compounding-error mechanics seen in rollout.

## 4) Residual-Bias and Integral Control (Medium Risk)

Add explicit controls for residual drift:
- zero-mean residual projection
- global bias-correction head on predicted tendency
- domain-integral consistency penalties

Why now: residual diagnostics showed nontrivial bias/integral drift despite decent state-field visuals.

## 5) Boundary-Aware Loss and Operator Design (Medium Risk)

Treat boundary artifacts as co-equal objective:
- boundary/interior weighted loss scheduling
- boundary mask channels and boundary-conditioned skip fusion
- artifact metrics as gating criteria for model acceptance

Why now: boundary reflections remain unresolved and materially affect rollout trust.

## 6) Multi-Objective Training Beyond MSE (Medium Risk)

Expand objective to include structure-preserving terms:
- spectral-shape penalties for residual updates
- gradient/phase-consistency losses
- controlled trade-off curves between rollout cleanliness and mean MSE

Why now: some runs are MSE-competitive but structurally flawed in residual/rollout behavior.

## 7) Transport-Structured Update Blocks (Higher Risk)

Move beyond pure image-to-image updates:
- two-stage “advect then correct” modules
- velocity-conditioned transport branch with correction branch
- tendency-form heads with explicit update composition

Why now: aligns architecture with governing dynamics; potentially high upside, higher implementation risk.

## 8) Multi-Scale Global-Local Coupling (Higher Risk)

Add mechanisms for basin-scale coherence:
- global/spectral bottleneck pathway fused with local conv pathway
- coarse-grid correction branch for long-range coupling
- anti-ringing constraints in low-frequency channels

Why now: long-horizon artifacts likely involve under-modeled global coupling.

## 9) Structured Cross-Variable Coupling (Higher Risk)

Explicitly model interactions among `h`, `u`, and `v`:
- variable-specific encoders + learned coupling modules
- constrained interaction blocks (thickness-momentum exchange)
- variable-aware heads with shared physical trunk

Why now: current shared-trunk coupling may be too implicit for stable long rollouts.

## 10) Temporal Memory Architecture Upgrade (Higher Risk)

Test controlled recurrent memory:
- ConvGRU/latent recurrence in bottleneck or correction path
- memory only for residual correction, not full state synthesis
- norm-constrained recurrent updates for stability

Why now: repeated “good early, bad late” behavior indicates temporal memory limits.

## Portfolio Guidance

Use a mixed pipeline rather than single-mode exploration:
- 60% lower-risk and medium-risk experiments (areas 1–6) for steady progress
- 40% higher-risk architecture bets (areas 7–10) for breakout gains

Gate progression by two hard criteria:
- competitive rollout error vs incumbent
- reduced numerical artifacts (especially boundary reflection growth)

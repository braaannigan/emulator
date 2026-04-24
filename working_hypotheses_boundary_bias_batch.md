# Boundary Bias Batch

Base setup: current 2-layer shifting-wind incumbent family with 30 epochs, periodic eval every 5 epochs, reference comparison retained, and early stopping disabled.

## B01 Mean-Centered Boundary Delta
Hypothesis: If the reflected artifact is partly a simple mean drift in the predicted edge-band increment, subtracting the per-step boundary-band mean residual will reduce cumulative boundary growth without changing the interior operator.

## B02 Learned Static Boundary Map
Hypothesis: A learned static boundary-band correction map can absorb repeatable edge bias that the incumbent currently reintroduces at every autoregressive step.

## B03 State-Scaled Static Boundary Map
Hypothesis: A static correction pattern whose amplitude is modulated by the current decoded state will handle state-dependent artifact strength better than a fixed correction map.

## B04 Dynamic Decoder Correction Head
Hypothesis: A lightweight correction head on the final decoder features can learn to identify and subtract boundary artifact structure before the step is committed to rollout.

## B05 Edge-State Correction Head
Hypothesis: A correction head driven by the current prognostic state plus explicit boundary features can predict a boundary-local artifact component that is more separable than the main evolution signal.

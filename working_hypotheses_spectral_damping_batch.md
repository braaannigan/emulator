# Spectral Damping Batch

Base setup: current 2-layer shifting-wind incumbent family with 30 epochs, periodic eval every 5 epochs, reference comparison retained, and early stopping disabled.

## D01 Hard Low-Pass Mild
Hypothesis: A mild global low-pass filter applied after each predicted step will suppress unstable high-wavenumber growth without materially damaging interior large-scale dynamics.

## D02 Hard Low-Pass Strong
Hypothesis: A stronger global low-pass filter will show whether the artifact is primarily carried by high-wavenumber content, even at some cost to detail.

## D03 Exponential Damping Mild
Hypothesis: Smooth exponential attenuation of the highest wavenumbers will control rollout instability more gently than a hard cutoff.

## D04 Exponential Damping Moderate
Hypothesis: Increasing exponential damping strength will better suppress autoregressive growth while preserving more low and mid frequencies than hard filtering.

## D05 Exponential Damping Aggressive
Hypothesis: Aggressive high-order exponential damping will reveal whether the remaining artifact is robust even after broad global spectral stabilization.

# Boundary Padding Batch

Base setup: current 2-layer shifting-wind incumbent family with 30 epochs, periodic eval every 5 epochs, reference comparison retained, and early stopping disabled.

## P01 Replicate Padding
Hypothesis: Replacing zero padding with replicate padding will reduce the artificial wall discontinuity and suppress reflected edge artifacts.

## P02 Replicate With 3-Cell Inland Decay
Hypothesis: Replicate padding that decays linearly to zero over three outside-domain grid cells will soften the edge discontinuity more effectively than plain replicate padding.

## P03 Random Inland Padding
Hypothesis: Filling outside-domain cells with random channel-scaled values will break the coherent reflected edge mode if the current artifact is being reinforced by structured padding.

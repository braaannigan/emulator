# Shifted-Stencil Convolution Batch

These hypotheses test a pure convolutional alternative to padding: keep the kernel fully in-domain by shifting the effective stencil center inward near boundaries. For a `3x3` stencil, boundary-adjacent cells can therefore reuse the same interior patch rather than sampling padded exterior values.

## Hypotheses

1. `s01` `shifted-center everywhere`
   Replace all spatial convolutions in the incumbent ConvNeXt-style U-Net with shifted-center convolutions. Hypothesis: removing exterior support everywhere will reduce boundary reflection without sacrificing too much interior fidelity.

2. `s02` `shifted-center finest level`
   Apply shifted-center convolutions only at the finest-resolution encoder/decoder blocks. Hypothesis: the boundary failure is primarily introduced at the highest-resolution spatial operator.

3. `s03` `shifted-center finest two levels`
   Apply shifted-center convolutions at the two finest-resolution scales. Hypothesis: boundary contamination enters across a wider high-resolution band than just the first/last block.

4. `s04` `shifted-center everywhere with milder dilation`
   Apply shifted-center convolutions everywhere, but reduce dilation aggressiveness. Hypothesis: one-sided receptive-field reuse interacts badly with highly dilated kernels, so a calmer dilation cycle may be more stable.

5. `s05` `standard conv block with shifted-center everywhere`
   Replace ConvNeXt blocks with standard two-conv residual blocks, all using shifted-center spatial convolutions. Hypothesis: the shifted-stencil idea may work better in a simpler dense-convolution block than in a depthwise ConvNeXt operator.

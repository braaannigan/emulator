# Boundary Stencil Hypotheses

Base setup: 2-layer shifting-wind incumbent family, 30 epochs, periodic eval every 5 epochs, reference comparison enabled, no early stopping.

Goal: test whether the boundary artifact is caused by the convolution operator itself rather than by missing boundary metadata. These hypotheses all change how convolutions behave near the domain edge so that out-of-domain values are not fabricated by padding.

## s01 Valid Convolution With Cropped U-Net Alignment

Hypothesis: Replacing same-size padded convolutions with valid convolutions throughout the network, while cropping and re-aligning skip connections as needed, will reduce autoregressive boundary reflections because the model never sees fabricated outside-domain values.

Expected signal:
- boundary artifact amplitude decreases near the wall
- interior MSE may worsen if excessive cropping removes too much usable context

## s02 Masked Renormalized Convolution Everywhere

Hypothesis: Applying masked convolutions that ignore out-of-domain stencil taps and renormalize by valid support count will reduce boundary artifact growth while preserving output scale better than valid convolutions.

Expected signal:
- cleaner near-boundary sections than padding-based baselines
- less global bias drift than naive valid-conv shrink/crop

## s03 Partial Convolution Everywhere

Hypothesis: Partial convolutions with an explicit validity mask propagated through the network will outperform simpler masked-renormalized convolutions because the model can learn how much trust to place in boundary-band activations as support decreases.

Expected signal:
- boundary artifact reduction with less degradation of interior structure
- potentially smoother transition from edge band into interior

## s04 Interior-Only Convolution With Separate Boundary Update Rule

Hypothesis: Using normal convolutions only where the full stencil is available and a separate lightweight boundary operator for the edge band will outperform single-rule approaches because the true near-boundary operator differs structurally from the interior one.

Expected signal:
- strongest reduction of edge-driven reflections if the failure is truly operator mismatch
- risk of visible seam where boundary and interior updates meet

## s05 Support-Adaptive Kernel Near Boundaries

Hypothesis: Using support-adaptive convolutions that reduce effective kernel width near boundaries, rather than padding or fully masked renormalization, will reduce artifact injection while preserving more local resolution than strict valid convolutions.

Expected signal:
- smaller edge distortion than padded kernels
- less loss of local detail than full valid-conv cropping

## s06 Boundary-Band Masked Convolution Only In Early And Late Blocks

Hypothesis: Restricting masked or partial convolution to the highest-resolution encoder and decoder blocks, while leaving deeper interior blocks unchanged, will capture most of the boundary benefit at much lower implementation risk and compute cost.

Expected signal:
- meaningful boundary improvement if the artifact is injected mainly at high-resolution edge processing
- weaker effect than full masked-conv models if the failure is distributed through the hierarchy

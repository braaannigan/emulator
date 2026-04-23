# Boundary Information Hypotheses

Base setup: current 2-layer shifting-wind incumbent family with 30 epochs and boundary-aware modifications applied one branch at a time.

## h01 Boundary Mask Input

Hypothesis: Adding a binary domain-edge mask channel will let the model distinguish physical gradients from truncation effects and reduce reflected boundary artifacts.

## h02 Scalar Distance Input

Hypothesis: Adding a single normalized distance-to-boundary channel will let the model learn a graded edge treatment and reduce inward artifact growth more effectively than a hard edge mask alone.

## h03 Anisotropic Distance Input

Hypothesis: Adding separate normalized `distance_to_x_boundary` and `distance_to_y_boundary` channels will let the model represent anisotropic edge effects and improve rollout stability relative to a single scalar boundary distance.

## h04 Boundary Strip Encoder To Bottleneck

Hypothesis: Encoding the boundary strips with a shallow CNN and fusing that embedding into the bottleneck will improve global awareness of edge state and reduce late-rollout boundary reflections.

## h05 Boundary Features At Every Scale

Hypothesis: Injecting boundary features at every encoder scale will preserve edge-awareness through the hierarchy and reduce the growth of artifacts that re-enter the interior during rollout.

## h06 Boundary-Conditioned Skip Gating

Hypothesis: Using boundary-conditioned skip gating will suppress the transfer of edge-contaminated high-resolution features into the decoder and improve rollout stability near boundaries.

## h07 FiLM Boundary Modulation

Hypothesis: FiLM-style modulation from a boundary embedding will let the network adjust interior feature processing based on boundary context and improve long-rollout behaviour without forcing a full separate edge pathway.

## h08 Boundary/Interior Dual Kernels

Hypothesis: Using separate convolution kernels near the boundary and in the interior selected by a learned boundary mask will better match the distinct operator regime at the edges and reduce reflected structure.

## h09 Sponge-Layer Damping Loss

Hypothesis: Adding a thin sponge-layer damping target near the domain boundary will discourage nonphysical edge persistence and reduce inward propagation of reflected signals even if core rollout MSE changes only modestly.

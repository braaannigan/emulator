# Working Hypotheses

## Boundary Handling via Convolution Variants

1. **h01 - Reflect ConvNeXt baseline**
Reflect padding in all convolutions (`boundary_padding_mode: reflect`) will reduce inward boundary reflections versus zero padding in the incumbent ConvNeXt setup.

2. **h02 - Replicate ConvNeXt baseline**
Replicate padding (`boundary_padding_mode: replicate`) will stabilize edge amplitudes better than zero padding while preserving interior skill.

3. **h03 - Circular ConvNeXt baseline**
Circular padding (`boundary_padding_mode: circular`) will reduce hard-edge reflection artifacts if periodic structure dominates boundary behavior.

4. **h04 - Reflect standard conv block**
Switching from ConvNeXt to standard conv blocks under reflect padding will improve boundary smoothness by reducing depthwise-induced edge ringing.

5. **h05 - Reflect depthwise-separable block**
Depthwise-separable blocks with reflect padding will preserve local directional structure and lower rollout boundary noise at similar compute.

6. **h06 - Reflect ConvNeXt with larger kernel**
Using `kernel_size: 7` with reflect padding will improve boundary-context capture and reduce reflected wave build-up.

7. **h07 - Reflect ConvNeXt with smaller kernel**
Using `kernel_size: 3` with reflect padding will reduce oversmoothing and decrease long-horizon edge amplification.

8. **h08 - Reflect ConvNeXt with no dilation cycling**
Setting `dilation_cycle: 1` with reflect padding will reduce gridding/alias-like artifacts near edges caused by aggressive dilation schedules.

9. **h09 - Reflect ConvNeXt with mild dilation cycling**
Setting `dilation_cycle: 2` with reflect padding will balance receptive field growth and boundary stability better than cycle 4.

10. **h10 - Reflect depthwise-separable + larger kernel**
Combining depthwise-separable blocks with `kernel_size: 7` and reflect padding will give the strongest boundary-aware local filtering and reduce reflections most.

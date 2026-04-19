# Working Hypotheses

## Area 1: Rollout-First Model Selection and Checkpointing

1. Selecting checkpoints by best periodic rollout eval (not final epoch) will outperform final-epoch selection on held-out rollout MSE.
2. EMA-smoothed periodic eval model selection will reduce run-to-run incumbent churn versus raw periodic minima.
3. Early-stop on best-to-current degradation will improve median final rollout quality versus threshold-only early-stop.
4. Using both mean rollout MSE and last-step MSE for checkpoint ranking will reduce late-horizon failures.
5. Requiring two consecutive non-degrading periodic evals before accepting a new incumbent will improve stability of leaderboard gains.

## Area 2: Exposure-to-Generated-States Curriculum

6. Reducing scheduled sampling max probability from 0.05 to 0.01 will improve long-horizon autoregressive stability.
7. A delayed increase in teacher-forcing leakage (low early, slightly higher late) will outperform linear scheduled sampling ramps.
8. Adaptive exposure schedules based on periodic eval slope will outperform fixed exposure schedules.
9. Exposure curricula coupled to rollout horizon transitions will reduce post-transition eval spikes.
10. Near-zero teacher forcing with longer warmup epochs will improve final rollout robustness versus default exposure settings.

## Area 3: One-Step Recursive Training Line

11. Recursive one-step training (`output_steps=1`) with horizon-2 rollout will beat multi-step heads on stability at equal compute.
12. Recursive one-step with horizon-3 will improve boundary artifact behavior versus recursive horizon-1.
13. Recursive one-step with 2->4 curriculum will outperform fixed-horizon recursive baselines on final rollout MSE.
14. Recursive one-step plus mild scheduled sampling (0.01-0.02) will improve robustness versus strict self-feeding.
15. Recursive one-step models will show lower best-to-last periodic degradation than multi-output models.

## Area 4: Residual-Bias and Integral Control

16. Enforcing zero-mean predicted residual each step will reduce domain-integrated drift without large MSE penalty.
17. Adding a learned global residual bias-correction head will reduce residual mean bias and improve rollout quality.
18. Penalizing domain-integral residual error will reduce low-frequency drift accumulation over long rollouts.
19. Combining residual-bias correction with recursive one-step training will outperform either method alone.
20. Residual integral constraints applied only after curriculum transition will reduce late-horizon instability.

## Area 5: Boundary-Aware Loss and Operator Design

21. Boundary-weighted loss (higher boundary penalty) will reduce inward-propagating reflection artifacts.
22. Boundary mask channels injected at model input will improve boundary artifact metrics at similar MSE.
23. Boundary-conditioned skip gating will outperform plain additive skip fusion on artifact suppression.
24. Curriculum that emphasizes boundary stabilization early then rebalances globally will improve both artifact and MSE outcomes.
25. Rejecting candidates with worsening boundary artifact score will improve practical rollout quality of accepted incumbents.

## Area 6: Multi-Objective Training Beyond MSE

26. Mixed objective (state + residual) with residual-dominant weighting will improve structural rollout fidelity versus residual-only loss.
27. Adding a spectral consistency term will improve residual field correlation at equal or slightly worse MSE.
28. Adding gradient-consistency loss will reduce high-frequency noise amplification in long rollouts.
29. Dynamic loss reweighting by rollout horizon stage will outperform static loss weights.
30. Objective terms tuned for residual structure will increase acceptance of slightly higher-MSE but artifact-cleaner models.

## Area 7: Transport-Structured Update Blocks

31. A two-stage “advect then correct” block will reduce phase drift compared with direct state mapping.
32. Velocity-conditioned transport branch + correction branch will improve long-horizon coherence versus single-branch U-Net.
33. Predicting tendencies and integrating updates explicitly will reduce rollout compounding error.
34. Transport-first blocks combined with recursive one-step training will outperform transport-first with multi-output heads.
35. Constraining correction magnitude after transport will reduce late-horizon blowups.

## Area 8: Multi-Scale Global-Local Coupling

36. Adding a low-frequency global pathway at bottleneck will improve basin-scale stability.
37. Hybrid spectral+convolution bottleneck will reduce long-range error growth versus conv-only bottleneck.
38. Coarse-grid correction branch fused into decoder will improve late-rollout MSE trend.
39. Global-local fusion with residual bias control will reduce both drift and boundary reflection growth.
40. Multi-scale global coupling will lower best-to-last periodic eval degradation compared with local-only models.

## Area 9: Structured Cross-Variable Coupling

41. Variable-specific encoders with learned coupling will improve joint `h/u/v` consistency versus shared-trunk-only design.
42. Explicit thickness-momentum interaction blocks will reduce physically inconsistent residual updates.
43. Channel-wise coupling gates conditioned on velocity magnitude will improve transport fidelity.
44. Cross-variable coupling constraints applied only in later decoder stages will improve stability versus full-depth coupling.
45. Structured coupling + recursive one-step training will improve long-horizon robustness over either alone.

## Area 10: Temporal Memory Architecture Upgrade

46. Bottleneck ConvGRU memory will reduce late-horizon degradation versus memoryless baselines.
47. Recurrent correction-only memory branch will be more stable than recurrent full-state synthesis.
48. Norm-constrained recurrent updates will reduce rollout volatility without sacrificing early convergence.
49. Short-memory recurrence plus low teacher forcing will outperform long-memory recurrence plus higher teacher forcing.
50. Memory-enhanced models selected by rollout-aware checkpointing will produce more durable incumbents than non-recurrent models selected by final epoch.

# Current Best Results

This file summarizes the current best setup for each registered experiment family in [`experiment_list.md`](/Users/liambrannigan/playModels/emulator/experiment_list.md).

## `double_gyre`

Current best result:
- [`20260321T182400-residual-depthwise-kernel5/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre/emulator/residual_thickness/20260321T182400-residual-depthwise-kernel5/metrics.json)

Best setup:
- Architecture: `residual_thickness`
- Config: [`residual_thickness_depthwise_kernel5.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/residual_thickness_depthwise_kernel5.yaml)
- Source experiment: `20260319T200102`
- `hidden_channels = 16`
- `num_blocks = 4`
- `kernel_size = 5`
- `block_type = depthwise_separable`
- `normalization = none`
- `dilation_cycle = 1`
- `epochs = 50`
- `learning_rate = 1.0e-3`
- `weight_decay = 5.0e-4`

Best metrics:
- `eval_mse_mean = 2.6112`
- `eval_mse_max = 8.1768`
- `train_loss = 0.00182`

Source log:
- [`experiments_double_gyre.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre.md)

## `double_gyre_2x_fine`

Current best result:
- [`skyd-9b938741c1/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_2x_fine/emulator/unet_thickness/skyd-9b938741c1/metrics.json)

Best setup:
- Architecture: `unet_thickness`
- Config: [`unet_thickness_double_gyre_2x_fine_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr2p6e4_wd1p6e5.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/unet_thickness_double_gyre_2x_fine_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr2p6e4_wd1p6e5.yaml)
- Source experiment: `20260401T2LAYER-5000d`
- State fields: `layer_thickness`, `zonal_velocity_centered`, `meridional_velocity_centered`
- `hidden_channels = 24`
- `num_levels = 4`
- `kernel_size = 5`
- `block_type = convnext`
- `stage_depth = 2`
- `dilation_cycle = 4`
- `state_history = 2`
- `output_steps = 2`
- `forcing_mode = none`
- `fusion_mode = input`
- `skip_fusion_mode = add`
- `upsample_mode = bilinear`
- `boundary_padding_mode = zeros`
- `residual_connection = true`
- `learning_rate = 2.6e-4`
- `weight_decay = 1.6e-5`
- Curriculum: rollout steps `2 -> 4`, transition epochs `0, 15`
- `train_start_day = 100`
- `eval_window_days = 250`

Best metrics:
- `eval_mse_mean = 44.5149`
- `eval_mse_max = 124.5174`
- `train_loss = 0.00491`

Source log:
- [`experiments_double_gyre_2x_fine.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre_2x_fine.md)

## `double_gyre_shifting_wind`

Current best result on the standard `250`-day benchmark:
- [`20260403T081920/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind/emulator/unet_thickness/20260403T081920/metrics.json)

Best setup:
- Architecture: `unet_thickness`
- Closest committed base config: [`unet_thickness_shifting_wind_huv_tau_current_window250_deep_spinup100_convnext_multistep2_dilated_lr3p8e4_wd1p8e5_latecurr_residual.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/unet_thickness_shifting_wind_huv_tau_current_window250_deep_spinup100_convnext_multistep2_dilated_lr3p8e4_wd1p8e5_latecurr_residual.yaml)
- Exact winning run differs from that base in two decoder choices recorded in the metrics artifact:
  `skip_fusion_mode = add`
  `upsample_mode = bilinear`
- Source experiment: `20260323T123500-period300d-duration5000d-shift160km`
- State fields: `layer_thickness`, `zonal_velocity_centered`, `meridional_velocity_centered`
- `hidden_channels = 24`
- `num_levels = 4`
- `kernel_size = 5`
- `block_type = convnext`
- `stage_depth = 2`
- `dilation_cycle = 4`
- `state_history = 2`
- `output_steps = 2`
- `forcing_mode = wind_current`
- `fusion_mode = input`
- `boundary_padding_mode = zeros`
- `residual_connection = true`
- `learning_rate = 3.8e-4`
- `weight_decay = 1.8e-5`
- Curriculum: rollout steps `2 -> 4`, transition epochs `0, 15`
- `train_start_day = 100`
- `eval_window_days = 250`

Best metrics:
- `eval_mse_mean = 6.0891`
- `eval_mse_max = 14.3282`
- `train_loss = 0.000499`

Source log:
- [`experiments_double_gyre_shifting_wind.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre_shifting_wind.md)

## `double_gyre_shifting_wind_2layer`

Current best result on the standard `250`-day benchmark:
- [`20260408T072700-dgsw2l-unet-benchmark-5000d-fg/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/20260408T072700-dgsw2l-unet-benchmark-5000d-fg/metrics.json)

Best setup:
- Architecture: `unet_thickness`
- Config: [`unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_benchmark.yaml`](/Users/liambrannigan/playModels/emulator/config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_benchmark.yaml)
- Source experiment: `20260408T071500-period40d-duration5000d-2layer`
- State fields: `layer_thickness`, `zonal_velocity_centered`, `meridional_velocity_centered`
- `hidden_channels = 24`
- `num_levels = 4`
- `kernel_size = 5`
- `block_type = convnext`
- `stage_depth = 2`
- `dilation_cycle = 4`
- `state_history = 2`
- `output_steps = 2`
- `forcing_mode = wind_current`
- `fusion_mode = input`
- `skip_fusion_mode = add`
- `upsample_mode = bilinear`
- `boundary_padding_mode = zeros`
- `residual_connection = true`
- `learning_rate = 3.8e-4`
- `weight_decay = 1.8e-5`
- Curriculum: rollout steps `2 -> 4`, transition epochs `0, 15`
- Periodic eval checkpoints at epochs `5`, `10`, `15`, `20`
- `train_start_day = 100`
- `eval_window_days = 250`

Best metrics:
- `eval_mse_mean = 57.0078`
- `eval_mse_max = 124.2078`
- `train_loss = 0.00721`

Source log:
- [`experiments_double_gyre_shifting_wind_2layer.md`](/Users/liambrannigan/playModels/emulator/experiments_double_gyre_shifting_wind_2layer.md)

## Best U-Net Comparison

| Field | `double_gyre_shifting_wind` best | `double_gyre_2x_fine` 2-layer best |
| --- | --- | --- |
| Metrics artifact | [`20260403T081920/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_shifting_wind/emulator/unet_thickness/20260403T081920/metrics.json) | [`skyd-9b938741c1/metrics.json`](/Users/liambrannigan/playModels/emulator/data/raw/double_gyre_2x_fine/emulator/unet_thickness/skyd-9b938741c1/metrics.json) |
| Source experiment | `20260323T123500-period300d-duration5000d-shift160km` | `20260401T2LAYER-5000d` |
| Architecture | `unet_thickness` | `unet_thickness` |
| State fields | `layer_thickness`, `zonal_velocity_centered`, `meridional_velocity_centered` | `layer_thickness`, `zonal_velocity_centered`, `meridional_velocity_centered` |
| Evaluated channels | `1` | `2` |
| Hidden channels | `24` | `24` |
| Num levels | `4` | `4` |
| Kernel size | `5` | `5` |
| Block type | `convnext` | `convnext` |
| Stage depth | `2` | `2` |
| Dilation cycle | `4` | `4` |
| State history | `2` | `2` |
| Output steps | `2` | `2` |
| Forcing mode | `wind_current` | `none` |
| Fusion mode | `input` | `input` |
| Skip fusion mode | `add` | `add` |
| Upsample mode | `bilinear` | `bilinear` |
| Boundary padding mode | `zeros` | `zeros` |
| Residual connection | `true` | `true` |
| Learning rate | `3.8e-4` | `2.6e-4` |
| Weight decay | `1.8e-5` | `1.6e-5` |
| Curriculum rollout steps | `2 -> 4` | `2 -> 4` |
| Curriculum transition epochs | `0, 15` | `0, 15` |
| Train start day | `100` | `100` |
| Eval window days | `250` | `250` |
| Eval MSE mean | `6.0891` | `44.5149` |
| Eval MSE max | `14.3282` | `124.5174` |
| Train loss | `0.000499` | `0.00491` |

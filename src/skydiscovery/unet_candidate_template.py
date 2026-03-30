from __future__ import annotations

# EVOLVE-BLOCK-START
CONFIG_OVERRIDES = {
    "learning_rate": 1.0e-3,
    "weight_decay": 5.0e-4,
    "hidden_channels": 24,
    "num_levels": 4,
    "kernel_size": 3,
    "block_type": "convnext",
    "stage_depth": 1,
    "norm_type": "groupnorm",
    "batch_size": 2,
    "train_start_day": 100.0,
    "forcing_mode": "wind_current",
    "fusion_mode": "input",
    "skip_fusion_mode": "concat",
    "upsample_mode": "transpose",
    "residual_connection": False,
}
# EVOLVE-BLOCK-END

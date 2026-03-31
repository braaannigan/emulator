from __future__ import annotations

# EVOLVE-BLOCK-START
CONFIG_OVERRIDES = {
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-4,
    "hidden_channels": 24,
    "num_levels": 4,
    "kernel_size": 5,
    "block_type": "convnext",
    "stage_depth": 2,
    "norm_type": "groupnorm",
    "skip_fusion_mode": "concat",
    "upsample_mode": "transpose",
}
# EVOLVE-BLOCK-END

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_PREFIX="${1:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-objcurr}"

declare -a CONFIGS=(
  "a_out1_h1_fixed_30ep"
  "b_out2_h1_fixed_30ep"
  "c_out1_h2_fixed_30ep"
  "d_out2_h2_fixed_30ep"
  "e_out1_h2to4_30ep"
  "f_out2_h2to4_30ep"
)

for suffix in "${CONFIGS[@]}"; do
  config_path="config/emulator/unet_thickness_shifting_wind_2layer_objcurr_ablation_${suffix}.yaml"
  experiment_id="${RUN_PREFIX}-${suffix}"
  echo "Running ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
done

echo "Completed output_steps x curriculum ablation batch: ${RUN_PREFIX}"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_PREFIX="${1:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-batchsize-followup40}"

declare -a CONFIGS=(
  "a_b1_lr2p7e4"
  "b_b2_baseline"
  "c_b3_lr4p6e4"
  "d_b4_lr5p4e4_wd2p7e5"
  "e_b4_lr3p8e4_wd3p6e5"
)

for suffix in "${CONFIGS[@]}"; do
  config_path="config/emulator/unet_thickness_shifting_wind_2layer_batchsize_followup_40ep_${suffix}.yaml"
  experiment_id="${RUN_PREFIX}-${suffix}"
  echo "Running ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
done

echo "Completed batch-size follow-up batch: ${RUN_PREFIX}"

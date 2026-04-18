#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_PREFIX="${1:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-curriculum-followup40}"

declare -a CONFIGS=(
  "a_h2_fixed"
  "b_h2to4_t20"
  "c_h2to4_t30"
  "d_h2to3to4_t12_t24"
  "e_h2to3_fixed"
)

for suffix in "${CONFIGS[@]}"; do
  config_path="config/emulator/unet_thickness_shifting_wind_2layer_curriculum_followup_40ep_${suffix}.yaml"
  experiment_id="${RUN_PREFIX}-${suffix}"
  echo "Running ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
done

echo "Completed curriculum follow-up batch: ${RUN_PREFIX}"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_ID="${1:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-incumbent-objB1-100ep-notransition}"
CONFIG_PATH="config/emulator/unet_thickness_shifting_wind_2layer_incumbent_objB1_residual_ss005_100ep_notransition_manual.yaml"

echo "Running incumbent 100-epoch no-transition experiment"
echo "config: ${CONFIG_PATH}"
echo "experiment_id: ${EXPERIMENT_ID}"

.venv/bin/python src/pipelines/train_unet_thickness.py \
  --config "${CONFIG_PATH}" \
  --experiment-id "${EXPERIMENT_ID}"

echo ""
echo "Training history:"
echo "  data/interim/emulator/unet_thickness/${EXPERIMENT_ID}/training_history.json"
echo "Metrics:"
echo "  data/raw/double_gyre_shifting_wind_2layer/emulator/unet_thickness/${EXPERIMENT_ID}/metrics.json"

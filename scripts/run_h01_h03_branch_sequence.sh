#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_PREFIX="${1:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-h01-h03}"
BASE_BRANCH="${2:-main}"
TEST_CMD="${TEST_CMD:-.venv/bin/python -m pytest -q test/unit/test_unet_thickness_config.py test/unit/test_unet_thickness_training.py test/unit/test_unet_thickness_pipeline.py}"

IDS=(
  "h01"
  "h02"
  "h03"
)
BRANCHES=(
  "2026-04-19-dgsw2l-unet-h01-rollout-checkpoint-selection"
  "2026-04-19-dgsw2l-unet-h02-ema-checkpoint-selection"
  "2026-04-19-dgsw2l-unet-h03-degradation-early-stop"
)
CONFIGS=(
  "config/emulator/hypotheses/unet_thickness_dgsw2l_h01.yaml"
  "config/emulator/hypotheses/unet_thickness_dgsw2l_h02.yaml"
  "config/emulator/hypotheses/unet_thickness_dgsw2l_h03.yaml"
)

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree must be clean before sequence run." >&2
  exit 1
fi

ORIGINAL_BRANCH="$(git branch --show-current)"
restore_branch() {
  git switch "${ORIGINAL_BRANCH}" >/dev/null 2>&1 || true
}
trap restore_branch EXIT

run_one() {
  local id="$1"
  local branch="$2"
  local config_path="$3"
  local experiment_id="${RUN_PREFIX}-${id}"

  echo "==== ${id} ${branch} ===="
  git switch "${branch}"

  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Branch ${branch} has uncommitted changes; aborting." >&2
    exit 1
  fi

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${id}: ${config_path}" >&2
    exit 1
  fi

  echo "test: ${TEST_CMD}"
  eval "${TEST_CMD}"

  echo "train: ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
}

for i in "${!IDS[@]}"; do
  run_one "${IDS[$i]}" "${BRANCHES[$i]}" "${CONFIGS[$i]}"
done

git switch "${BASE_BRANCH}"
trap - EXIT
echo "Completed h01-h03 branch sequence."

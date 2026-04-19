#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${1:-hypothesis_manifest.csv}"
RUN_PREFIX="${2:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-hypotheses}"
BASE_BRANCH="${3:-main}"
TEST_CMD="${TEST_CMD:-.venv/bin/python -m pytest -q}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree must be clean before sequence run." >&2
  exit 1
fi

run_one() {
  local id="$1"
  local branch="$2"
  local config_path="$3"
  local experiment_id="${RUN_PREFIX}-${id}"

  echo "==== ${id} ${branch} ===="
  git switch "${branch}"

  # Enforce "working and tested code before moving to next".
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Branch ${branch} has uncommitted changes; aborting." >&2
    exit 1
  fi

  echo "test: ${TEST_CMD}"
  eval "${TEST_CMD}"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${id}: ${config_path}" >&2
    echo "Implement hypothesis code/config on ${branch} before sequence can continue." >&2
    exit 1
  fi

  echo "train: ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
}

# Skip header
tail -n +2 "${MANIFEST}" | while IFS=, read -r id area branch config_path hypothesis; do
  run_one "${id}" "${branch}" "${config_path}"
done

git switch "${BASE_BRANCH}"
echo "Completed branch sequence for manifest ${MANIFEST}."

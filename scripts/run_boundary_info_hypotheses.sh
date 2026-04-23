#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${1:-boundary_info_hypothesis_manifest.csv}"
RUN_PREFIX="${2:-$(date -u +%Y%m%dT%H%M%S)-dgsw2l-binfo}"
RETURN_BRANCH="${3:-2026-04-23-dgsw2l-boundaryinfo-setup}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

status_output="$(git status --porcelain | grep -vE '^\?\? figs(/|$)' || true)"
if [[ -n "${status_output}" ]]; then
  echo "Working tree must be clean before running branch sequence." >&2
  printf '%s\n' "${status_output}" >&2
  exit 1
fi

while IFS=, read -r id area branch config_path hypothesis; do
  if [[ "${id}" == "id" ]]; then
    continue
  fi

  echo "==== ${id} (${branch}) ===="
  git switch "${branch}"

  if [[ ! -f "${config_path}" ]]; then
    echo "Missing config for ${id}: ${config_path}" >&2
    exit 1
  fi

  experiment_id="${RUN_PREFIX}-${id}"
  echo "train: ${experiment_id}"
  .venv/bin/python src/pipelines/train_unet_thickness.py \
    --config "${config_path}" \
    --experiment-id "${experiment_id}"
done < "${MANIFEST}"

git switch "${RETURN_BRANCH}"
echo "Completed boundary-information branch sequence."

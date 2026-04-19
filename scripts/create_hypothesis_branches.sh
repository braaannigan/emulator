#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="${1:-hypothesis_manifest.csv}"
BASE_BRANCH="${2:-main}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree must be clean before branch creation." >&2
  exit 1
fi

git switch "${BASE_BRANCH}"

# Skip header
tail -n +2 "${MANIFEST}" | while IFS=, read -r id area branch config_path hypothesis; do
  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "exists: ${branch}"
    continue
  fi

  echo "create: ${branch}"
  git switch -c "${branch}" "${BASE_BRANCH}"
  mkdir -p hypotheses
  cat > "hypotheses/${id}.md" <<EOF
# ${id} (${area})

Branch: \`${branch}\`  
Config Path: \`${config_path}\`

## Hypothesis

${hypothesis}

## Implementation Checklist

- [ ] Implement minimal, narrow code/config change for this hypothesis.
- [ ] Add/adjust tests for changed behavior.
- [ ] Run \`.venv/bin/python -m pytest -q\`.
- [ ] Ensure config exists at \`${config_path}\`.
- [ ] Update matching experiment log entry.
EOF

  git add "hypotheses/${id}.md"
  git commit -m "Scaffold ${id} hypothesis branch metadata"
done

git switch "${BASE_BRANCH}"
echo "Branch scaffolding complete."

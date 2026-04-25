#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <rollout.nc> [output.mp4] [fps]" >&2
  exit 1
fi

rollout_path="$1"
output_path="${2:-${rollout_path%.nc}.mp4}"
fps="${3:-8}"

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD" .venv/bin/python - "$rollout_path" "$output_path" "$fps" <<'PY'
from pathlib import Path
import sys

from src.models.cnn_thickness.animation import create_rollout_comparison_animation

rollout_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
fps = int(sys.argv[3])

create_rollout_comparison_animation(rollout_path, output_path, fps=fps)
print(output_path)
PY

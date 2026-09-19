#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export ISAACLAB_PATH="$PROJECT_DIR/.deps/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
VIEWER_PREFIX=()
if [[ $# == 0 ]]; then
  if command -v vglrun >/dev/null; then VIEWER_PREFIX=(vglrun); fi
  export DISPLAY="${DISPLAY:-:20}"
  set -- scripts/rsl_rl/play.py --task Template-G1-Datn-v0 --num_envs 4 --visualizer newton --real-time --checkpoint logs/rsl_rl/g1_datn/2026-09-17_03-30-23/best_walk.pt
fi
exec "${VIEWER_PREFIX[@]}" "$PROJECT_DIR/.venv/bin/python" "$@"

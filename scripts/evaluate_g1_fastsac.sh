#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DISPLAY=${DISPLAY:-:20}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL=glfw
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
unset LD_PRELOAD
exec uv run --no-project --python .venv-fastsac/bin/python python scripts/fast_sac/evaluate_holosoma.py "$@"

#!/usr/bin/env bash
# Run G1-MultiSkill using the isolated Python environment without touching run.sh.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv-multiskill"
PY="$VENV/bin/python"
LAB="$ROOT/.deps/IsaacLab-multiskill"
[[ -x "$PY" ]] || { echo "Missing $PY. Run bash setup_multiskill.sh first." >&2; exit 1; }
[[ -d "$LAB/source" ]] || { echo "Missing $LAB. Run bash setup_multiskill.sh first." >&2; exit 1; }

# Exact working directory of original Humaniod scripts; imports g1_datn.tasks.
cd "$ROOT"
export ISAACLAB_PATH="$LAB"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "${1:-}" in
  --preflight)
    exec "$PY" - <<'PY'
import g1_datn
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
name = 'G1-MultiSkill-v0'
print('Task:', gym.spec(name).id)
scene = load_cfg_from_registry(name, 'env_cfg_entry_point')
agent = load_cfg_from_registry(name, 'rsl_rl_cfg_entry_point')
print('Environment:', type(scene).__name__)
print('Robot prim path:', scene.scene.robot.prim_path)
print('PPO:', type(agent).__name__)
print('Default num_envs:', scene.scene.num_envs)
print('Default max_iterations:', agent.max_iterations)
print('[PASS] Config preflight. This is not a GPU simulation test.')
PY
    ;;
  --smoke)
    shift
    set -- scripts/rsl_rl/train.py --task G1-MultiSkill-v0 --num_envs 1 --max_iterations 2 --visualizer none "$@"
    ;;
  --smoke64)
    shift
    set -- scripts/rsl_rl/train.py --task G1-MultiSkill-v0 --num_envs 64 --max_iterations 2 --visualizer none "$@"
    ;;
  '')
    echo "Usage: ./run_multiskill.sh --preflight|--smoke|--smoke64|<project-script.py> [args...]" >&2
    exit 2
    ;;
esac

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != YES ]]; then
  echo 'NVIDIA EULA not pre-accepted. Review and accept terms, then set OMNI_KIT_ACCEPT_EULA=YES.' >&2
  exit 2
fi

# Running an explicit project script also works, but forbid pointing to project setup scripts.
exec "$PY" "$@"

#!/usr/bin/env bash
# Recreate the tested Linux x86_64 environment using uv.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != --desktop ]]; then
  echo "Usage: $0 [--desktop]" >&2; exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This setup requires Linux x86_64 with an NVIDIA RTX GPU.' >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/humanoid-uv-install.sh
  sh /tmp/humanoid-uv-install.sh
  export PATH="$HOME/.local/bin:$PATH"
fi
LAB_REV=28a37cecdd433c22d9eabd6a5954add9f13a8951
LAB_DIR="$PROJECT_DIR/.deps/IsaacLab"
if [[ ! -d "$LAB_DIR/.git" ]]; then
  mkdir -p "$PROJECT_DIR/.deps"
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git "$LAB_DIR"
fi
if [[ "$(git -C "$LAB_DIR" rev-parse HEAD)" != "$LAB_REV" ]]; then
  echo "Expected Isaac Lab revision $LAB_REV in $LAB_DIR" >&2
  exit 1
fi
uv sync --frozen --python 3.12
./run.sh scripts/zero_agent.py --task Template-G1-Datn-v0 --num_envs 4 --visualizer none --max_steps 100
printf '\nSetup and GPU simulation check succeeded. Run ./run.sh to view the saved walking policy.\n'

if [[ "$MODE" == --desktop ]]; then
  ./start-desktop.sh
fi

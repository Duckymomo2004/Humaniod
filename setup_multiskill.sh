#!/usr/bin/env bash
# Install G1-MultiSkill into isolated environment; do not modify Humaniod/FastSAC environments.
# Usage: bash setup_multiskill.sh [--preflight]
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB="$ROOT/.deps/IsaacLab-multiskill"
UNITREE="$ROOT/.deps/unitree_rl_lab-multiskill"
VENV="$ROOT/.venv-multiskill"
PY="$VENV/bin/python"
PATCH="$ROOT/patches/unitree_isaaclab3.patch"
REVFILE="$ROOT/patches/unitree_commit.txt"
LAB_TAG="v3.0.0-beta2.patch1"
SIM_VERSION="6.0.1.0"

log() { printf '\n[multiskill] %s\n' "$*"; }
die() { printf '[multiskill] ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "[multiskill] Failed at line %s. Other environments were not intentionally modified.\n" "$LINENO" >&2' ERR

MODE="${1:-}"
[[ $# -le 1 && ( -z "$MODE" || "$MODE" == "--preflight" ) ]] || die 'Usage: bash setup_multiskill.sh [--preflight]'
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || die 'Requires Linux x86_64 (Ubuntu 22.04/24.04 recommended).'
for cmd in git bash; do command -v "$cmd" >/dev/null || die "Missing command: $cmd"; done
[[ -s "$PATCH" ]] || die "Missing patch: $PATCH. Commit the file to branch g1-multiskill first."
[[ -s "$REVFILE" ]] || die "Missing commit file: $REVFILE"
[[ -f "$ROOT/source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/__init__.py" ]] || die 'Missing g1_multiskill task: check the branch and gitignore rules.'
[[ -f "$ROOT/source/g1_datn/setup.py" ]] || die 'Missing editable local package source/g1_datn/setup.py'
[[ -f "$ROOT/run_multiskill.sh" ]] || die 'Missing run_multiskill.sh in project root.'
UNITREE_REV="$(tr -d '\r\n ' < "$REVFILE")"
[[ "$UNITREE_REV" =~ ^[0-9a-fA-F]{7,40}$ ]] || die 'Unitree revision file does not contain a valid Git SHA.'

log "Project: $ROOT"
log "Isaac Lab: $LAB_TAG; Isaac Sim: $SIM_VERSION; Unitree: $UNITREE_REV"
log "Isolated paths: $VENV, $LAB, $UNITREE"
if [[ -n "$MODE" ]]; then
  log 'Preflight-only mode: files present and paths checked; no packages installed.'
  exit 0
fi

command -v nvidia-smi >/dev/null || die 'nvidia-smi missing. Choose a VM with an NVIDIA GPU and working driver.'
nvidia-smi || die 'GPU/driver is not usable. Fix the VM GPU/driver before installation.'
command -v ldd >/dev/null || die 'ldd missing; pip-installed Isaac Sim requires GLIBC 2.35+.'
"$(command -v bash)" -c 'v=$(ldd --version | head -n 1 | awk "{print \$NF}"); test "$(printf "%s\n2.35\n" "$v" | sort -V | head -n1)" = "2.35"' || die 'Require GLIBC >= 2.35.'
command -v cmake >/dev/null || log 'WARNING: cmake missing; install cmake and build-essential (sudo apt install -y cmake build-essential).'
command -v uv >/dev/null || die 'uv not installed. Install from https://docs.astral.sh/uv/getting-started/installation/'

# Clear any previously activated environment and prevent reading Humaniod uv config.
unset CONDA_PREFIX || true
unset UV_PROJECT_ENVIRONMENT || true
unset UV_PYTHON || true
unset VIRTUAL_ENV || true
export UV_NO_CONFIG=1
# These settings apply only to this script and its child installation commands.
export UV_PRERELEASE=allow
export UV_INDEX_STRATEGY=unsafe-best-match

mkdir -p "$ROOT/.deps"

if [[ ! -e "$LAB" ]]; then
  log "Cloning Isaac Lab ($LAB_TAG)..."
  git clone --depth 1 --branch "$LAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$LAB"
fi
[[ -d "$LAB/.git" ]] || die "Isaac Lab target directory exists but is not a git checkout: $LAB"
LAB_HEAD="$(git -C "$LAB" rev-parse HEAD)"
LAB_TAG_SHA="$(git -C "$LAB" rev-parse "refs/tags/$LAB_TAG^{commit}")" || die 'Local checkout has no expected tag.'
[[ "$LAB_HEAD" == "$LAB_TAG_SHA" ]] || die "Isaac Lab checkout at $LAB_HEAD, expected tag $LAB_TAG; refusing to modify it."

if [[ ! -e "$UNITREE" ]]; then
  log 'Cloning upstream Unitree RL Lab...'
  git clone https://github.com/unitreerobotics/unitree_rl_lab.git "$UNITREE"
  git -C "$UNITREE" checkout --detach "$UNITREE_REV"
fi
[[ -d "$UNITREE/.git" ]] || die "Unitree target directory is not a Git checkout: $UNITREE"
UNITREE_HEAD="$(git -C "$UNITREE" rev-parse HEAD)"
UNITREE_EXPECT="$(git -C "$UNITREE" rev-parse "$UNITREE_REV^{commit}")" || die 'Recorded Unitree commit is not available locally.'
[[ "$UNITREE_HEAD" == "$UNITREE_EXPECT" ]] || die "Unitree checkout at $UNITREE_HEAD, expected $UNITREE_EXPECT; refusing to modify it."

# Re-running setup must not duplicate the patch or overwrite other local changes.
if git -C "$UNITREE" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  log 'Unitree compatibility patch already applied; skipping.'
else
  git -C "$UNITREE" diff --quiet && git -C "$UNITREE" diff --cached --quiet || die 'Unitree checkout contains unrelated changes; do not overwrite them.'
  git -C "$UNITREE" apply --check "$PATCH" || die 'Patch does not apply cleanly to the recorded Unitree commit.'
  git -C "$UNITREE" apply --whitespace=nowarn "$PATCH"
  log 'Applied Unitree compatibility patch.'
fi

if [[ ! -x "$PY" ]]; then
  [[ ! -e "$VENV" ]] || die "Target venv exists but lacks Python: $VENV. Inspect it manually before retrying."
  log 'Creating isolated Python 3.12 environment...'
  uv --no-config python install 3.12
  uv --no-config venv --python 3.12 --seed "$VENV"
fi
[[ "$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == '3.12' ]] || die 'Expected Python 3.12 in the isolated environment.'
source "$VENV/bin/activate"
export UV_PYTHON="$PY"

log "Installing Isaac Sim $SIM_VERSION (large download)..."
uv --no-config pip install --python "$PY" \
  "isaacsim[all,extscache]==$SIM_VERSION" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow

log 'Installing CUDA PyTorch version for Isaac Lab 3.0 beta 2 patch 1...'
uv --no-config pip install --python "$PY" \
  'torch==2.10.0' 'torchvision==0.25.0' \
  --index-url https://download.pytorch.org/whl/cu128

log 'Installing Isaac Lab core + RSL-RL dependencies (in this venv only)...'
(
  cd "$LAB"
  ./isaaclab.sh -i 'rl[rsl-rl]'
)

log 'Installing Unitree and local g1_datn in editable mode...'
uv --no-config pip install --python "$PY" \
  -e "$UNITREE/source/unitree_rl_lab" \
  -e "$ROOT/source/g1_datn"

log 'Checking pinned key versions...'
"$PY" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
expected = {'isaacsim': '6.0.1.0', 'rsl-rl-lib': '5.0.1'}
for pkg, want in expected.items():
    actual = version(pkg)
    print(f'{pkg}={actual} (expected {want})')
    if actual != want:
        raise RuntimeError(f'{pkg} version mismatch: {actual} != {want}')
import torch
print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable to PyTorch. Inspect GPU driver / CUDA wheel before training.')
PY

log 'Running offline task/config registration checks...'
"$ROOT/run_multiskill.sh" --preflight
log 'Setup completed. Training NOT started.'
log 'After reviewing NVIDIA EULA, export OMNI_KIT_ACCEPT_EULA=YES, then run:'
log '  ./run_multiskill.sh --smoke       # 1 env, 2 iterations'
log '  ./run_multiskill.sh --smoke64     # 64 envs, 2 iterations'

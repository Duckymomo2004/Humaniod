#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=d18d6cc50f872c15e904a22ceac22313cec955c8
if [[ ! -d .deps/holosoma/.git ]]; then
  git clone https://github.com/amazon-far/holosoma.git .deps/holosoma
  git -C .deps/holosoma checkout "$revision"
fi
if [[ "$(git -C .deps/holosoma rev-parse HEAD)" != "$revision" ]]; then
  echo "Holosoma revision differs from the validated version: $revision" >&2
  exit 1
fi
patch_file="$PWD/scripts/fast_sac/holosoma_warp_compat.patch"
if git -C .deps/holosoma apply --check "$patch_file"; then
  git -C .deps/holosoma apply "$patch_file"
elif ! git -C .deps/holosoma apply --reverse --check "$patch_file"; then
  echo "Cannot apply the compatibility patch; inspect local Holosoma changes." >&2
  exit 1
fi
if [[ ! -d .venv-fastsac ]]; then
  uv --no-config venv --python 3.11 .venv-fastsac
fi
if [[ ! -d .deps/IsaacLab-fastsac/.git ]]; then
  git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab-fastsac
fi
# --no-config isolates this Python 3.11 stack from the project's IsaacSim 6 /
# Python 3.12 dependency overrides.
uv --no-config pip install --python .venv-fastsac/bin/python \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv --no-config pip install --python .venv-fastsac/bin/python \
  'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
uv --no-config pip install --python .venv-fastsac/bin/python \
  --overrides scripts/fast_sac/constraints.txt --build-constraint scripts/fast_sac/constraints.txt \
  -e .deps/holosoma/src/holosoma -e .deps/IsaacLab-fastsac/source/isaaclab \
  -e .deps/IsaacLab-fastsac/source/isaaclab_assets \
  -e .deps/IsaacLab-fastsac/source/isaaclab_tasks \
  -e .deps/IsaacLab-fastsac/source/isaaclab_rl h5py mujoco==3.8.0

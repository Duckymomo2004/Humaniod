#!/usr/bin/env bash
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh"
. "$utils/environment.sh"
export DISPLAY=:20
cd "$PROJECT_DIR"
# Use the OpenGL viewer with VirtualGL on this Xvfb desktop.
# Physics remains Isaac Sim PhysX on CUDA.
pty ./run.sh 2>&1

#!/usr/bin/env bash
# Manage the visible demo with the Vast instance's existing supervisor.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_USER="$(id -un)"
if [[ "$SIM_USER" == root ]]; then SIM_USER=user; fi
SIM_HOME="$(getent passwd "$SIM_USER" | cut -d: -f6)"
mkdir -p "$PROJECT_DIR/run"
cat > "$PROJECT_DIR/run/humanoid.conf" <<EOF
[program:humanoid]
command=$PROJECT_DIR/run-desktop.sh
directory=$PROJECT_DIR
user=$SIM_USER
environment=PROC_NAME="humanoid",HOME="$SIM_HOME",DISPLAY=":20",XDG_RUNTIME_DIR="/run/user/$(id -u "$SIM_USER")"
autostart=true
autorestart=unexpected
startsecs=10
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
EOF
sudo install -m 644 "$PROJECT_DIR/run/humanoid.conf" /etc/supervisor/conf.d/humanoid.conf
sudo supervisorctl reread
sudo supervisorctl update
state="$(sudo supervisorctl status humanoid | awk '{print $2}' || true)"
if [[ "$state" != RUNNING && "$state" != STARTING ]]; then
  sudo supervisorctl start humanoid
fi
for attempt in {1..30}; do
  state="$(sudo supervisorctl status humanoid | awk '{print $2}' || true)"
  if [[ "$state" == RUNNING ]]; then
    sudo supervisorctl status humanoid
    exit 0
  fi
  if [[ "$state" != STARTING ]]; then
    echo "Demo failed to start: $state. See /var/log/portal/humanoid.log" >&2
    exit 1
  fi
  sleep 1
done
echo 'Demo is still starting; check sudo supervisorctl status humanoid.' >&2
exit 1

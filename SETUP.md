# G1 simulation: work completed and instructions

Bản tiếng Việt, bao gồm hướng dẫn đổi checkpoint và giải thích Isaac Sim/Newton: [HUONG_DAN_VI.md](HUONG_DAN_VI.md).

## What was done

The project is located at `/workspace/Humaniod` (this is the actual folder spelling).
It defines the Isaac Lab task `Template-G1-Datn-v0`, based on the Unitree G1 flat-ground
locomotion environment. The completed work made this existing task and its saved
walking policy runnable on this Vast Linux desktop instance.

1. **Inspected the project and machine.** Identified the RTX 5060 Ti with 16 GB
   VRAM, the existing desktop on display `:20`, and the saved training checkpoints.
   The original dependency manifest selected Windows and referenced an Isaac Lab
   checkout that was absent from this machine.
2. **Repaired the dependency configuration.** Updated `pyproject.toml` for Linux
   x86_64, changed Isaac Lab source paths to `.deps/IsaacLab`, and added the local
   `g1_datn` package and Isaac Sim. Removed unused template extras and resolved
   conflicting upstream package constraints with explicit uv overrides.
3. **Installed with uv.** Created `.venv`, installed the project in editable mode,
   and generated the Linux `uv.lock`. The main components are Isaac Lab
   `v3.0.0-beta2`, Isaac Sim `6.0.0.1`, PyTorch `2.10.0+cu128`, and RSL-RL `5.0.1`.
   CUDA 12.8 wheels support this GPU. The host NVIDIA driver was kept in place.
4. **Added repeatable setup and launch scripts.** `setup.sh` recreates the
   environment from the lockfile and runs an actual GPU simulation check.
   `run.sh` supplies the runtime environment and defaults to four robots using
   the existing `best_walk.pt` checkpoint.
5. **Made verification finite and observable.** Added `--max_steps` to
   `scripts/zero_agent.py` and `scripts/rsl_rl/play.py`, numerical checks for
   rewards or policy actions, and success messages. Playback also checks for a
   closed viewer window and reports progress every 500 steps.
6. **Fixed the desktop viewing path.** The Kit Vulkan window appeared blank on
   this Xvfb desktop. The working configuration uses the Newton OpenGL viewer
   through VirtualGL. The physics backend remains Isaac Sim PhysX on CUDA.
7. **Created a managed demo.** `start-desktop.sh` installs the supervisor service
   named `humanoid`; `run-desktop.sh` connects it to the instance logging system.
   The process runs as the desktop user and restarts after an unexpected exit.
8. **Verified the result.** Ran 100 physics steps with four robots, 500 headless
   steps using the saved policy, and more than 5,000 steps in the visible demo.
   Inspected a screenshot showing the robots and successfully reran the one-shot
   setup. These are completed verification results, not a guarantee that the
   service is still running at the time you read this file; use the status
   command below to check.

The walking policy was already supplied with the project. This work installed,
configured, and verified its playback; it did not train a new policy. Full
training and policy quality evaluation were not performed.

## One-shot setup

From the existing project checkout on this Vast Linux desktop instance:

```bash
cd /workspace/Humaniod
bash setup.sh --desktop
```

This installs dependencies with **uv**, creates `.venv`, fetches Isaac Lab at
commit `28a37cecdd433c22d9eabd6a5954add9f13a8951` (v3.0.0-beta2), uses the exact
packages in `uv.lock`, checks 100 physics steps with four G1 robots on CUDA, and
starts the saved walking policy on display `:20` under supervisor. Re-running
it reuses downloaded packages and an already running demo.

Open **Selkies Low Latency Desktop** from the Vast portal to see the
OpenGL viewer (window title: **Newton Viewer**). Physics still runs in Isaac Sim
PhysX on CUDA; only the viewer uses Newton/OpenGL because the Kit Vulkan window
does not present correctly on this Xvfb desktop. The existing desktop authentication applies. No new public port is used.
The desktop launch requires this image's supervisor, sudo, and desktop services.

For installation and the GPU check without starting the desktop service:

```bash
bash setup.sh
```

The installation targets Linux x86_64, Python 3.12, Isaac Sim 6.0.0.1,
PyTorch 2.10.0 with CUDA 12.8, and RSL-RL 5.0.1. It was configured for this
RTX 5060 Ti (16 GB). Internet access is needed for packages and NVIDIA robot
assets on the first run. Allow approximately 60 GB of free disk space for
packages, caches, and the environment. The launcher sets
`OMNI_KIT_ACCEPT_EULA=YES` for NVIDIA Isaac Sim's noninteractive startup.

## Run and control

```bash
# Status, logs, stop, and start the managed demo:
sudo supervisorctl status humanoid
tail -f /var/log/portal/humanoid.log
sudo supervisorctl stop humanoid
sudo supervisorctl start humanoid

# After stopping the service, run interactively in the desktop terminal:
./run.sh

# Verify the saved policy for 500 steps without a window:
./run.sh scripts/rsl_rl/play.py --task Template-G1-Datn-v0 \
  --num_envs 4 --visualizer none --max_steps 500 \
  --checkpoint logs/rsl_rl/g1_datn/2026-09-17_03-30-23/best_walk.pt

# Train this task (choose a number of environments that fits your GPU):
./run.sh scripts/rsl_rl/train.py --task Template-G1-Datn-v0 \
  --num_envs 64 --visualizer none
```

The demo uses the supplied `best_walk.pt`, with the project's existing commands,
rewards, and randomizations. It is a policy demonstration; robots can fall and
reset. `--max_steps 0` runs playback until the window closes or Ctrl+C is pressed.

## What each file does

| File | Purpose |
| --- | --- |
| `setup.sh` | Fetch the pinned Isaac Lab checkout, install with `uv sync --frozen`, verify GPU physics, optionally start the desktop demo. |
| `pyproject.toml` | Linux dependencies, editable source paths, package indexes, and compatibility overrides. |
| `uv.lock` | Exact resolved package versions used to reproduce the environment. |
| `run.sh` | Select the project Python and runtime variables; launch the saved policy when called without arguments. |
| `start-desktop.sh` | Create/update `/etc/supervisor/conf.d/humanoid.conf` and start the managed demo. |
| `run-desktop.sh` | Launch the demo on display `:20` using the instance's logging wrapper. |
| `scripts/zero_agent.py` | Run the environment with zero actions; default verification length is 100 steps. |
| `scripts/rsl_rl/play.py` | Load the saved policy and simulate it, with optional finite playback. |
| `.gitignore` | Exclude dependency checkouts, virtual environments, runtime output, exported policies, and viewer UI state. |
| `SETUP.md` | This work record and operating guide. |

## Troubleshooting

- **No visible simulation:** open the instance's Selkies desktop and look for
  **Newton Viewer**. Check `sudo supervisorctl status humanoid` and
  `/var/log/portal/humanoid.log`. Initial asset loading and viewer startup can
  take longer than supervisor's process-start check.
- **An empty Kit window:** use the supplied default launcher, which selects
  `--visualizer newton`. That is the viewer verified on this virtual desktop.
- **Need to restart after changing code or runtime settings:** run
  `sudo supervisorctl restart humanoid`. Re-running setup reuses an already
  running service; it does not necessarily restart it.
- **Running from another directory:** call the setup script using its full
  path, such as `bash /workspace/Humaniod/setup.sh --desktop`.
- **Unexpected Isaac Lab revision:** setup deliberately stops if `.deps/IsaacLab`
  points to a different commit. Preserve any changes in that checkout before
  replacing it with the expected revision.
- **Recreating on another machine:** preserve the files listed below. The
  `--desktop` option assumes this Vast desktop image's supervisor helpers and
  display `:20`; it does not provision a desktop on a generic Linux server.

## Files to preserve

Keep the project source, checkpoints, `pyproject.toml`, `uv.lock`, `setup.sh`,
`run.sh`, `start-desktop.sh`, and `run-desktop.sh` together. The scripts locate the
project relative to themselves, so its directory can move. `.venv` and `.deps`
can be recreated by the setup. `setup.sh --desktop` creates the system supervisor
config `/etc/supervisor/conf.d/humanoid.conf` with the current project path.

This instance's `/workspace` is **not backed by a persistent host volume**.
Copy the project off the instance before destroying or recycling it.

## Dependency choices

The original manifest referenced an absent Windows Isaac Lab checkout. The
Linux manifest uses a pinned local source checkout. Explicit uv overrides
reconcile upstream Isaac Sim / Isaac Lab pins for PyTorch, Newton, coverage,
packaging, and NumPy. The resulting environment is frozen in `uv.lock` and
verified by running physics and checkpoint inference, rather than import checks
alone. Avoid running `uv lock --upgrade` unless you intend to retest those pins.

Upstream release installation reference:
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/pip_installation.html

## Verified on this instance

- Four G1 robots: 100 PhysX steps on `cuda:0`, finite rewards.
- Supplied walking checkpoint: 500 headless steps, finite actions.
- Managed OpenGL demo: over 5,000 steps and visually inspected moving robots.
- Screenshot: `run/verified-desktop.png`; verification logs: `run/setup.log` and
  `run/play-check.log`.

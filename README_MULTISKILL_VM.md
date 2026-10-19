# G1-MultiSkill installation on Vast/Linux VM

This is a **separate** install for G1-MultiSkill, not a replacement of the repository's existing Humaniod/FastSAC stack.

## Pinned versions
- Linux x86_64, GLIBC >= 2.35, NVIDIA GPU + suitable driver
- Python 3.12
- Isaac Lab `v3.0.0-beta2.patch1`
- Isaac Sim `6.0.1.0`
- Torch `2.10.0` / CUDA 12.8 wheels
- RSL-RL `5.0.1`
- Unitree RL Lab revision in `patches/unitree_commit.txt`, with `patches/unitree_isaaclab3.patch`

## New files to commit in your `g1-multiskill` branch

- `setup_multiskill.sh`
- `run_multiskill.sh`
- `README_MULTISKILL_VM.md` (optional)
- `patches/unitree_commit.txt` and `patches/unitree_isaaclab3.patch`
- All `source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill` files including `__init__.py`

Add `/.venv-multiskill/` to `.gitignore` before running setup on the VM. The `.deps/` directory is already ignored by this repo.

## First-time setup

```bash
cd /workspace
# Clone your branch if it is not cloned already:
git clone --branch g1-multiskill https://github.com/Duckymomo2004/Humaniod.git Humaniod-MultiSkill
cd /workspace/Humaniod-MultiSkill
bash setup_multiskill.sh --preflight
bash setup_multiskill.sh
# Fetch only the official G1 29 DoF USD files (about 28 MB), without overwriting existing files:
.venv-multiskill/bin/python -B scripts/fetch_multiskill_asset.py
```

The script performs file validations **before** downloading. It uses `.venv-multiskill` and `.deps/*-multiskill`, so it does not write into `.venv`, `.venv-fastsac`, `.deps/IsaacLab`, `.deps/holosoma`, or their checkpoints. Do not run `uv sync` from the project's existing `uv.lock` for MultiSkill.

The Unitree Git commit must be available upstream. The patch is applied exactly once, and setup refuses to change any existing checkout on a different commit. Repeated setup runs reuse downloads/checkout and skip a patch that is already present.

## Verification/training

```bash
# Read the NVIDIA software EULA first, then accept it:
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh --preflight
./run_multiskill.sh --smoke      # 1 real G1 environment, 2 PPO iterations
./run_multiskill.sh --smoke64    # 64 real G1 environments, 2 PPO iterations
```

The launcher uses `scripts/rsl_rl/train.py` **from your Humaniod project**, which explicitly imports `g1_datn.tasks` and recognizes the `--visualizer none` option. It does not use the old `run.sh` or runpy workaround. These commands will attempt real Isaac Sim physics and **require a correctly configured VM/GPU**.

The setup's final check is offline config import, not a successful simulation or PPO update. The earlier asset/config issues can only be validated in the actual simulator.

## Startup recovery verified on RTX 3090 (2026-10-19)

Both smoke commands completed two PPO iterations with exit code 0. The live robot has
29 joint DoFs, 29 controlled joints, and 29 actions. Physics runs on `cuda:0` through
`PhysxCfg`; each test collected 48 control steps per robot (24 per PPO iteration).
The 64-robot test collected 3,072 transitions. This verifies startup and PPO updates,
not locomotion quality or long-run stability.

The smoke commands pass `--kit_first` so AppLauncher starts **before any Lab task or
sensor configs are imported**. Unitree's wildcard MDP imports otherwise bring in
runtime Articulation/SimulationContext and pip USD before Kit, causing native USD
registry collisions. Starting Kit after those imports can also leave duplicate
sensor-config class identities. Do not remove this bootstrap when using PhysX.
For an explicit training-script invocation, include `--kit_first` yourself.

`UNITREE_MODEL_DIR` defaults to `.deps/unitree_model-multiskill` in the launcher and
can point to another existing copy of the official model tree. The fetch script
pins the dataset revision and verifies file hashes. The robot config only replaces
the upstream placeholder asset path; rewards, curriculum, and actuator settings are
unchanged.

Successful runs still emit the duplicate `grpc/health/v1/health.proto` warning.
Removing inherited VirtualGL alone did not fix the original crash; the final smoke
runs succeeded with the inherited VirtualGL libraries present. No framework or
dependency version was changed during recovery.

See [MULTISKILL_RECOVERY_REPORT.md](MULTISKILL_RECOVERY_REPORT.md) for environment
comparison, evidence, checkpoints, the complete joint list, and rollback details.

## Evaluate the 200-iteration walking checkpoint

The local playback script supports the same early Kit startup as training. Use an
explicit checkpoint; `--skip_export` avoids unrelated JIT/ONNX export work.

```bash
cd /workspace/Humaniod-MultiSkill
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh scripts/rsl_rl/play.py \
  --task G1-MultiSkill-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --visualizer newton --kit_first --skip_export --camera_env 0 \
  --max_steps 1000 --real-time
```

`--camera_env 0` follows the selected robot's terrain tile. Without it, the
default camera looks at world origin while the robot may be tens of meters away.
Newton is the viewer; physics remains PhysX CUDA. Use `--visualizer none` for
headless playback and omit the camera option.

The quantitative evaluator restores the saved configuration, loads actor weights
strictly through RSL-RL 5.0.1, uses deterministic inference, and records terminal
physics states before auto-reset. Fixed commands are evaluation inputs; training
rewards, curriculum, observation noise, and randomizations are retained.

```bash
./run_multiskill.sh scripts/rsl_rl/evaluate_multiskill.py \
  --checkpoints \
    logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_100.pt \
    logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 64 --seconds 20 --visualizer none --zero_reference \
  --output run/my-walking-evaluation
.venv-multiskill/bin/python -B scripts/rsl_rl/summarize_multiskill_eval.py \
  --input run/my-walking-evaluation
```

No training is performed by these commands. Results and media from the completed
evaluation are described in `EVALUATION_WALKING_BASELINE_20261019.md`.

If the first smoke run fails, save the *first full traceback* and fix that specific issue. Do not run `--smoke64` until the 1-env test passes. Do not change the original Humaniod/FastSAC version pins to fix MultiSkill issues.

## VM housekeeping

Before training, stop the old `humanoid` supervisor demo or other GPU jobs if they are running, and check `nvidia-smi`. Save checkpoints and logs to persistent storage; some Vast.ai `/workspace` disks are ephemeral. Don't share instance credentials.

Use `bash -n setup_multiskill.sh run_multiskill.sh` to check Bash syntax. Script installation steps need actual Linux GPU, network, and the user's branch/patch; they cannot be fully validated on Windows.

## Troubleshooting
- Missing `g1_multiskill` `__init__.py`: the repo's `.gitignore` has `**/__*`; stage it explicitly with `git add -f .../__init__.py`.
- `uv` missing: install via official uv instructions: https://docs.astral.sh/uv/getting-started/installation/
- Pinned version mismatch: stop and inspect the installed environment instead of forcing an upgrade.
- GPU not visible: solve `nvidia-smi`/driver/GPU instance selection before installing.
- `patch does not apply`: ensure `unitree_commit.txt` matches the original Unitree commit and that the patch file is unchanged after upload.

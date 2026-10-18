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
```

The script performs file validations **before** downloading. It uses `.venv-multiskill` and `.deps/*-multiskill`, so it does not write into `.venv`, `.venv-fastsac`, `.deps/IsaacLab`, `.deps/holosoma`, or their checkpoints. Do not run `uv sync` from the project's existing `uv.lock` for MultiSkill.

The Unitree Git commit must be available upstream. The patch is applied exactly once, and setup refuses to change any existing checkout on a different commit. Repeated setup runs reuse downloads/checkout and skip a patch that is already present.

## Verification/training

```bash
./run_multiskill.sh --preflight
# Read the NVIDIA software EULA first, then accept it:
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh --smoke      # 1 real G1 environment, 2 PPO iterations
./run_multiskill.sh --smoke64    # 64 real G1 environments, 2 PPO iterations
```

The launcher uses `scripts/rsl_rl/train.py` **from your Humaniod project**, which explicitly imports `g1_datn.tasks` and recognizes the `--visualizer none` option. It does not use the old `run.sh` or runpy workaround. These commands will attempt real Isaac Sim physics and **require a correctly configured VM/GPU**.

The setup's final check is offline config import, not a successful simulation or PPO update. The earlier asset/config issues can only be validated in the actual simulator.

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

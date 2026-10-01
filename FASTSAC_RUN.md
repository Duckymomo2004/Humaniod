# G1 FastSAC training

Reference: https://younggyo.me/fastsac-humanoid/
Upstream Holosoma commit: `d18d6cc50f872c15e904a22ceac22313cec955c8`.

Training completed at iteration 50,000 on 2026-10-01 at 08:41 UTC.
The final checkpoint has finite actor, critic and target-critic tensors, and its
ONNX export passes the ONNX checker. The recovery run took about 4 hours 22 minutes,
including initialization; setup and earlier attempts are additional time.

Final checkpoint:
`logs/fastsac_holosoma/hv-g1-manager/20261001_041946-g1_fastsac_recovery-locomotion/model_0050000.pt`.

A 3,000-step evaluation completed successfully (60 simulated seconds). All recorded
numeric channels are finite. Forward/lateral/yaw command tracking RMSE was
0.230 m/s, 0.242 m/s and 0.269 rad/s. Mean body tilt was 1.77 degrees, maximum
4.68 degrees. This evaluation covered one constant mixed velocity command,
including startup; it is not a broad robustness or real-robot validation.
Its trajectory and `evaluation_summary.json` are in
`logs/fastsac_holosoma/hv-g1-manager/20261001_091356-g1_fastsac_recovery_evaluation-eval/`.

The first full run was stopped after iteration 7,300 because critic metrics
exploded beginning around 6,800. Recovery resumed from 6,000 with target
validation and copied metric tensors. The exact cause remains unresolved.

## Run and watch

```bash
bash scripts/setup_holosoma.sh
bash scripts/train_g1_fastsac.sh --headless
```

Open the separate live viewer with `bash scripts/view_g1_fastsac.sh`.
On this instance both processes are already managed by Supervisor:

```bash
sudo supervisorctl status g1-fastsac g1-fastsac-viewer
```

Watch on the [instance desktop](http://118.100.214.170:63770/) using your existing
instance login. The viewer reads actual G1 poses from training; it does not run
independent physics. It shows environment 0, the iteration, and the age of the
latest state. The displayed floor is a reference plane; training uses the
upstream randomized rough terrain. Snapshots publish at most about 6.7 times
per second through an atomic file replacement.

## Configuration and environment

Training uses the authors' `g1_29dof_fast_sac` preset: G1 29 DOF, seed 1,
4,096 environments, 50,000 iterations, batch 8,192, replay length 1,024,
8 updates per step, actor frequency 4, learning rates 0.0003, gamma 0.97,
tau 0.125, 101 atoms on [-20,20], critic width 768, actor width 512,
symmetry augmentation, normalization, compilation and BF16.
All reward, randomization, terrain, observation, action, termination and
curriculum settings come from the upstream preset. The physics backend is
IsaacSim rather than the paper's IsaacGym backend. Physics runs at 200 Hz;
control runs at 50 Hz. This machine has an RTX 5060 Ti, rather than the RTX 4090
used for the paper's 15-minute demonstration.

The separate `.venv-fastsac` environment uses Python 3.11, PyTorch 2.7 with
CUDA 12.8, IsaacSim 5.1 and IsaacLab v2.3.0 (commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`). Installation and execution use uv.
`uv --no-config` isolates installation from the main project's dependency
overrides; `uv run --no-project --python ...` selects this environment.
The exact package snapshot is `run/fastsac/isaacsim_environment.txt`.
The original Python 3.12 / IsaacSim 6 environment remains available.

The entry point builds upstream dataclasses directly because the dynamic Tyro
CLI fails to parse the MuJoCo randomization schema. `LiveFastSACAgent` inherits
upstream FastSAC and only publishes simulator states after collection steps;
it does not change the learning rule. Recovery retains the paper hyperparameters,
checks categorical targets before optimizer updates, copies logged tensor values
to avoid reused storage, and repeats the original 10-step collection warm-up
because checkpoints do not contain replay data. The target check introduces an
eager validation boundary within the compiled update. The original eager and
compiled BF16 projection passed an isolated 100-call probability-invariant
check, so the initial instability has not yet been explained.
The compatibility patch adds finite-value
checks before replay insertion and replaces removed Warp array APIs. It also
contains a close camera initialization for the earlier MuJoCo experiment.

## Validation and earlier attempts

The original scripts depend on undeclared, unavailable `isaaclab_fast_sac`.
The new scripts use Holosoma directly. The installed IsaacSim 6 / IsaacLab 3
stack could not run its IsaacSim backend (`isaacsim.core.utils` was missing).
An initial MuJoCo Warp attempt became non-finite. A diagnostic with learning
disabled reproduced non-finite observations at step 12, so that invalid run
was stopped. Its logs are preserved in `run/fastsac/invalid_initial_run.log`
and `run/fastsac/physics_diagnostic.log`. Its 20-step checkpoint is only a
smoke-test artifact, not a trained policy.

The separate IsaacSim stack passed a 300-iteration, 128-environment check:
finite rewards and losses, finite actor and critic checkpoint tensors,
checkpoint and ONNX export, and clean exit. The native GUI renderer crashed
on this host; headless simulation succeeded. A separate MuJoCo pose viewer
provides the live display without invoking the IsaacSim renderer.

## Artifacts and evaluation

Full training logs: `run/fastsac/training.log` (includes earlier run output).
Viewer logs: `run/fastsac/viewer.log`.
Training artifacts: `logs/fastsac_holosoma/hv-g1-manager/`.
The serialized `holosoma_config.yaml` in each run directory records its actual
configuration. Checkpoints save every 1,000 iterations and at the final step,
along with ONNX exports and TensorBoard event files.

After training, evaluate its final checkpoint with:

```bash
bash scripts/evaluate_g1_fastsac.sh path/to/model_0050000.pt --steps 3000 --headless
```

This loads the checkpoint's configuration, applies upstream single-environment
evaluation overrides, performs deterministic inference for 60 simulated
seconds, and records `trajectory.npz` in a separate evaluation directory.
The trajectory includes commands, root states, joints, actions and torques.
Summarize it using the same uv environment:

```bash
uv run --no-project --python .venv-fastsac/bin/python python scripts/fast_sac/summarize_evaluation.py path/to/trajectory.npz
```

This checks finite values and reports command tracking RMSE in the robot's body
frame, body tilt, command coverage and the recorded duration. Results describe
that trajectory, including startup and resets; they do not establish real-robot
performance.
The final policy evaluation described above completed successfully. A screenshot confirming the
live state display is `run/fastsac/isaacsim_live_view.png`.

## Run the finished policy

```bash
bash scripts/evaluate_g1_fastsac.sh logs/fastsac_holosoma/hv-g1-manager/20261001_041946-g1_fastsac_recovery-locomotion/model_0050000.pt --demo --headless
FASTSAC_STATE_FILE=run/fastsac/demo_state.npz bash scripts/view_g1_fastsac.sh
```

The demo runs deterministic inference continuously without trajectory recording
or optimizer updates. It caps simulation stepping at 50 Hz; actual speed may
be lower. Supervisor currently manages `g1-fastsac-demo` and the viewer, which
reads its separate `demo_state.npz` stream. Stop with
`sudo supervisorctl stop g1-fastsac-demo g1-fastsac-viewer`.

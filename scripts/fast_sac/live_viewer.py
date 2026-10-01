"""Display actual G1 training states; no independent physics stepping."""

import time
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = mujoco.MjSpec.from_file(
    str(ROOT / ".deps/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
)
spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01], rgba=[0.25, 0.3, 0.35, 1]
)
spec.worldbody.add_light(
    pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.9, 0.9, 0.9]
)
model = spec.compile()
model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
data = mujoco.MjData(model)
state_file = ROOT / os.environ.get("FASTSAC_STATE_FILE", "run/fastsac/live_state.npz")
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0
    while viewer.is_running():
        if state_file.exists():
            with np.load(state_file, allow_pickle=False) as state:
                root, joints = state["root"], state["joints"]
                if np.isfinite(root).all() and np.isfinite(joints).all():
                    data.qpos[:3] = root[:3]
                    data.qpos[3:7] = root[[6, 3, 4, 5]]  # xyzw -> wxyz
                    for name, position in zip(state["joint_names"], joints, strict=True):
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
                        if joint_id < 0:
                            raise ValueError(f"Training joint missing from viewer: {name}")
                        data.qpos[model.jnt_qposadr[joint_id]] = position
                    mujoco.mj_forward(model, data)
                    viewer.cam.lookat[:] = root[:3]
                age = time.time() - float(state["timestamp"])
                mode = str(state["mode"]) if "mode" in state else "Live training"
                label = mode if age < 5.0 else "Simulation state is stale"
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     label, f"G1 env 0 | step {int(state['step'])} | age {age:.1f}s | reference ground")
                ])
        viewer.sync()
        time.sleep(1.0 / 30.0)

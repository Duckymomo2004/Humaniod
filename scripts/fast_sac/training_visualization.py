"""Publish live simulator states without changing the FastSAC update rule."""

import os
import time
from pathlib import Path

import numpy as np
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent


class LiveFastSACAgent(FastSACAgent):
    def setup(self):
        super().setup()
        original_step = self.env.step
        output = Path(os.environ.get("FASTSAC_STATE_FILE", "run/fastsac/live_state.npz"))
        output.parent.mkdir(parents=True, exist_ok=True)
        last_publish = 0.0

        def step_with_snapshot(actions):
            nonlocal last_publish
            step_started = time.monotonic()
            result = original_step(actions)
            now = time.monotonic()
            if now - last_publish >= 0.15:
                sim = self.unwrapped_env.simulator
                root = sim.robot_root_states[0].detach().cpu().numpy().copy()
                joints = sim.dof_pos[0].detach().cpu().numpy().copy()
                temporary = output.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(
                        handle, root=root, joints=joints,
                        joint_names=np.asarray(sim.dof_names), step=self.global_step,
                        timestamp=time.time(), run=str(self.log_dir),
                        mode=os.environ.get("FASTSAC_LIVE_MODE", "Live training"),
                    )
                os.replace(temporary, output)
                last_publish = now
            if os.environ.get("FASTSAC_LIVE_REALTIME") == "1":
                time.sleep(max(0.0, 0.02 - (time.monotonic() - step_started)))
            return result

        self.env.step = step_with_snapshot

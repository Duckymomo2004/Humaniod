"""Project-local reset workaround for Isaac Lab v3.0.0-beta2.patch1 PhysX."""

from isaaclab.envs import ManagerBasedRLEnv


class G1MultiSkillEnv(ManagerBasedRLEnv):
    """Refresh root-derived data after reset, before new observation history.

    The pinned PhysX pose/velocity setters update native state and source
    buffers, but leave these derived buffers current at the same simulation
    timestamp. Termination/reward evaluation has already populated them before
    auto-reset. Invalidating timestamps lets the normal getters recompute from
    the new state, including nonzero reset velocities/orientations.

    This is deliberately scoped to this task and pinned backend. The buffers
    are batched: recomputation also covers non-reset instances, whose physical
    state is untouched. No extra physics step, observation override, or global
    Isaac Lab monkey patch is involved.
    """

    _RESET_ROOT_DERIVED_BUFFERS = (
        "_root_link_vel_w",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_com_pose_w",
        "_projected_gravity_b",
        "_heading_w",
    )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._invalidate_reset_root_cache()

    def _invalidate_reset_root_cache(self):
        """Invalidate only root-derived caches omitted by the pinned setters.

        The setters already invalidate the legacy concatenated state buffers.
        Keep the source pose/COM velocity buffers, history and finite-difference
        joint acceleration state under the upstream reset lifecycle.
        """
        data = self.scene["robot"].data
        for name in self._RESET_ROOT_DERIVED_BUFFERS:
            # Fail visibly if a future backend changes this private interface.
            getattr(data, name).timestamp = -1.0

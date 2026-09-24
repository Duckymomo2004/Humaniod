from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventsCfg as VelocityEventCfg,
)
from .g1_holosoma_robot_cfg import (
    G1_HOLOSOMA_CFG,
    HOLOSOMA_G1_JOINTS,
)

from . import mdp




@configclass
class G1DatnEventCfg(VelocityEventCfg):
    """Domain randomization adapted from HoloSoma G1."""

    # ============================================================
    # Startup randomization
    # ============================================================

    # HoloSoma friction: [0.5, 1.25]
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # HoloSoma:
    # selected link mass *= Uniform(0.9, 1.2)
    link_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[
                    "pelvis",
                    "left_hip_yaw_link",
                    "left_hip_roll_link",
                    "left_hip_pitch_link",
                    "left_knee_link",
                    "right_hip_yaw_link",
                    "right_hip_roll_link",
                    "right_hip_pitch_link",
                    "right_knee_link",
                ],
            ),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # HoloSoma:
    # torso mass += Uniform(-1, 3) kg
    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["torso_link"],
            ),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # HoloSoma COM randomization:
    # x/y/z Â± 0.05 m
    base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["torso_link"],
            ),
            "com_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    # ============================================================
    # Reset randomization
    # ============================================================

    # HoloSoma Kp/Kd *= Uniform(0.9, 1.1)
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*",
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # HoloSoma:
    # q_reset = q_default * Uniform(0.5, 1.5)
    # qdot = 0
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*",
            ),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


    # Reset base vá» initial state.
    # HoloSoma khÃ´ng dÃ¹ng random yaw Â±pi nhÆ° baseline Isaac Lab.
    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # ============================================================
    # During episode
    # ============================================================

    # HoloSoma push:
    # every 5â€“10 s
    # vx/vy += Uniform(-1, 1)
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
            },
        },
    )

@configclass
class G1DatnEnvCfg(G1FlatEnvCfg):
    """Unitree G1 locomotion environment for DATN."""
    events: G1DatnEventCfg = G1DatnEventCfg()


    def __post_init__(self):
        super().__post_init__()

        self.events = G1DatnEventCfg()
        self.events.base_external_force_torque = None                                                                                                       

        self.scene.robot = G1_HOLOSOMA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot = G1_HOLOSOMA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.commands.gait_phase = mdp.GaitPhaseCommandCfg(
            resampling_time_range=(10.0, 10.0),
            gait_period=1.0,
            gait_frequency_randomization_width=0.2,
            randomize_phase=True,
            velocity_command_name="base_velocity",
        )

        # Scene

        # Debug vá»›i Ã­t robot trÆ°á»›c
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        # Episode length
        self.episode_length_s = 20.0

        # Commands
        # vx, vy, wz in [-1, 1]
        # resample every 10 s
        # stand probability = 0.2
        # ============================================================

        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        self.commands.base_velocity.rel_standing_envs = 0.2

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ============================================================
        # Actions
        # ============================================================

        self.actions.joint_pos.joint_names = HOLOSOMA_G1_JOINTS
        self.actions.joint_pos.preserve_order = True
        # q_des = q_default + scale * action
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        # ============================================================
        # Observations
        # ============================================================

        self.observations.policy.base_lin_vel = None

        # Angular velocity scale = 0.25
        self.observations.policy.base_ang_vel.scale = 0.25

        # KhÃ´ng thÃªm noise vÃ o angular velocity
        self.observations.policy.base_ang_vel.noise = None

        # Projected gravity
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.projected_gravity.noise = None

        # Joint position
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_pos.noise = Unoise(
            n_min=-0.01,
            n_max=0.01,
        )

        # Joint velocity
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_vel.noise = Unoise(
            n_min=-0.1,
            n_max=0.1,
        )

        self.observations.policy.sin_phase = ObsTerm(
            func=mdp.sin_phase,
            params={
                "command_name": "gait_phase",
            },
        )

        self.observations.policy.cos_phase = ObsTerm(
            func=mdp.cos_phase,
            params={
                "command_name": "gait_phase",
            },
        )
        # ============================================================
        # Rewards
        # ============================================================

        # tracking_lin_vel = 2.0
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5

        # tracking_ang_vel = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # penalty_ang_vel_xy = -1.0
        self.rewards.ang_vel_xy_l2.weight = -1.0

        # penalty_orientation = -10.0
        self.rewards.flat_orientation_l2.weight = -10.0

        # penalty_action_rate = -2.0
        self.rewards.action_rate_l2.weight = -2.0

        self.rewards.termination_penalty.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_slide.weight = 0.0

        self.rewards.dof_pos_limits.weight = 0.0

        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None

        self.rewards.pose = RewTerm(
            func=mdp.pose,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",

                        "right_hip_pitch_joint",
                        "right_hip_roll_joint",
                        "right_hip_yaw_joint",
                        "right_knee_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",

                        "waist_yaw_joint",
                        "waist_roll_joint",
                        "waist_pitch_joint",

                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                        "left_wrist_yaw_joint",

                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                        "right_wrist_roll_joint",
                        "right_wrist_pitch_joint",
                        "right_wrist_yaw_joint",
                    ],
                ),

                "pose_weights": [
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,

                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,

                    50.0,
                    50.0,
                    50.0,

                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,

                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
            },
        )


        self.rewards.close_feet = RewTerm(
    func=mdp.penalty_close_feet_xy,
    weight=-10.0,
    params={
        "close_feet_threshold": 0.15,
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
        ),
    },
)

        #Hai reward chÃ¢n

        self.rewards.feet_orientation = RewTerm(
            func=mdp.penalty_feet_ori,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                ),
            },
        )

        self.rewards.feet_phase = RewTerm(
            func=mdp.feet_phase,
            weight=5.0,
            params={
                "swing_height": 0.09,
                "tracking_sigma": 0.008,
                "command_name": "gait_phase",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                    ],
                ),
            },
        )
        # Alive reward = +1
        self.rewards.alive = RewTerm(
            func=base_mdp.is_alive,
            weight=1.0,
        )

        # ============================================================
        # Termination
        #
        # Terminate khi pelvis / shoulder / hip
        # cháº¡m máº¡nh hÆ¡n threshold.
        # ============================================================

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                "pelvis",
                ".*_hip_.*_link",
                ".*_shoulder_.*_link",
            ],
        )

        self.terminations.base_contact.params["threshold"] = 1.0

        # ============================================================
        # Domain Randomization - HoloSoma
        # ============================================================

        # ------------------------------------------------------------
        # Friction
        #
        # HoloSoma:
        # friction = [0.5, 1.25]
        # ------------------------------------------------------------
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=".*",
                ),
                "static_friction_range": (0.5, 1.25),
                "dynamic_friction_range": (0.5, 1.25),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        # ------------------------------------------------------------
        # Link mass
        #
        # HoloSoma:
        # mass_link *= Uniform(0.9, 1.2)
        # ------------------------------------------------------------
        self.events.randomize_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "pelvis",

                        "left_hip_yaw_link",
                        "left_hip_roll_link",
                        "left_hip_pitch_link",
                        "left_knee_link",

                        "right_hip_yaw_link",
                        "right_hip_roll_link",
                        "right_hip_pitch_link",
                        "right_knee_link",
                    ],
                ),
                "mass_distribution_params": (0.9, 1.2),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )

        # ------------------------------------------------------------
        # Torso mass
        #
        # HoloSoma:
        # torso_mass += Uniform(-1, 3) kg
        # ------------------------------------------------------------
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["torso_link"],
                ),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
                "recompute_inertia": True,
            },
        )

        # ------------------------------------------------------------
        # Torso COM
        #
        # HoloSoma:
        # x/y/z += Uniform(-0.05, 0.05)
        # ------------------------------------------------------------
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["torso_link"],
                ),
                "com_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                },
            },
        )

        # ------------------------------------------------------------
        # Randomize Kp / Kd
        #
        # HoloSoma:
        # Kp *= Uniform(0.9, 1.1)
        # Kd *= Uniform(0.9, 1.1)
        # ------------------------------------------------------------
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*",
                ),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # ------------------------------------------------------------
        # Joint reset
        #
        # HoloSoma:
        #
        # q_reset = q_default * Uniform(0.5, 1.5)
        # qdot_reset = 0
        # ------------------------------------------------------------
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=".*",
                ),
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )

        # ------------------------------------------------------------
        # Push robot
        #
        # HoloSoma:
        # interval = 5 - 10 seconds
        # max vx/vy = 1 m/s
        # ------------------------------------------------------------
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                },
            },
        )

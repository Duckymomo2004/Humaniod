from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import G1_29DOF_CFG


HOLOSOMA_G1_JOINTS = [
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
]


G1_HOLOSOMA_CFG = G1_29DOF_CFG.copy()

# cần contact sensor cho reward/termination
G1_HOLOSOMA_CFG.spawn.activate_contact_sensors = True

# HoloSoma spawn ở z = 0.8
G1_HOLOSOMA_CFG.init_state.pos = (0.0, 0.0, 0.8)

# ============================================================
# Default pose của HoloSoma
# ============================================================

G1_HOLOSOMA_CFG.init_state.joint_pos = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,

    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,

    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,

    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,

    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

G1_HOLOSOMA_CFG.init_state.joint_vel = {
    ".*": 0.0,
}


# ============================================================
# Actuator
#
# HoloSoma tự tính:
#
# tau = Kp (q_des - q) - Kd qdot
#
# rồi clip torque.
#
# Vì vậy IdealPDActuator gần với HoloSoma hơn
# ImplicitActuator.
# ============================================================

G1_HOLOSOMA_CFG.actuators = {
    "all_joints": IdealPDActuatorCfg(
        joint_names_expr=HOLOSOMA_G1_JOINTS,

        stiffness={
            ".*_hip_pitch_joint": 40.179238471,
            ".*_hip_roll_joint": 99.098427777,
            ".*_hip_yaw_joint": 40.179238471,
            ".*_knee_joint": 99.098427777,

            ".*_ankle_pitch_joint": 28.501246196,
            ".*_ankle_roll_joint": 28.501246196,

            "waist_yaw_joint": 40.179238471,
            "waist_roll_joint": 28.501246196,
            "waist_pitch_joint": 28.501246196,

            ".*_shoulder_pitch_joint": 14.250623098,
            ".*_shoulder_roll_joint": 14.250623098,
            ".*_shoulder_yaw_joint": 14.250623098,

            ".*_elbow_joint": 14.250623098,

            ".*_wrist_roll_joint": 14.250623098,
            ".*_wrist_pitch_joint": 16.778327481,
            ".*_wrist_yaw_joint": 16.778327481,
        },

        damping={
            ".*_hip_pitch_joint": 2.557889765,
            ".*_hip_roll_joint": 6.308801854,
            ".*_hip_yaw_joint": 2.557889765,
            ".*_knee_joint": 6.308801854,

            ".*_ankle_pitch_joint": 1.814445687,
            ".*_ankle_roll_joint": 1.814445687,

            "waist_yaw_joint": 2.557889765,
            "waist_roll_joint": 1.814445687,
            "waist_pitch_joint": 1.814445687,

            ".*_shoulder_pitch_joint": 0.907222843,
            ".*_shoulder_roll_joint": 0.907222843,
            ".*_shoulder_yaw_joint": 0.907222843,

            ".*_elbow_joint": 0.907222843,

            ".*_wrist_roll_joint": 0.907222843,
            ".*_wrist_pitch_joint": 1.068141502,
            ".*_wrist_yaw_joint": 1.068141502,
        },

        effort_limit={
            ".*_hip_pitch_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_yaw_joint": 88.0,
            ".*_knee_joint": 139.0,

            ".*_ankle_pitch_joint": 50.0,
            ".*_ankle_roll_joint": 50.0,

            "waist_yaw_joint": 88.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,

            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,

            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },

        velocity_limit={
            ".*_hip_pitch_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_yaw_joint": 32.0,
            ".*_knee_joint": 20.0,

            ".*_ankle_pitch_joint": 37.0,
            ".*_ankle_roll_joint": 37.0,

            "waist_yaw_joint": 32.0,
            "waist_roll_joint": 37.0,
            "waist_pitch_joint": 37.0,

            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,

            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },

        armature={
            ".*_hip_pitch_joint": 0.010177520,
            ".*_hip_roll_joint": 0.025101925,
            ".*_hip_yaw_joint": 0.010177520,
            ".*_knee_joint": 0.025101925,

            ".*_ankle_pitch_joint": 0.007219450,
            ".*_ankle_roll_joint": 0.007219450,

            "waist_yaw_joint": 0.010177520,
            "waist_roll_joint": 0.007219450,
            "waist_pitch_joint": 0.007219450,

            ".*_shoulder_pitch_joint": 0.003609725,
            ".*_shoulder_roll_joint": 0.003609725,
            ".*_shoulder_yaw_joint": 0.003609725,
            ".*_elbow_joint": 0.003609725,

            ".*_wrist_roll_joint": 0.003609725,
            ".*_wrist_pitch_joint": 0.00425,
            ".*_wrist_yaw_joint": 0.00425,
        },

        friction=0.0,
    ),
}
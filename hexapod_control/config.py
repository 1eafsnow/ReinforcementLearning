from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent / "hexapod_description"
LEG_ORDER = ("RR", "RM", "RF", "LR", "LM", "LF")
JOINT_TYPE_ORDER = ("coxa", "femur", "tibia", "tarsus")
JOINT_NAMES = tuple(f"{joint}_joint_{leg}" for leg in LEG_ORDER for joint in JOINT_TYPE_ORDER)
ACTUATOR_NAMES = tuple(f"{name}_motor" for name in JOINT_NAMES)
FOOT_SITE_NAMES = tuple(f"foot_{leg}_site" for leg in LEG_ORDER)
FOOT_GEOM_NAMES = tuple(f"foot_{leg}_contact_geom" for leg in LEG_ORDER)


@dataclass(frozen=True)
class RewardConfig:
    velocity_sigma: float = 0.06
    standing_velocity_sigma: float = 0.025
    standing_angular_velocity_sigma: float = 0.20
    yaw_pose_sigma: float = float(np.deg2rad(8.0))
    pitch_pose_sigma: float = float(np.deg2rad(4.0))
    roll_pose_sigma: float = float(np.deg2rad(4.0))
    yaw_rate_sigma: float = 0.50
    pitch_rate_sigma: float = 0.30
    yaw_error_to_rate_gain: float = 2.0
    pitch_error_to_rate_gain: float = 2.0
    max_target_yaw_rate: float = 1.0
    max_target_pitch_rate: float = 0.5
    yaw_motion_gate_angle: float = float(np.deg2rad(15.0))
    pitch_motion_gate_angle: float = float(np.deg2rad(5.0))
    standing_pose_tolerance: float = float(np.deg2rad(1.0))
    height_sigma: float = 0.025
    swing_clearance_min: float = 0.020
    swing_clearance_max: float = 0.050
    swing_clearance_speed_range: Tuple[float, float] = (0.03, 0.25)
    clearance_sigma: float = 0.015
    clearance_deficit_scale: float = 0.020
    swing_contact_gate_start: float = 0.10
    swing_contact_gate_end: float = 0.40
    vertical_velocity_scale: float = 0.20
    slip_velocity_scale: float = 0.20
    joint_velocity_scale: float = 10.0
    joint_acceleration_scale: float = 100.0
    power_scale: float = 10.0
    velocity_weight: float = 3.00
    progress_weight: float = 0.60
    yaw_pose_weight: float = 1.00
    pitch_pose_weight: float = 0.60
    roll_pose_weight: float = 0.40
    pose_rate_weight: float = 0.10
    standing_weight: float = 0.50
    standing_contact_weight: float = 0.50
    height_weight: float = 0.30
    tripod_weight: float = 0.15
    clearance_weight: float = 0.15
    clearance_deficit_weight: float = 0.20
    swing_contact_weight: float = 0.35
    stall_weight: float = 1.00
    flight_weight: float = 0.50
    vertical_velocity_weight: float = 0.10
    slip_weight: float = 0.35
    action_rate_weight: float = 0.25
    action_acceleration_weight: float = 0.10
    torque_weight: float = 0.02
    power_weight: float = 0.01
    joint_velocity_weight: float = 0.02
    joint_acceleration_weight: float = 0.05
    forbidden_contact_weight: float = 2.00
    termination_penalty: float = 10.0
    invalid_physics_penalty: float = 20.0


@dataclass(frozen=True)
class EnvConfig:
    xml_path: Path = PROJECT_DIR / "mjcf" / "scene.xml"
    mesh_dir: Path = PROJECT_DIR / "meshes"
    base_name: str = "base_link"
    free_joint_name: str = "floating_base"
    floor_name: str = "floor"
    joint_names: Tuple[str, ...] = JOINT_NAMES
    actuator_names: Tuple[str, ...] = ACTUATOR_NAMES
    foot_site_names: Tuple[str, ...] = FOOT_SITE_NAMES
    foot_geom_names: Tuple[str, ...] = FOOT_GEOM_NAMES
    q_nominal: Tuple[float, ...] = (0.0, -0.10, 0.20, -0.10) * 3 + (0.0,  0.10, -0.20, 0.10) * 3
    kp_per_leg: Tuple[float, ...] = (10.0, 18.0, 16.0, 8.0)
    kd_per_leg: Tuple[float, ...] = (0.35, 0.55, 0.55, 0.30)
    action_scale_per_leg: Tuple[float, ...] = (0.20, 0.30, 0.35, 0.25)
    armature_per_leg: Tuple[float, ...] = (0.002, 0.004, 0.004, 0.002)
    damping_per_leg: Tuple[float, ...] = (0.05, 0.08, 0.08, 0.05)
    frame_skip: int = 10
    max_episode_steps: int = 1200
    target_height: float = 0.134
    initial_ground_clearance: float = 0.002
    min_base_height: float = 0.090
    max_base_height: float = 0.210
    max_tilt_deg: float = 40.0
    joint_limit_margin: float = 0.05
    action_filter_alpha: float = 0.08
    action_rate_limit: float = 8.0
    contact_force_threshold: float = 0.50
    # phase每转一周为一个完整Tripod周期，支撑组在每半周期切换一次。
    gait_frequency_range: Tuple[float, float] = (1.00, 1.80)
    gait_frequency_speed_range: Tuple[float, float] = (0.03, 0.25)
    # 平移Command在机器人实时Yaw坐标系中定义，方向角0为前进、pi/2为向左。
    command_speed_range: Tuple[float, float] = (0.03, 0.07)
    command_direction_range: Tuple[float, float] = (-float(np.pi), float(np.pi))
    # 第一阶段保持Yaw/Pitch范围为0，后续只改范围和概率即可继续训练同一95维网络。
    command_yaw_offset_range: Tuple[float, float] = (0.0, 0.0)
    command_pitch_range: Tuple[float, float] = (0.0, 0.0)
    command_pitch_limit: Tuple[float, float] = (-float(np.deg2rad(10.0)), float(np.deg2rad(10.0)))
    command_yaw_slew_rate: float = float(np.deg2rad(45.0))
    command_pitch_slew_rate: float = float(np.deg2rad(15.0))
    command_resample_time_range: Tuple[float, float] = (2.0, 4.0)
    resample_commands_during_episode: bool = False
    standing_command_probability: float = 0.20
    turn_in_place_probability: float = 0.0
    pitch_in_place_probability: float = 0.0
    initial_yaw_range: Tuple[float, float] = (-float(np.pi), float(np.pi))
    initial_roll_pitch_deg: float = 1.0
    joint_position_noise: float = 0.005
    joint_velocity_noise: float = 0.01
    base_position_xy_noise: float = 0.001
    base_linear_velocity_noise: float = 0.01
    base_angular_velocity_noise: float = 0.02
    floor_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    floor_solref: Tuple[float, float] = (0.01, 1.0)
    floor_solimp: Tuple[float, float, float] = (0.9, 0.95, 0.001)
    solver_iterations: int = 50
    solver_ls_iterations: int = 20
    use_tripod_gait_reward: bool = True
    terminate_on_forbidden_contact: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    forbidden_contact_force_threshold: float = 1.0
    forbidden_contact_grace_steps: int = 20
    forbidden_contact_persistence_steps: int = 3
    command_velocity_scale: float = 0.30
    command_pitch_scale: float = float(np.deg2rad(10.0))


@dataclass(frozen=True)
class SACConfig:
    obs_dim: int = 95
    action_dim: int = 24
    hidden_dims: Tuple[int, ...] = (512, 512)
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 2e-4
    batch_size: int = 512
    replay_size: int = 400_000
    start_steps: int = 10_000
    update_after: int = 10_000
    updates_per_step: int = 1
    initial_alpha: float = 0.10
    target_entropy: Optional[float] = -12.0
    log_std_min: float = -5.0
    log_std_max: float = 0.5
    reward_scale: float = 0.20
    gradient_clip: float = 10.0
    total_steps: int = 2_000_000
    device: str = "auto"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    total_steps: int = 2_000_000
    checkpoint_every: int = 100_000
    eval_every: int = 50_000
    eval_episodes: int = 5
    log_updates_every: int = 5_000
    warmup_action_limit: float = 0.15
    velocity_curriculum: Tuple[Tuple[int, float, float], ...] = (
        (0, 0.03, 0.07),
        (300_000, 0.04, 0.12),
        (800_000, 0.05, 0.18),
        (1_400_000, 0.06, 0.25),
        (2_000_000, 0.10, 0.35),
        (2_600_000, 0.15, 0.50),
    )


ENV_CONFIG = EnvConfig()
SAC_CONFIG = SACConfig()
TRAIN_CONFIG = TrainConfig()
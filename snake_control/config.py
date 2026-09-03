from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


PROJECT_DIR = Path(__file__).resolve().parent.parent / "snake_description"
JOINT_NAMES = ("front_joint1", "front_joint2", "back_joint1", "back_joint2")
ACTUATOR_NAMES = ("f1_motor", "f2_motor", "b1_motor", "b2_motor")
FRONT_TRACK_PAD_NAMES = ("front_track_pad1_geom", "front_track_pad2_geom", "front_track_pad3_geom")
BACK_TRACK_PAD_NAMES = ("back_track_pad1_geom", "back_track_pad2_geom", "back_track_pad3_geom")
SUPPORT_GEOM_NAMES = FRONT_TRACK_PAD_NAMES + BACK_TRACK_PAD_NAMES + ("base_link_pad",)


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 18.0
    heading_weight: float = 0.12
    speed_toward_goal_weight: float = 0.25
    clearance_weight: float = 0.30
    clearance_distance: float = 0.65
    avoid_turn_weight: float = 0.45
    blocked_stall_weight: float = 0.20
    avoid_start_distance: float = 1.25
    avoid_full_distance: float = 0.45
    avoid_front_fraction: float = 0.25
    avoid_side_difference_scale: float = 0.50
    avoid_turn_rate_scale: float = 1.00
    avoid_heading_relaxation: float = 0.90
    avoid_speed_relaxation: float = 0.85
    avoid_negative_progress_relaxation: float = 0.85
    avoid_symmetry_turn_bonus: float = 0.25
    stall_weight: float = 0.12
    action_rate_weight: float = 0.04
    action_acceleration_weight: float = 0.015
    joint_pose_weight: float = 0.015
    joint_velocity_weight: float = 0.01
    joint_torque_weight: float = 0.008
    track_torque_weight: float = 0.008
    time_penalty: float = 0.005
    success_reward: float = 30.0
    collision_penalty: float = 25.0
    termination_penalty: float = 12.0
    invalid_physics_penalty: float = 30.0


@dataclass(frozen=True)
class EnvConfig:
    xml_path: Path = PROJECT_DIR / "mjcf" / "scene_slider.xml"
    base_name: str = "base_link"
    free_joint_name: str = "floating_base"
    floor_name: str = "floor"
    lidar_site_name: str = "lidar_site"
    obstacle_prefix: str = "obstacle"
    joint_names: Tuple[str, ...] = JOINT_NAMES
    actuator_names: Tuple[str, ...] = ACTUATOR_NAMES
    front_track_pad_names: Tuple[str, ...] = FRONT_TRACK_PAD_NAMES
    back_track_pad_names: Tuple[str, ...] = BACK_TRACK_PAD_NAMES
    support_geom_names: Tuple[str, ...] = SUPPORT_GEOM_NAMES
    q_nominal: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    kp: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0)
    kd: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0)
    joint_action_scale: Tuple[float, ...] = (0.90, 0.90, 0.90, 0.90)
    joint_limit_margin: float = 0.10
    frame_skip: int = 20
    max_episode_steps: int = 600
    action_filter_alpha: float = 0.15
    action_rate_limit: float = 8.0
    track_effective_radius: float = 0.075
    track_virtual_inertia: float = 0.08
    track_motor_damping: float = 0.20
    track_kd: float = 8.0
    track_torque_limit: float = 36.0
    track_speed_center: float = 4.0
    track_speed_scale: float = 6.0
    track_speed_limit: float = 15.0
    lidar_rows: int = 4
    lidar_cols: int = 120
    lidar_hfov_deg: float = 120.0
    lidar_vfov_deg: float = 45.0
    lidar_max_range: float = 30.0
    lidar_scan_hz: float = 10.0
    lidar_group: int = 0
    floor_lidar_group: int = 5
    goal_distance_range: Tuple[float, float] = (2.5, 4.0)
    goal_lateral_range: Tuple[float, float] = (-0.60, 0.60)
    goal_radius: float = 0.30
    obstacle_path_fraction_range: Tuple[float, float] = (0.38, 0.68)
    obstacle_lateral_offset_range: Tuple[float, float] = (-0.40, 0.40)
    obstacle_start_clearance: float = 1.10
    obstacle_goal_clearance: float = 0.65
    initial_ground_clearance: float = 0.003
    initial_yaw_noise_deg: float = 8.0
    joint_position_noise: float = 0.02
    joint_velocity_noise: float = 0.03
    base_position_xy_noise: float = 0.03
    base_linear_velocity_noise: float = 0.02
    base_angular_velocity_noise: float = 0.03
    max_tilt_deg: float = 55.0
    min_base_height: float = 0.05
    max_base_height: float = 1.00
    world_xy_limit: float = 4.80
    collision_force_threshold: float = 0.5
    goal_position_scale: float = 4.0
    linear_velocity_scale: float = 1.0
    angular_velocity_scale: float = 3.0
    joint_velocity_scale: float = 8.0
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class SACConfig:
    obs_dim: int = 510
    action_dim: int = 6
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
    target_entropy: Optional[float] = -3.0
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
    eval_episodes: int = 8
    log_updates_every: int = 5_000
    warmup_action_limit: float = 0.50
    obstacle_offset_curriculum: Tuple[Tuple[int, float], ...] = (
        (0, 0.60),
        (250_000, 0.45),
        (600_000, 0.35),
        (1_000_000, 0.25),
        (1_500_000, 0.15),
    )


ENV_CONFIG = EnvConfig()
SAC_CONFIG = SACConfig()
TRAIN_CONFIG = TrainConfig()

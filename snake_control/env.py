from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from config import ENV_CONFIG, EnvConfig


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    return q / norm


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class SnakeAvoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: EnvConfig = ENV_CONFIG, render_mode: Optional[str] = None):
        super().__init__()
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.cfg = config
        self.reward_cfg = config.reward
        self._validate_config()
        self._require_surfacevel_support()
        self.model = self._load_model(config.xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = int(config.frame_skip)
        self.policy_dt = float(self.model.opt.timestep * self.frame_skip)
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.policy_dt))

        self.base_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, config.base_name)
        free_joint_id = self._require_id(mujoco.mjtObj.mjOBJ_JOINT, config.free_joint_name)
        self.free_qpos_adr = int(self.model.jnt_qposadr[free_joint_id])
        self.free_dof_adr = int(self.model.jnt_dofadr[free_joint_id])
        self.joint_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in config.joint_names], dtype=np.int32)
        self.joint_qpos_adr = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_dof_adr = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.joint_lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_upper = self.model.jnt_range[self.joint_ids, 1].copy()
        self.actuator_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in config.actuator_names], dtype=np.int32)
        self.ctrl_lower = self.model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.ctrl_upper = self.model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.floor_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, config.floor_name)
        self.lidar_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, config.lidar_site_name)
        self.front_track_pad_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.front_track_pad_names], dtype=np.int32)
        self.back_track_pad_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.back_track_pad_names], dtype=np.int32)
        self.support_geom_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.support_geom_names], dtype=np.int32)
        self.obstacle_geom_ids = self._discover_obstacle_geoms()
        self.obstacle_geom_set = frozenset(int(value) for value in self.obstacle_geom_ids)
        self.obstacle_z = self.model.geom_pos[self.obstacle_geom_ids, 2].copy()

        self.model.geom_group[self.floor_geom_id] = int(config.floor_lidar_group)
        self.model.geom_group[self.obstacle_geom_ids] = int(config.lidar_group)
        self.model.flg_surfacevel = 1

        self.action_dim = 6
        self.q_nominal = np.asarray(config.q_nominal, dtype=np.float64)
        self.kp = np.asarray(config.kp, dtype=np.float64)
        self.kd = np.asarray(config.kd, dtype=np.float64)
        self.joint_action_scale = np.asarray(config.joint_action_scale, dtype=np.float64)
        if self.q_nominal.shape != (4,) or self.kp.shape != (4,) or self.kd.shape != (4,) or self.joint_action_scale.shape != (4,):
            raise ValueError("q_nominal, kp, kd and joint_action_scale must each contain four values")

        self.track_pad_ids = (self.front_track_pad_ids, self.back_track_pad_ids)
        self.track_omega = np.zeros(2, dtype=np.float64)
        self.track_angle = np.zeros(2, dtype=np.float64)
        self.track_load_tau = np.zeros(2, dtype=np.float64)
        self.track_motor_tau = np.zeros(2, dtype=np.float64)

        self.lidar_dirs_local = self._build_lidar_directions()
        self.lidar_nray = int(config.lidar_rows * config.lidar_cols)
        self.lidar_geomid = np.full(self.lidar_nray, -1, dtype=np.int32)
        self.lidar_dist = np.full(self.lidar_nray, -1.0, dtype=np.float64)
        self.lidar_geomgroup = np.zeros(6, dtype=np.uint8)
        self.lidar_geomgroup[int(config.lidar_group)] = 1
        self.lidar_scan = np.full((config.lidar_rows, config.lidar_cols), config.lidar_max_range, dtype=np.float32)
        self.lidar_interval_steps = max(1, int(round(1.0 / (config.lidar_scan_hz * self.policy_dt))))

        self.filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.previous_filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.older_filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.goal_position = np.zeros(3, dtype=np.float64)
        self.previous_goal_distance = 0.0
        self.episode_progress = 0.0
        self.path_length = 0.0
        self.last_base_xy = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self._episode_ended = False
        self.obstacle_offset_limit = max(abs(float(config.obstacle_lateral_offset_range[0])), abs(float(config.obstacle_lateral_offset_range[1])))

        lidar_dim = config.lidar_rows * config.lidar_cols
        self.observation_dim = lidar_dim + 30
        self._last_valid_obs = np.zeros(self.observation_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_dim,), dtype=np.float32)
        self.init_qpos = self.model.qpos0.copy()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if render_mode == "human" else None

        self._set_all_track_surfacevel(0.0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _version_tuple() -> Tuple[int, int, int]:
        values = []
        for part in mujoco.__version__.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            values.append(int(digits) if digits else 0)
        while len(values) < 3:
            values.append(0)
        return tuple(values[:3])

    @classmethod
    def _require_surfacevel_support(cls) -> None:
        if cls._version_tuple() < (3, 11, 0):
            raise RuntimeError(f"SnakeAvoidEnv requires MuJoCo >= 3.11.0 for geom surfacevel; installed version is {mujoco.__version__}")

    def _validate_config(self) -> None:
        if self.cfg.frame_skip < 1 or self.cfg.max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        if len(self.cfg.joint_names) != 4 or len(self.cfg.actuator_names) != 4:
            raise ValueError("Snake slider model requires exactly four joints and four joint actuators")
        if len(self.cfg.front_track_pad_names) != 3 or len(self.cfg.back_track_pad_names) != 3:
            raise ValueError("Each virtual track requires exactly three fixed surfacevel pads")
        if not 0.0 < self.cfg.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must lie in (0, 1]")
        if self.cfg.action_rate_limit <= 0.0 or self.cfg.track_effective_radius <= 0.0 or self.cfg.track_virtual_inertia <= 0.0:
            raise ValueError("Action rate and virtual track parameters must be positive")
        if self.cfg.lidar_rows < 1 or self.cfg.lidar_cols < 1 or self.cfg.lidar_scan_hz <= 0.0 or self.cfg.lidar_max_range <= 0.0:
            raise ValueError("LiDAR dimensions, rate and range must be positive")
        if not 0 <= self.cfg.lidar_group < 6 or not 0 <= self.cfg.floor_lidar_group < 6:
            raise ValueError("MuJoCo geom groups must be in [0, 5]")
        self._validate_range(self.cfg.goal_distance_range, "goal_distance_range")
        self._validate_range(self.cfg.goal_lateral_range, "goal_lateral_range")
        self._validate_range(self.cfg.obstacle_path_fraction_range, "obstacle_path_fraction_range")
        self._validate_range(self.cfg.obstacle_lateral_offset_range, "obstacle_lateral_offset_range")

    @staticmethod
    def _validate_range(value: Tuple[float, float], name: str) -> Tuple[float, float]:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
            raise ValueError(f"{name} must be a finite (low, high) pair")
        return float(array[0]), float(array[1])

    @staticmethod
    def _load_model(xml_path: Path) -> mujoco.MjModel:
        xml_path = Path(xml_path).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        return model

    def _require_id(self, object_type, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return object_id

    def _discover_obstacle_geoms(self) -> np.ndarray:
        ids = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is not None and name.startswith(self.cfg.obstacle_prefix):
                ids.append(geom_id)
        if not ids:
            raise RuntimeError(f"No obstacle geoms beginning with '{self.cfg.obstacle_prefix}' were found in {self.cfg.xml_path}")
        return np.asarray(ids, dtype=np.int32)

    def _build_lidar_directions(self) -> np.ndarray:
        h_step = np.deg2rad(self.cfg.lidar_hfov_deg / self.cfg.lidar_cols)
        v_step = np.deg2rad(self.cfg.lidar_vfov_deg / self.cfg.lidar_rows)
        azimuth = (np.arange(self.cfg.lidar_cols, dtype=np.float64) - (self.cfg.lidar_cols - 1) * 0.5) * h_step
        elevation = (np.arange(self.cfg.lidar_rows, dtype=np.float64) - (self.cfg.lidar_rows - 1) * 0.5) * v_step
        az, el = np.meshgrid(azimuth, elevation)
        directions = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1)
        return np.ascontiguousarray(directions.reshape(-1, 3), dtype=np.float64)

    def set_obstacle_offset_limit(self, max_abs_offset: float) -> None:
        if not np.isfinite(max_abs_offset) or max_abs_offset < 0.0:
            raise ValueError("max_abs_offset must be finite and non-negative")
        self.obstacle_offset_limit = float(max_abs_offset)

    def _get_heading_yaw(self) -> float:
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        forward = -rotation[:, 0]
        return float(np.arctan2(forward[1], forward[0]))

    def _get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        velocity_world = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity_world, 0)
        yaw = self._get_heading_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_heading = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return world_to_heading @ velocity_world[:3], world_to_heading @ velocity_world[3:]

    def _get_goal_state(self) -> Tuple[np.ndarray, float, float]:
        base_xy = self.data.xpos[self.base_id, :2]
        delta_world = self.goal_position[:2] - base_xy
        yaw = self._get_heading_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        world_to_heading = np.array([[c, s], [-s, c]], dtype=np.float64)
        goal_local = world_to_heading @ delta_world
        goal_distance = float(np.linalg.norm(goal_local))
        heading_error = float(np.arctan2(goal_local[1], goal_local[0])) if goal_distance > 1e-9 else 0.0
        return goal_local, goal_distance, heading_error

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        base_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        base_ang_vel, base_lin_vel = self._get_base_velocity()
        goal_local, goal_distance, heading_error = self._get_goal_state()
        return {
            "base_pos": self.data.xpos[self.base_id].copy(),
            "projected_gravity": base_rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64),
            "base_ang_vel": base_ang_vel,
            "base_lin_vel": base_lin_vel,
            "q": self.data.qpos[self.joint_qpos_adr].copy(),
            "dq": self.data.qvel[self.joint_dof_adr].copy(),
            "goal_local": goal_local,
            "goal_distance": np.array(goal_distance),
            "heading_error": np.array(heading_error),
            "track_omega": self.track_omega.copy(),
        }

    def _scan_lidar(self) -> None:
        origin = self.data.site_xpos[self.lidar_site_id].copy()
        rotation = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        directions_world = np.ascontiguousarray(self.lidar_dirs_local @ rotation.T, dtype=np.float64).reshape(-1)
        self.lidar_geomid.fill(-1)
        self.lidar_dist.fill(-1.0)
        mujoco.mj_multiRay(self.model, self.data, origin, directions_world, self.lidar_geomgroup, 1, -1, self.lidar_geomid, self.lidar_dist, None, self.lidar_nray, self.cfg.lidar_max_range)
        ranges = self.lidar_dist.copy()
        ranges[ranges < 0.0] = self.cfg.lidar_max_range
        np.clip(ranges, 0.0, self.cfg.lidar_max_range, out=ranges)
        self.lidar_scan = ranges.reshape(self.cfg.lidar_rows, self.cfg.lidar_cols).astype(np.float32)

    def _maybe_scan_lidar(self, force: bool = False) -> None:
        if force or self.step_count % self.lidar_interval_steps == 0:
            self._scan_lidar()

    def _get_obs(self, state: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        state = self._get_robot_state() if state is None else state
        heading_error = float(state["heading_error"])
        obs = np.concatenate([
            self.lidar_scan.reshape(-1).astype(np.float64) / self.cfg.lidar_max_range,
            state["projected_gravity"],
            state["base_ang_vel"] / self.cfg.angular_velocity_scale,
            state["base_lin_vel"] / self.cfg.linear_velocity_scale,
            state["goal_local"] / self.cfg.goal_position_scale,
            np.array([float(state["goal_distance"]) / self.cfg.goal_position_scale]),
            np.array([np.sin(heading_error), np.cos(heading_error)]),
            (state["q"] - self.q_nominal) / np.maximum(self.joint_action_scale, 0.20),
            state["dq"] / self.cfg.joint_velocity_scale,
            state["track_omega"] / self.cfg.track_speed_limit,
            self.filtered_action,
        ])
        if obs.shape != (self.observation_dim,) or not np.isfinite(obs).all():
            raise FloatingPointError(f"Observation is invalid: expected {self.observation_dim}, got {obs.shape}")
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _set_track_surfacevel(self, track_index: int, omega: float) -> None:
        surface_speed = self.cfg.track_effective_radius * float(omega)
        for geom_id in self.track_pad_ids[track_index]:
            self.model.geom_surfacevel[int(geom_id), :] = 0.0
            self.model.geom_surfacevel[int(geom_id), 0] = surface_speed

    def _set_all_track_surfacevel(self, omega: float) -> None:
        for track_index in range(2):
            self._set_track_surfacevel(track_index, omega)

    def _measure_track_load_torque(self, track_index: int) -> float:
        pad_ids = frozenset(int(value) for value in self.track_pad_ids[track_index])
        contact_force = np.zeros(6, dtype=np.float64)
        total_surface_force = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == self.floor_geom_id and g2 in pad_ids:
                pad_id, sign = g2, 1.0
            elif g2 == self.floor_geom_id and g1 in pad_ids:
                pad_id, sign = g1, -1.0
            else:
                continue
            if int(contact.efc_address) < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            force_on_pad_world = sign * (frame.T @ contact_force[:3])
            pad_axis_world = self.data.geom_xmat[pad_id].reshape(3, 3)[:, 0]
            total_surface_force += float(np.dot(force_on_pad_world, pad_axis_world))
        return self.cfg.track_effective_radius * total_surface_force

    def _update_track_motors(self, target_omega: np.ndarray) -> np.ndarray:
        motor_tau = np.zeros(2, dtype=np.float64)
        dt = float(self.model.opt.timestep)
        for track_index in range(2):
            load_tau = self._measure_track_load_torque(track_index)
            omega = float(self.track_omega[track_index])
            tau = self.cfg.track_kd * (float(target_omega[track_index]) - omega)
            tau = float(np.clip(tau, -self.cfg.track_torque_limit, self.cfg.track_torque_limit))
            net_tau = tau + load_tau - self.cfg.track_motor_damping * omega
            omega += (net_tau / self.cfg.track_virtual_inertia) * dt
            omega = float(np.clip(omega, -self.cfg.track_speed_limit, self.cfg.track_speed_limit))
            self.track_angle[track_index] += omega * dt
            self.track_omega[track_index] = omega
            self.track_load_tau[track_index] = load_tau
            self.track_motor_tau[track_index] = tau
            self._set_track_surfacevel(track_index, omega)
            motor_tau[track_index] = tau
        return motor_tau

    def _physics_is_finite(self) -> bool:
        arrays = (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl, self.track_omega, self.track_angle)
        return bool(all(np.isfinite(array).all() for array in arrays))

    def _get_obstacle_collision(self) -> Tuple[bool, Tuple[str, ...]]:
        names = set()
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            obstacle_id = g1 if g1 in self.obstacle_geom_set else g2 if g2 in self.obstacle_geom_set else -1
            if obstacle_id < 0:
                continue
            other = g2 if obstacle_id == g1 else g1
            if other in self.obstacle_geom_set or int(contact.efc_address) < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            if abs(float(contact_force[0])) < self.cfg.collision_force_threshold:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_id)
            names.add(name if name is not None else f"obstacle_geom_{obstacle_id}")
        result = tuple(sorted(names))
        return bool(result), result

    def _align_support_with_ground(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        lowest = np.inf
        for geom_id in self.support_geom_ids:
            geom_id = int(geom_id)
            if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            half_extent_z = float(np.dot(np.abs(rotation[2, :]), self.model.geom_size[geom_id]))
            lowest = min(lowest, float(self.data.geom_xpos[geom_id, 2] - half_extent_z))
        if not np.isfinite(lowest):
            raise RuntimeError("Could not determine the lowest support geom")
        self.data.qpos[self.free_qpos_adr + 2] += self.cfg.initial_ground_clearance - lowest
        mujoco.mj_forward(self.model, self.data)

    def _sample_goal_and_obstacles(self) -> None:
        start = self.data.xpos[self.base_id, :2].copy()
        yaw = self._get_heading_yaw()
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        left = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
        goal_distance = float(self.np_random.uniform(*self.cfg.goal_distance_range))
        goal_lateral = float(self.np_random.uniform(*self.cfg.goal_lateral_range))
        goal_xy = start + forward * goal_distance + left * goal_lateral
        goal_xy = np.clip(goal_xy, -self.cfg.world_xy_limit + 0.25, self.cfg.world_xy_limit - 0.25)
        self.goal_position[:] = (goal_xy[0], goal_xy[1], 0.08)

        segment = goal_xy - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            raise RuntimeError("Sampled goal is too close to the robot")
        direction = segment / segment_length
        perpendicular = np.array([-direction[1], direction[0]], dtype=np.float64)
        fraction_low, fraction_high = self.cfg.obstacle_path_fraction_range
        obstacle_count = len(self.obstacle_geom_ids)
        base_fractions = np.linspace(fraction_low, fraction_high, obstacle_count) if obstacle_count > 1 else np.array([self.np_random.uniform(fraction_low, fraction_high)])
        for index, geom_id in enumerate(self.obstacle_geom_ids):
            fraction = float(np.clip(base_fractions[index] + self.np_random.uniform(-0.04, 0.04), fraction_low, fraction_high)) if obstacle_count > 1 else float(base_fractions[index])
            along = float(np.clip(segment_length * fraction, self.cfg.obstacle_start_clearance, max(self.cfg.obstacle_start_clearance, segment_length - self.cfg.obstacle_goal_clearance)))
            lateral = float(self.np_random.uniform(-self.obstacle_offset_limit, self.obstacle_offset_limit))
            obstacle_xy = start + direction * along + perpendicular * lateral
            self.model.geom_pos[int(geom_id), 0] = obstacle_xy[0]
            self.model.geom_pos[int(geom_id), 1] = obstacle_xy[1]
            self.model.geom_pos[int(geom_id), 2] = self.obstacle_z[index]
        mujoco.mj_forward(self.model, self.data)

    def _compute_reward(self, state: Dict[str, np.ndarray], collision: bool, success: bool, mean_joint_torque_sq: float, mean_track_torque_sq: float, old_filtered_action: np.ndarray, older_filtered_action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.reward_cfg
        goal_distance = float(state["goal_distance"])
        progress = float(np.clip(self.previous_goal_distance - goal_distance, -0.12, 0.12))
        heading_error = float(state["heading_error"])
        heading_reward = 0.5 * (np.cos(heading_error) + 1.0)
        goal_local = state["goal_local"]
        goal_norm = max(float(np.linalg.norm(goal_local)), 1e-6)
        speed_toward_goal = float(np.dot(state["base_lin_vel"][:2], goal_local / goal_norm))
        speed_reward = float(np.clip(speed_toward_goal, -1.0, 1.0))
        lidar_min = float(np.min(self.lidar_scan))
        clearance_penalty = float(max(self.cfg.reward.clearance_distance - lidar_min, 0.0) / self.cfg.reward.clearance_distance) ** 2
        stall_penalty = float(goal_distance > self.cfg.goal_radius and abs(progress) < 5e-4 and abs(speed_toward_goal) < 0.03)
        action_rate_penalty = float(np.mean((self.filtered_action - old_filtered_action) ** 2))
        action_acceleration_penalty = float(np.mean((self.filtered_action - 2.0 * old_filtered_action + older_filtered_action) ** 2))
        joint_pose_penalty = float(np.mean(((state["q"] - self.q_nominal) / np.maximum(self.joint_action_scale, 0.20)) ** 2))
        joint_velocity_penalty = float(np.mean((state["dq"] / self.cfg.joint_velocity_scale) ** 2))
        joint_torque_penalty = float(mean_joint_torque_sq / max(float(np.mean(self.ctrl_upper ** 2)), 1e-6))
        track_torque_penalty = float(mean_track_torque_sq / max(self.cfg.track_torque_limit ** 2, 1e-6))

        reward = (
            cfg.progress_weight * progress
            + cfg.heading_weight * heading_reward
            + cfg.speed_toward_goal_weight * speed_reward
            - cfg.clearance_weight * clearance_penalty
            - cfg.stall_weight * stall_penalty
            - cfg.action_rate_weight * action_rate_penalty
            - cfg.action_acceleration_weight * action_acceleration_penalty
            - cfg.joint_pose_weight * joint_pose_penalty
            - cfg.joint_velocity_weight * joint_velocity_penalty
            - cfg.joint_torque_weight * joint_torque_penalty
            - cfg.track_torque_weight * track_torque_penalty
            - cfg.time_penalty
        )
        if success:
            reward += cfg.success_reward
        if collision:
            reward -= cfg.collision_penalty

        terms = {
            "total": float(reward), "progress": progress, "heading": float(heading_reward), "speed_toward_goal": speed_toward_goal,
            "clearance_penalty": clearance_penalty, "stall_penalty": stall_penalty, "action_rate_penalty": action_rate_penalty,
            "action_acceleration_penalty": action_acceleration_penalty, "joint_pose_penalty": joint_pose_penalty,
            "joint_velocity_penalty": joint_velocity_penalty, "joint_torque_penalty": joint_torque_penalty,
            "track_torque_penalty": track_torque_penalty, "lidar_min": lidar_min, "goal_distance": goal_distance,
            "heading_error": heading_error, "success": float(success), "collision": float(collision),
        }
        return float(reward), terms

    def _check_termination(self, state: Dict[str, np.ndarray], collision: bool) -> Tuple[bool, bool, str]:
        goal_distance = float(state["goal_distance"])
        if goal_distance <= self.cfg.goal_radius:
            return True, True, "goal_reached"
        if collision:
            return True, False, "obstacle_collision"
        base_z = float(state["base_pos"][2])
        if base_z < self.cfg.min_base_height:
            return True, False, "base_too_low"
        if base_z > self.cfg.max_base_height:
            return True, False, "base_too_high"
        if state["projected_gravity"][2] > -np.cos(np.deg2rad(self.cfg.max_tilt_deg)):
            return True, False, "excessive_tilt"
        if np.any(np.abs(state["base_pos"][:2]) > self.cfg.world_xy_limit):
            return True, False, "out_of_bounds"
        return False, False, ""

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        randomize = bool(options.get("randomize", True))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qpos[self.joint_qpos_adr] = self.q_nominal
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        qadr, dadr = self.free_qpos_adr, self.free_dof_adr
        if randomize:
            self.data.qpos[self.joint_qpos_adr] += self.np_random.uniform(-self.cfg.joint_position_noise, self.cfg.joint_position_noise, 4)
            self.data.qvel[self.joint_dof_adr] = self.np_random.uniform(-self.cfg.joint_velocity_noise, self.cfg.joint_velocity_noise, 4)
            self.data.qpos[qadr:qadr + 2] += self.np_random.uniform(-self.cfg.base_position_xy_noise, self.cfg.base_position_xy_noise, 2)
            yaw_noise = float(self.np_random.uniform(-np.deg2rad(self.cfg.initial_yaw_noise_deg), np.deg2rad(self.cfg.initial_yaw_noise_deg)))
            self.data.qpos[qadr + 3:qadr + 7] = quat_multiply(euler_to_quat(0.0, 0.0, yaw_noise), self.data.qpos[qadr + 3:qadr + 7].copy())
            self.data.qvel[dadr:dadr + 3] = self.np_random.uniform(-self.cfg.base_linear_velocity_noise, self.cfg.base_linear_velocity_noise, 3)
            self.data.qvel[dadr + 3:dadr + 6] = self.np_random.uniform(-self.cfg.base_angular_velocity_noise, self.cfg.base_angular_velocity_noise, 3)

        self.track_omega.fill(0.0)
        self.track_angle.fill(0.0)
        self.track_load_tau.fill(0.0)
        self.track_motor_tau.fill(0.0)
        self._set_all_track_surfacevel(0.0)
        self._align_support_with_ground()
        self._sample_goal_and_obstacles()
        self.filtered_action.fill(0.0)
        self.previous_filtered_action.fill(0.0)
        self.older_filtered_action.fill(0.0)
        self.step_count = 0
        self._episode_ended = False
        self.episode_progress = 0.0
        self.path_length = 0.0
        self.last_base_xy = self.data.xpos[self.base_id, :2].copy()
        state = self._get_robot_state()
        self.previous_goal_distance = float(state["goal_distance"])
        self._maybe_scan_lidar(force=True)
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, {}, "", (), np.zeros(4), np.zeros(2), self.q_nominal.copy(), np.full(2, self.cfg.track_speed_center))
        self.render()
        return obs, info

    def _make_info(self, state: Dict[str, np.ndarray], reward_terms: Dict[str, float], termination_reason: str, collision_names: Tuple[str, ...], joint_torque: np.ndarray, track_torque: np.ndarray, q_des: np.ndarray, track_target: np.ndarray) -> Dict[str, object]:
        return {
            "reward_terms": reward_terms, "termination_reason": termination_reason,
            "base_position": state["base_pos"].copy(), "base_linear_velocity": state["base_lin_vel"].copy(), "base_angular_velocity": state["base_ang_vel"].copy(),
            "goal_position": self.goal_position.copy(), "goal_local": state["goal_local"].copy(), "goal_distance": float(state["goal_distance"]),
            "heading_error": float(state["heading_error"]), "lidar_min": float(np.min(self.lidar_scan)),
            "collision_geoms": collision_names, "success": bool(termination_reason == "goal_reached"), "collision": bool(termination_reason == "obstacle_collision"),
            "episode_progress": float(self.episode_progress), "path_length": float(self.path_length),
            "joint_torque": joint_torque.copy(), "track_torque": track_torque.copy(), "q_des": q_des.copy(),
            "track_target_omega": track_target.copy(), "track_omega": self.track_omega.copy(), "filtered_action": self.filtered_action.copy(),
            "obstacle_positions": self.model.geom_pos[self.obstacle_geom_ids, :3].copy(),
        }

    def step(self, action: np.ndarray):
        if self._episode_ended:
            raise RuntimeError("step() called after episode end; call reset() first")
        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (self.action_dim,) or not np.isfinite(raw_action).all():
            raise ValueError(f"action must be finite with shape ({self.action_dim},)")
        raw_action = np.clip(raw_action, -1.0, 1.0)

        old_filtered_action = self.filtered_action.copy()
        older_filtered_action = self.previous_filtered_action.copy()
        filter_delta = self.cfg.action_filter_alpha * (raw_action - self.filtered_action)
        max_filter_delta = self.cfg.action_rate_limit * self.policy_dt
        self.filtered_action += np.clip(filter_delta, -max_filter_delta, max_filter_delta)

        margin = self.cfg.joint_limit_margin
        q_des_start = np.clip(self.q_nominal + self.joint_action_scale * old_filtered_action[:4], self.joint_lower + margin, self.joint_upper - margin)
        q_des_target = np.clip(self.q_nominal + self.joint_action_scale * self.filtered_action[:4], self.joint_lower + margin, self.joint_upper - margin)
        track_target = self.cfg.track_speed_center + self.cfg.track_speed_scale * self.filtered_action[4:6]
        track_target = np.clip(track_target, -self.cfg.track_speed_limit, self.cfg.track_speed_limit)

        joint_torque = np.zeros(4, dtype=np.float64)
        track_torque = np.zeros(2, dtype=np.float64)
        joint_torque_sq_sum = 0.0
        track_torque_sq_sum = 0.0
        invalid_state = False
        executed_steps = 0
        q_des = q_des_start.copy()

        for substep in range(self.frame_skip):
            interpolation = float(substep + 1) / self.frame_skip
            q_des = q_des_start + interpolation * (q_des_target - q_des_start)
            q = self.data.qpos[self.joint_qpos_adr]
            dq = self.data.qvel[self.joint_dof_adr]
            joint_torque = np.clip(self.kp * (q_des - q) - self.kd * dq, self.ctrl_lower, self.ctrl_upper)
            self.data.ctrl[self.actuator_ids] = joint_torque
            track_torque = self._update_track_motors(track_target)
            joint_torque_sq_sum += float(np.mean(joint_torque ** 2))
            track_torque_sq_sum += float(np.mean(track_torque ** 2))
            mujoco.mj_step(self.model, self.data)
            executed_steps += 1
            if not self._physics_is_finite():
                invalid_state = True
                break

        self.step_count += 1
        if invalid_state:
            self._episode_ended = True
            reward = -self.reward_cfg.invalid_physics_penalty
            info = {"termination_reason": "invalid_physics", "reward_terms": {"total": reward, "invalid_physics": 1.0}}
            return self._last_valid_obs.copy(), reward, True, False, info

        self._maybe_scan_lidar()
        state = self._get_robot_state()
        current_xy = state["base_pos"][:2]
        self.path_length += float(np.linalg.norm(current_xy - self.last_base_xy))
        self.last_base_xy = current_xy.copy()
        current_goal_distance = float(state["goal_distance"])
        self.episode_progress += self.previous_goal_distance - current_goal_distance
        collision, collision_names = self._get_obstacle_collision()
        terminated, success, termination_reason = self._check_termination(state, collision)
        mean_joint_torque_sq = joint_torque_sq_sum / max(executed_steps, 1)
        mean_track_torque_sq = track_torque_sq_sum / max(executed_steps, 1)
        reward, reward_terms = self._compute_reward(state, collision, success, mean_joint_torque_sq, mean_track_torque_sq, old_filtered_action, older_filtered_action)
        if terminated and not success and not collision:
            reward -= self.reward_cfg.termination_penalty
            reward_terms["termination_penalty"] = self.reward_cfg.termination_penalty
            reward_terms["total"] = float(reward)

        truncated = bool(not terminated and self.step_count >= self.cfg.max_episode_steps)
        self._episode_ended = bool(terminated or truncated)
        self.older_filtered_action[:] = older_filtered_action
        self.previous_filtered_action[:] = old_filtered_action
        self.previous_goal_distance = current_goal_distance
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, reward_terms, termination_reason, collision_names, joint_torque, track_torque, q_des_target, track_target)
        self.render()
        return obs, float(reward), terminated, truncated, info

    def _draw_debug(self) -> None:
        if self.viewer is None or not self.viewer.is_running():
            return
        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            if scene.maxgeom < 2:
                return
            target = scene.geoms[0]
            mujoco.mjv_initGeom(target, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.10, 0.10, 0.10]), self.goal_position, np.eye(3).reshape(-1), np.array([0.1, 1.0, 0.1, 0.9], dtype=np.float32))
            scene.ngeom = 1
            base = self.data.xpos[self.base_id].copy()
            start = base + np.array([0.0, 0.0, 0.25])
            delta = self.goal_position - start
            delta[2] = 0.0
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                end = start + 0.55 * delta / distance
                arrow = scene.geoms[1]
                mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, start, end)
                arrow.rgba[:] = np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)
                scene.ngeom = 2

    def render(self):
        if self.viewer is not None and self.viewer.is_running():
            self._draw_debug()
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

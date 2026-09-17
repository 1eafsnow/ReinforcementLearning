from pathlib import Path
import tempfile
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


class SnakeTraverseEnv(gym.Env):
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
        self.front_track_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, config.front_track_body_name)
        self.back_track_body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, config.back_track_body_name)
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
        self.obstacle_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, config.obstacle_geom_name)
        self.hfield_id = self._require_id(mujoco.mjtObj.mjOBJ_HFIELD, config.obstacle_hfield_name)
        self.hfield_adr = int(self.model.hfield_adr[self.hfield_id])
        self.hfield_nrow = int(self.model.hfield_nrow[self.hfield_id])
        self.hfield_ncol = int(self.model.hfield_ncol[self.hfield_id])
        self.lidar_site_id = self._require_id(mujoco.mjtObj.mjOBJ_SITE, config.lidar_site_name)
        self.front_track_pad_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.front_track_pad_names], dtype=np.int32)
        self.back_track_pad_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.back_track_pad_names], dtype=np.int32)
        self.support_geom_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.support_geom_names], dtype=np.int32)

        self.model.geom_group[self.floor_geom_id] = int(config.floor_lidar_group)
        self.model.geom_group[self.obstacle_geom_id] = int(config.lidar_group)
        self.model.flg_surfacevel = 1

        self.action_dim = 6
        self.q_nominal = np.asarray(config.q_nominal, dtype=np.float64)
        self.kp = np.asarray(config.kp, dtype=np.float64)
        self.kd = np.asarray(config.kd, dtype=np.float64)
        self.joint_action_scale = np.asarray(config.joint_action_scale, dtype=np.float64)
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
        self.previous_course_progress = 0.0
        self.episode_progress = 0.0
        self.path_length = 0.0
        self.last_base_xy = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self._episode_ended = False
        self.obstacle_height_limit = float(config.obstacle_height_limit)
        self.obstacle_height = float(config.obstacle_min_height)
        self.obstacle_ramp_length = 0.0
        self.obstacle_platform_length = 0.0
        self.obstacle_start_progress = 0.0
        self.obstacle_top_start_progress = 0.0
        self.obstacle_top_end_progress = 0.0
        self.obstacle_end_progress = 0.0
        self.course_origin_xy = np.zeros(2, dtype=np.float64)
        self.course_forward = np.array([-1.0, 0.0], dtype=np.float64)
        self.course_left = np.array([0.0, -1.0], dtype=np.float64)
        self.stage_front_track = False
        self.stage_base = False
        self.stage_back_track = False
        self.obstacle_cleared = False

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
            raise RuntimeError(f"SnakeTraverseEnv requires MuJoCo >= 3.11.0; installed version is {mujoco.__version__}")

    def _validate_config(self) -> None:
        if self.cfg.frame_skip < 1 or self.cfg.max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        if len(self.cfg.joint_names) != 4 or len(self.cfg.actuator_names) != 4:
            raise ValueError("Snake slider model requires exactly four joints and four joint actuators")
        if len(self.cfg.front_track_pad_names) != 3 or len(self.cfg.back_track_pad_names) != 3:
            raise ValueError("Each virtual track requires exactly three surfacevel pads")
        if self.cfg.hfield_rows < 2 or self.cfg.hfield_cols < 4:
            raise ValueError("Heightfield resolution is too small")
        if self.cfg.obstacle_min_height <= 0.0 or self.cfg.obstacle_height_limit > self.cfg.obstacle_max_supported_height:
            raise ValueError("Obstacle height range is invalid")
        if self.cfg.hfield_base_z >= 0.0:
            raise ValueError("hfield_base_z must be negative so unused terrain stays below the floor")
        for value, name in ((self.cfg.obstacle_distance_range, "obstacle_distance_range"), (self.cfg.obstacle_ramp_length_range, "obstacle_ramp_length_range"), (self.cfg.obstacle_platform_length_range, "obstacle_platform_length_range"), (self.cfg.goal_after_obstacle_range, "goal_after_obstacle_range"), (self.cfg.goal_lateral_range, "goal_lateral_range")):
            self._validate_range(value, name)

    @staticmethod
    def _validate_range(value: Tuple[float, float], name: str) -> Tuple[float, float]:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
            raise ValueError(f"{name} must be a finite (low, high) pair")
        return float(array[0]), float(array[1])

    def _load_model(self, xml_path: Path) -> mujoco.MjModel:
        xml_path = Path(xml_path).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        xml = xml_path.read_text(encoding="utf-8")
        elevation_z = -self.cfg.hfield_base_z + self.cfg.obstacle_max_supported_height
        hfield_xml = f'\n    <hfield name="{self.cfg.obstacle_hfield_name}" nrow="{self.cfg.hfield_rows}" ncol="{self.cfg.hfield_cols}" size="{self.cfg.hfield_radius_x} {self.cfg.hfield_radius_y} {elevation_z} {self.cfg.hfield_base_depth}"/>\n'
        obstacle_xml = f'\n    <geom name="{self.cfg.obstacle_geom_name}" type="hfield" hfield="{self.cfg.obstacle_hfield_name}" pos="{self.cfg.hfield_center_x} {self.cfg.hfield_center_y} {self.cfg.hfield_base_z}" rgba="0.72 0.75 0.86 1" friction="1.3 0.005 0.0001" condim="3" contype="3" conaffinity="3" group="{self.cfg.lidar_group}"/>\n'
        if "</asset>" not in xml or "</worldbody>" not in xml:
            raise RuntimeError("scene_slider.xml must contain asset and worldbody sections")
        xml = xml.replace("</asset>", hfield_xml + "  </asset>", 1)
        xml = xml.replace("</worldbody>", obstacle_xml + "  </worldbody>", 1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", prefix=".snake_training_", dir=xml_path.parent, encoding="utf-8", delete=False) as file:
            file.write(xml)
            generated_path = Path(file.name)
        try:
            model = mujoco.MjModel.from_xml_path(str(generated_path))
        finally:
            generated_path.unlink(missing_ok=True)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        return model

    def _require_id(self, object_type, name: str) -> int:
        object_id = int(mujoco.mj_name2id(self.model, object_type, name))
        if object_id < 0:
            raise RuntimeError(f"MuJoCo object not found: {name}")
        return object_id

    def _build_lidar_directions(self) -> np.ndarray:
        h_step = np.deg2rad(self.cfg.lidar_hfov_deg / self.cfg.lidar_cols)
        v_step = np.deg2rad(self.cfg.lidar_vfov_deg / self.cfg.lidar_rows)
        azimuth = (np.arange(self.cfg.lidar_cols, dtype=np.float64) - (self.cfg.lidar_cols - 1) * 0.5) * h_step
        elevation = (np.arange(self.cfg.lidar_rows, dtype=np.float64) - (self.cfg.lidar_rows - 1) * 0.5) * v_step
        az, el = np.meshgrid(azimuth, elevation)
        directions = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1)
        return np.ascontiguousarray(directions.reshape(-1, 3), dtype=np.float64)

    def set_obstacle_height_limit(self, max_height: float) -> None:
        if not np.isfinite(max_height):
            raise ValueError("max_height must be finite")
        self.obstacle_height_limit = float(np.clip(max_height, self.cfg.obstacle_min_height, self.cfg.obstacle_max_supported_height))

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

    def _course_progress(self, xy: np.ndarray) -> float:
        return float(np.dot(np.asarray(xy, dtype=np.float64) - self.course_origin_xy, self.course_forward))

    def _course_lateral(self, xy: np.ndarray) -> float:
        return float(np.dot(np.asarray(xy, dtype=np.float64) - self.course_origin_xy, self.course_left))

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        base_rotation = self.data.xmat[self.base_id].reshape(3, 3)
        base_ang_vel, base_lin_vel = self._get_base_velocity()
        goal_local, goal_distance, heading_error = self._get_goal_state()
        base_xy = self.data.xpos[self.base_id, :2]
        return {"base_pos": self.data.xpos[self.base_id].copy(), "projected_gravity": base_rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64), "base_ang_vel": base_ang_vel, "base_lin_vel": base_lin_vel, "q": self.data.qpos[self.joint_qpos_adr].copy(), "dq": self.data.qvel[self.joint_dof_adr].copy(), "goal_local": goal_local, "goal_distance": np.array(goal_distance), "heading_error": np.array(heading_error), "track_omega": self.track_omega.copy(), "course_progress": np.array(self._course_progress(base_xy)), "course_lateral": np.array(self._course_lateral(base_xy))}

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
        obs = np.concatenate([self.lidar_scan.reshape(-1).astype(np.float64) / self.cfg.lidar_max_range, state["projected_gravity"], state["base_ang_vel"] / self.cfg.angular_velocity_scale, state["base_lin_vel"] / self.cfg.linear_velocity_scale, state["goal_local"] / self.cfg.goal_position_scale, np.array([float(state["goal_distance"]) / self.cfg.goal_position_scale]), np.array([np.sin(heading_error), np.cos(heading_error)]), (state["q"] - self.q_nominal) / np.maximum(self.joint_action_scale, 0.20), state["dq"] / self.cfg.joint_velocity_scale, state["track_omega"] / self.cfg.track_speed_limit, self.filtered_action])
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
        allowed_surfaces = (self.floor_geom_id, self.obstacle_geom_id)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in allowed_surfaces and g2 in pad_ids:
                pad_id, sign = g2, 1.0
            elif g2 in allowed_surfaces and g1 in pad_ids:
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

    def _write_trapezoid_hfield(self, height: float, start_progress: float, ramp_length: float, platform_length: float) -> None:
        elevation_z = float(self.model.hfield_size[self.hfield_id, 2])
        x_world = self.cfg.hfield_center_x + np.linspace(-self.cfg.hfield_radius_x, self.cfg.hfield_radius_x, self.hfield_ncol)
        progress = self.course_origin_xy[0] - x_world
        end_up = start_progress + ramp_length
        end_top = end_up + platform_length
        end_down = end_top + ramp_length
        physical_height = np.zeros(self.hfield_ncol, dtype=np.float64)
        inside_up = (progress >= start_progress) & (progress < end_up)
        inside_top = (progress >= end_up) & (progress <= end_top)
        inside_down = (progress > end_top) & (progress <= end_down)
        physical_height[inside_up] = height * (progress[inside_up] - start_progress) / ramp_length
        physical_height[inside_top] = height
        physical_height[inside_down] = height * (end_down - progress[inside_down]) / ramp_length
        active = inside_up | inside_top | inside_down
        normalized = np.zeros(self.hfield_ncol, dtype=np.float64)
        normalized[active] = (physical_height[active] - self.cfg.hfield_base_z) / elevation_z
        np.clip(normalized, 0.0, 1.0, out=normalized)
        count = self.hfield_nrow * self.hfield_ncol
        hfield = self.model.hfield_data[self.hfield_adr:self.hfield_adr + count].reshape(self.hfield_nrow, self.hfield_ncol)
        hfield[:] = normalized[None, :]
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.update_hfield(self.hfield_id)

    def _generate_course(self, options: Dict[str, object]) -> None:
        self.course_origin_xy = self.data.xpos[self.base_id, :2].copy()
        self.course_forward[:] = (-1.0, 0.0)
        self.course_left[:] = (0.0, -1.0)
        requested_height = options.get("obstacle_height")
        if requested_height is None:
            low = self.cfg.obstacle_min_height
            high = max(low, self.obstacle_height_limit)
            self.obstacle_height = float(self.np_random.uniform(low, high))
        else:
            self.obstacle_height = float(np.clip(float(requested_height), self.cfg.obstacle_min_height, self.cfg.obstacle_max_supported_height))
        self.obstacle_ramp_length = float(self.np_random.uniform(*self.cfg.obstacle_ramp_length_range))
        self.obstacle_platform_length = float(self.np_random.uniform(*self.cfg.obstacle_platform_length_range))
        self.obstacle_start_progress = float(self.np_random.uniform(*self.cfg.obstacle_distance_range))
        self.obstacle_top_start_progress = self.obstacle_start_progress + self.obstacle_ramp_length
        self.obstacle_top_end_progress = self.obstacle_top_start_progress + self.obstacle_platform_length
        self.obstacle_end_progress = self.obstacle_top_end_progress + self.obstacle_ramp_length
        self._write_trapezoid_hfield(self.obstacle_height, self.obstacle_start_progress, self.obstacle_ramp_length, self.obstacle_platform_length)
        goal_after = float(self.np_random.uniform(*self.cfg.goal_after_obstacle_range))
        goal_lateral = float(self.np_random.uniform(*self.cfg.goal_lateral_range))
        goal_progress = self.obstacle_end_progress + goal_after
        self.goal_position[:] = (self.course_origin_xy[0] - goal_progress, self.course_origin_xy[1] - goal_lateral, 0.08)
        mujoco.mj_forward(self.model, self.data)

    def _update_stage_flags(self) -> Dict[str, float]:
        rewards = {"front_stage": 0.0, "base_stage": 0.0, "rear_stage": 0.0, "clear_stage": 0.0}
        front_progress = self._course_progress(self.data.xpos[self.front_track_body_id, :2])
        base_progress = self._course_progress(self.data.xpos[self.base_id, :2])
        rear_progress = self._course_progress(self.data.xpos[self.back_track_body_id, :2])
        if not self.stage_front_track and front_progress >= self.obstacle_top_start_progress:
            self.stage_front_track = True
            rewards["front_stage"] = self.reward_cfg.front_track_stage_reward
        if not self.stage_base and base_progress >= self.obstacle_top_start_progress:
            self.stage_base = True
            rewards["base_stage"] = self.reward_cfg.base_stage_reward
        if not self.stage_back_track and rear_progress >= self.obstacle_top_start_progress:
            self.stage_back_track = True
            rewards["rear_stage"] = self.reward_cfg.rear_track_stage_reward
        if not self.obstacle_cleared and rear_progress >= self.obstacle_end_progress + self.cfg.obstacle_clear_margin:
            self.obstacle_cleared = True
            rewards["clear_stage"] = self.reward_cfg.obstacle_clear_reward
        return rewards

    def _get_obstacle_contact_count(self) -> int:
        count = 0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if int(contact.geom1) == self.obstacle_geom_id or int(contact.geom2) == self.obstacle_geom_id:
                count += 1
        return count

    def _compute_reward(self, state: Dict[str, np.ndarray], stage_rewards: Dict[str, float], mean_joint_torque_sq: float, mean_track_torque_sq: float, old_filtered_action: np.ndarray, older_filtered_action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        cfg = self.reward_cfg
        course_progress = float(state["course_progress"])
        progress_delta = float(np.clip(course_progress - self.previous_course_progress, -0.12, 0.12))
        heading_error = float(state["heading_error"])
        heading_reward = 0.5 * (np.cos(heading_error) + 1.0)
        forward_speed = float(np.clip(state["base_lin_vel"][0], -1.0, 1.0))
        reverse_penalty = max(-progress_delta, 0.0)
        stall_penalty = float(abs(progress_delta) < 4e-4 and abs(forward_speed) < 0.03)
        lateral_penalty = float(max(abs(float(state["course_lateral"])) - 0.10, 0.0) ** 2)
        action_rate_penalty = float(np.mean((self.filtered_action - old_filtered_action) ** 2))
        action_acceleration_penalty = float(np.mean((self.filtered_action - 2.0 * old_filtered_action + older_filtered_action) ** 2))
        joint_pose_penalty = float(np.mean(((state["q"] - self.q_nominal) / np.maximum(self.joint_action_scale, 0.20)) ** 2))
        joint_velocity_penalty = float(np.mean((state["dq"] / self.cfg.joint_velocity_scale) ** 2))
        joint_torque_penalty = float(mean_joint_torque_sq / max(float(np.mean(self.ctrl_upper ** 2)), 1e-6))
        track_torque_penalty = float(mean_track_torque_sq / max(self.cfg.track_torque_limit ** 2, 1e-6))
        reward = cfg.progress_weight * progress_delta + cfg.heading_weight * heading_reward + cfg.forward_speed_weight * forward_speed + stage_rewards["front_stage"] + stage_rewards["base_stage"] + stage_rewards["rear_stage"] + stage_rewards["clear_stage"] - cfg.stall_weight * stall_penalty - cfg.reverse_weight * reverse_penalty - cfg.lateral_drift_weight * lateral_penalty - cfg.action_rate_weight * action_rate_penalty - cfg.action_acceleration_weight * action_acceleration_penalty - cfg.joint_pose_weight * joint_pose_penalty - cfg.joint_velocity_weight * joint_velocity_penalty - cfg.joint_torque_weight * joint_torque_penalty - cfg.track_torque_weight * track_torque_penalty - cfg.time_penalty
        terms = {"total": float(reward), "progress": progress_delta, "heading": float(heading_reward), "forward_speed": forward_speed, "front_stage_reward": stage_rewards["front_stage"], "base_stage_reward": stage_rewards["base_stage"], "rear_stage_reward": stage_rewards["rear_stage"], "clear_stage_reward": stage_rewards["clear_stage"], "stall_penalty": stall_penalty, "reverse_penalty": reverse_penalty, "lateral_penalty": lateral_penalty, "action_rate_penalty": action_rate_penalty, "action_acceleration_penalty": action_acceleration_penalty, "joint_pose_penalty": joint_pose_penalty, "joint_velocity_penalty": joint_velocity_penalty, "joint_torque_penalty": joint_torque_penalty, "track_torque_penalty": track_torque_penalty, "goal_distance": float(state["goal_distance"]), "obstacle_height": self.obstacle_height, "obstacle_contact_count": float(self._get_obstacle_contact_count())}
        return float(reward), terms

    def _check_termination(self, state: Dict[str, np.ndarray]) -> Tuple[bool, bool, str]:
        goal_distance = float(state["goal_distance"])
        if self.obstacle_cleared and goal_distance <= self.cfg.goal_radius:
            return True, True, "goal_reached_after_obstacle"
        base_z = float(state["base_pos"][2])
        if base_z < self.cfg.min_base_height:
            return True, False, "base_too_low"
        if base_z > self.cfg.max_base_height:
            return True, False, "base_too_high"
        if state["projected_gravity"][2] > -np.cos(np.deg2rad(self.cfg.max_tilt_deg)):
            return True, False, "excessive_tilt"
        if np.any(np.abs(state["base_pos"][:2]) > self.cfg.world_xy_limit):
            return True, False, "out_of_bounds"
        if abs(float(state["course_lateral"])) > self.cfg.max_lateral_deviation:
            return True, False, "left_training_lane"
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
        self._generate_course(options)
        self.filtered_action.fill(0.0)
        self.previous_filtered_action.fill(0.0)
        self.older_filtered_action.fill(0.0)
        self.step_count = 0
        self._episode_ended = False
        self.episode_progress = 0.0
        self.path_length = 0.0
        self.last_base_xy = self.data.xpos[self.base_id, :2].copy()
        self.stage_front_track = False
        self.stage_base = False
        self.stage_back_track = False
        self.obstacle_cleared = False
        state = self._get_robot_state()
        self.previous_goal_distance = float(state["goal_distance"])
        self.previous_course_progress = float(state["course_progress"])
        self._maybe_scan_lidar(force=True)
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, {}, "", np.zeros(4), np.zeros(2), self.q_nominal.copy(), np.full(2, self.cfg.track_speed_center))
        self.render()
        return obs, info

    def _make_info(self, state: Dict[str, np.ndarray], reward_terms: Dict[str, float], termination_reason: str, joint_torque: np.ndarray, track_torque: np.ndarray, q_des: np.ndarray, track_target: np.ndarray) -> Dict[str, object]:
        return {"reward_terms": reward_terms, "termination_reason": termination_reason, "base_position": state["base_pos"].copy(), "base_linear_velocity": state["base_lin_vel"].copy(), "base_angular_velocity": state["base_ang_vel"].copy(), "goal_position": self.goal_position.copy(), "goal_local": state["goal_local"].copy(), "goal_distance": float(state["goal_distance"]), "heading_error": float(state["heading_error"]), "lidar_min": float(np.min(self.lidar_scan)), "success": bool(termination_reason == "goal_reached_after_obstacle"), "episode_progress": float(self.episode_progress), "path_length": float(self.path_length), "course_progress": float(state["course_progress"]), "course_lateral": float(state["course_lateral"]), "obstacle_height": self.obstacle_height, "obstacle_ramp_length": self.obstacle_ramp_length, "obstacle_platform_length": self.obstacle_platform_length, "obstacle_start_progress": self.obstacle_start_progress, "obstacle_end_progress": self.obstacle_end_progress, "front_track_stage": self.stage_front_track, "base_stage": self.stage_base, "back_track_stage": self.stage_back_track, "obstacle_cleared": self.obstacle_cleared, "obstacle_contact_count": self._get_obstacle_contact_count(), "joint_torque": joint_torque.copy(), "track_torque": track_torque.copy(), "q_des": q_des.copy(), "track_target_omega": track_target.copy(), "track_omega": self.track_omega.copy(), "filtered_action": self.filtered_action.copy()}

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
        stage_rewards = self._update_stage_flags()
        mean_joint_torque_sq = joint_torque_sq_sum / max(executed_steps, 1)
        mean_track_torque_sq = track_torque_sq_sum / max(executed_steps, 1)
        reward, reward_terms = self._compute_reward(state, stage_rewards, mean_joint_torque_sq, mean_track_torque_sq, old_filtered_action, older_filtered_action)
        terminated, success, termination_reason = self._check_termination(state)
        if success:
            reward += self.reward_cfg.success_reward
            reward_terms["success_reward"] = self.reward_cfg.success_reward
        elif terminated:
            reward -= self.reward_cfg.termination_penalty
            reward_terms["termination_penalty"] = self.reward_cfg.termination_penalty
        reward_terms["total"] = float(reward)
        truncated = bool(not terminated and self.step_count >= self.cfg.max_episode_steps)
        self._episode_ended = bool(terminated or truncated)
        self.older_filtered_action[:] = older_filtered_action
        self.previous_filtered_action[:] = old_filtered_action
        self.previous_goal_distance = current_goal_distance
        self.previous_course_progress = float(state["course_progress"])
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, reward_terms, termination_reason, joint_torque, track_torque, q_des_target, track_target)
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


SnakeAvoidEnv = SnakeTraverseEnv

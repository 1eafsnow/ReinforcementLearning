from pathlib import Path
from typing import Dict, Optional, Tuple
from xml.etree import ElementTree as ET

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
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    return q / norm


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class HexapodWalkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, config: EnvConfig = ENV_CONFIG, render_mode: Optional[str] = None):
        super().__init__()
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.cfg = config
        self.reward_cfg = config.reward
        self._validate_config()
        self.model = self._load_training_model(config.xml_path, config.mesh_dir)
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
        self.foot_site_ids = np.array([self._require_id(mujoco.mjtObj.mjOBJ_SITE, name) for name in config.foot_site_names], dtype=np.int32)
        self.foot_geom_id_array = np.array([self._require_id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in config.foot_geom_names], dtype=np.int32)
        foot_geom_types = self.model.geom_type[self.foot_geom_id_array]
        if np.any(foot_geom_types != mujoco.mjtGeom.mjGEOM_SPHERE):
            raise RuntimeError("All foot contact geoms must be spheres")
        self.foot_radii = self.model.geom_size[self.foot_geom_id_array, 0].astype(np.float64, copy=True)
        if not np.isfinite(self.foot_radii).all() or np.any(self.foot_radii <= 0.0):
            raise RuntimeError(f"Invalid foot geom radii: {self.foot_radii}")
        self.foot_geom_ids = frozenset(int(value) for value in self.foot_geom_id_array)
        self.foot_geom_to_index = {int(geom_id): index for index, geom_id in enumerate(self.foot_geom_id_array)}
        self.floor_geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, config.floor_name)

        self.action_dim = len(config.joint_names)
        self.q_nominal = np.asarray(config.q_nominal, dtype=np.float64).copy()
        self.kp = self._expand_per_leg(config.kp_per_leg, "kp_per_leg")
        self.kd = self._expand_per_leg(config.kd_per_leg, "kd_per_leg")
        self.action_scale = self._expand_per_leg(config.action_scale_per_leg, "action_scale_per_leg")
        armature = self._expand_per_leg(config.armature_per_leg, "armature_per_leg")
        damping = self._expand_per_leg(config.damping_per_leg, "damping_per_leg")
        self.model.dof_armature[self.joint_dof_adr] = armature
        self.model.dof_damping[self.joint_dof_adr] = damping
        if self.q_nominal.shape != (self.action_dim,) or not np.isfinite(self.q_nominal).all():
            raise ValueError(f"q_nominal must be finite with shape ({self.action_dim},)")
        if np.any(self.q_nominal <= self.joint_lower) or np.any(self.q_nominal >= self.joint_upper):
            raise ValueError("q_nominal must lie strictly inside joint limits")
        if np.any(self.kp <= 0.0) or np.any(self.kd < 0.0) or np.any(self.action_scale <= 0.0):
            raise ValueError("kp/action_scale must be positive and kd must be non-negative")

        self.init_qpos = self.model.qpos0.copy()
        self.max_episode_steps = int(config.max_episode_steps)
        self.command_speed_range = self._validate_range(config.command_speed_range, "command_speed_range")
        self.command_direction_range = self._validate_range(config.command_direction_range, "command_direction_range")
        self.command_yaw_offset_range = self._validate_range(config.command_yaw_offset_range, "command_yaw_offset_range")
        self.command_pitch_range = self._validate_range(config.command_pitch_range, "command_pitch_range")
        self.command_x = self.command_y = 0.0
        self.command_speed = self.command_direction = 0.0
        self.command_yaw = self.command_pitch = 0.0
        self.requested_command_yaw = self.requested_command_pitch = 0.0
        self.episode_command_progress = 0.0
        self.next_command_resample_step = 0
        self.phase = 0.0
        self.gait_frequency = float(config.gait_frequency_range[0])
        self.gait_phase_gate = 0.0
        self.step_count = 0
        self._episode_ended = False
        self.filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.previous_filtered_action = np.zeros(self.action_dim, dtype=np.float64)
        self.previous_dq = np.zeros(self.action_dim, dtype=np.float64)
        self.observation_dim = 15 + 3 * self.action_dim + len(config.foot_site_names) + 2
        self._last_valid_obs = np.zeros(self.observation_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_dim,), dtype=np.float32)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if render_mode == "human" else None
        self.forbidden_contact_count = 0

    def _validate_config(self) -> None:
        if self.cfg.frame_skip < 1 or self.cfg.max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        if not 0.0 < self.cfg.action_filter_alpha <= 1.0:
            raise ValueError("action_filter_alpha must lie in (0, 1]")
        if not np.isfinite(self.cfg.action_rate_limit) or self.cfg.action_rate_limit <= 0.0:
            raise ValueError("action_rate_limit must be finite and positive")
        if len(self.cfg.joint_names) != 24 or len(self.cfg.actuator_names) != 24:
            raise ValueError("This model requires exactly 24 joints and 24 actuators")
        if len(self.cfg.foot_site_names) != 6 or len(self.cfg.foot_geom_names) != 6:
            raise ValueError("This model requires exactly six feet")
        if self.cfg.command_velocity_scale <= 0.0 or self.cfg.command_pitch_scale <= 0.0:
            raise ValueError("command_velocity_scale and command_pitch_scale must be positive")
        if self.cfg.command_yaw_slew_rate < 0.0 or self.cfg.command_pitch_slew_rate < 0.0:
            raise ValueError("Command slew rates must be non-negative")
        probabilities = np.array([
            self.cfg.standing_command_probability,
            self.cfg.turn_in_place_probability,
            self.cfg.pitch_in_place_probability,
        ], dtype=np.float64)
        if np.any(probabilities < 0.0) or float(np.sum(probabilities)) > 1.0:
            raise ValueError("Command probabilities must be non-negative and sum to at most one")
        pitch_range = self._validate_range(self.cfg.command_pitch_range, "command_pitch_range")
        pitch_limit = self._validate_range(self.cfg.command_pitch_limit, "command_pitch_limit")
        resample_range = self._validate_range(self.cfg.command_resample_time_range, "command_resample_time_range")
        speed_range = self._validate_range(self.cfg.command_speed_range, "command_speed_range")
        gait_frequency_range = self._validate_range(self.cfg.gait_frequency_range, "gait_frequency_range")
        gait_frequency_speed_range = self._validate_range(self.cfg.gait_frequency_speed_range, "gait_frequency_speed_range")
        swing_clearance_speed_range = self._validate_range(self.reward_cfg.swing_clearance_speed_range, "swing_clearance_speed_range")
        self._validate_range(self.cfg.command_direction_range, "command_direction_range")
        self._validate_range(self.cfg.command_yaw_offset_range, "command_yaw_offset_range")
        if pitch_range[0] < pitch_limit[0] or pitch_range[1] > pitch_limit[1]:
            raise ValueError("command_pitch_range must lie inside command_pitch_limit")
        if speed_range[0] < 0.0 or gait_frequency_speed_range[0] < 0.0:
            raise ValueError("Command and gait-frequency speeds must be non-negative")
        if resample_range[0] <= 0.0 or gait_frequency_range[0] <= 0.0:
            raise ValueError("Resample times and gait frequencies must be positive")
        if gait_frequency_speed_range[0] == gait_frequency_speed_range[1]:
            raise ValueError("gait_frequency_speed_range must have non-zero width")
        if swing_clearance_speed_range[0] < 0.0 or self.reward_cfg.swing_clearance_min < 0.0 or self.reward_cfg.swing_clearance_max < self.reward_cfg.swing_clearance_min:
            raise ValueError("Swing-clearance speeds and heights must be non-negative and ordered")
        if not 0.0 <= self.reward_cfg.swing_contact_gate_start < self.reward_cfg.swing_contact_gate_end <= 1.0:
            raise ValueError("Swing-contact phase gates must satisfy 0 <= start < end <= 1")
        reward_scales = np.array([
            self.reward_cfg.velocity_sigma, self.reward_cfg.standing_velocity_sigma,
            self.reward_cfg.standing_angular_velocity_sigma, self.reward_cfg.yaw_pose_sigma,
            self.reward_cfg.pitch_pose_sigma, self.reward_cfg.roll_pose_sigma,
            self.reward_cfg.yaw_rate_sigma, self.reward_cfg.pitch_rate_sigma,
            self.reward_cfg.yaw_motion_gate_angle, self.reward_cfg.pitch_motion_gate_angle,
            self.reward_cfg.height_sigma, self.reward_cfg.clearance_sigma, self.reward_cfg.clearance_deficit_scale,
            self.reward_cfg.vertical_velocity_scale, self.reward_cfg.slip_velocity_scale,
            self.reward_cfg.joint_velocity_scale, self.reward_cfg.joint_acceleration_scale,
            self.reward_cfg.power_scale,
        ], dtype=np.float64)
        if not np.isfinite(reward_scales).all() or np.any(reward_scales <= 0.0):
            raise ValueError("Reward sigma and scale values must be finite and positive")
        if self.reward_cfg.standing_pose_tolerance <= 0.0 or self.reward_cfg.yaw_error_to_rate_gain < 0.0 or self.reward_cfg.pitch_error_to_rate_gain < 0.0:
            raise ValueError("Pose tolerances must be positive and pose rate gains must be non-negative")
        if self.reward_cfg.max_target_yaw_rate < 0.0 or self.reward_cfg.max_target_pitch_rate < 0.0:
            raise ValueError("Maximum target pose rates must be non-negative")

    def _load_training_model(self, xml_path: Path, mesh_dir: Path) -> mujoco.MjModel:
        xml_path = Path(xml_path).resolve()
        mesh_dir = Path(mesh_dir).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        if not mesh_dir.is_dir():
            raise FileNotFoundError(mesh_dir)
        root = ET.parse(xml_path).getroot()
        # 带include的场景必须保留源文件目录，否则MuJoCo无法解析相对引用。
        if root.find(".//include") is not None:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
        else:
            compiler = root.find("compiler")
            if compiler is None:
                compiler = ET.Element("compiler")
                root.insert(0, compiler)
            compiler.set("meshdir", str(mesh_dir))
            worldbody = root.find("worldbody")
            if worldbody is None:
                raise RuntimeError("Model XML has no worldbody")
            floor_exists = any(geom.get("name") == self.cfg.floor_name for geom in worldbody.iter("geom"))
            if not floor_exists:
                ET.SubElement(worldbody, "geom", {
                    "name": self.cfg.floor_name, "type": "plane", "size": "3 3 0.1", "pos": "0 0 0",
                    "friction": self._numbers(self.cfg.floor_friction), "solref": self._numbers(self.cfg.floor_solref),
                    "solimp": self._numbers(self.cfg.floor_solimp), "condim": "3", "contype": "1", "conaffinity": "1",
                    "rgba": "0.25 0.27 0.30 1",
                })
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.iterations = int(self.cfg.solver_iterations)
        model.opt.ls_iterations = int(self.cfg.solver_ls_iterations)
        return model

    @staticmethod
    def _numbers(values: Tuple[float, ...]) -> str:
        return " ".join(str(float(value)) for value in values)

    @staticmethod
    def _validate_range(value: Tuple[float, float], name: str) -> Tuple[float, float]:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.isfinite(array).all() or array[0] > array[1]:
            raise ValueError(f"{name} must be a finite (low, high) pair")
        return float(array[0]), float(array[1])

    def _expand_per_leg(self, value: Tuple[float, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (4,):
            array = np.tile(array, 6)
        if array.shape != (self.action_dim,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape (4,) or ({self.action_dim},)")
        return array.copy()

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Cannot find MuJoCo object: {name}")
        return int(object_id)

    def set_command_speed_range(self, speed: Tuple[float, float]) -> None:
        speed_range = self._validate_range(speed, "speed")
        if speed_range[0] < 0.0:
            raise ValueError("speed range must be non-negative")
        self.command_speed_range = speed_range

    def set_command(self, command_x: float, command_y: float, command_yaw: float, command_pitch: float) -> None:
        command = np.asarray([command_x, command_y, command_yaw, command_pitch], dtype=np.float64)
        if not np.isfinite(command).all():
            raise ValueError("Command values must be finite")

        # command_x和command_y始终表示机器人实时Yaw坐标系中的纵向、横向速度。
        self.command_x = float(command_x)
        self.command_y = float(command_y)
        self.command_speed = float(np.linalg.norm(command[:2]))
        self.command_direction = float(np.arctan2(self.command_y, self.command_x)) if self.command_speed > 1e-6 else 0.0
        self.requested_command_yaw = wrap_to_pi(float(command_yaw))
        self.requested_command_pitch = float(np.clip(command_pitch, *self.cfg.command_pitch_limit))

    def _sample_translation_command(self) -> Tuple[float, float]:
        command_speed = float(self.np_random.uniform(*self.command_speed_range))
        command_direction = float(self.np_random.uniform(*self.command_direction_range)) if command_speed > 1e-6 else 0.0
        command_x = float(command_speed * np.cos(command_direction))
        command_y = float(command_speed * np.sin(command_direction))
        return command_x, command_y

    def _sample_command(self) -> None:
        random_value = float(self.np_random.random())
        standing_limit = self.cfg.standing_command_probability
        turning_limit = standing_limit + self.cfg.turn_in_place_probability
        pitch_limit = turning_limit + self.cfg.pitch_in_place_probability
        _, _, current_yaw = self._get_base_rpy()

        if random_value < standing_limit:
            self.set_command(0.0, 0.0, current_yaw, 0.0)
        elif random_value < turning_limit:
            yaw_offset = float(self.np_random.uniform(*self.command_yaw_offset_range))
            self.set_command(0.0, 0.0, wrap_to_pi(current_yaw + yaw_offset), 0.0)
        elif random_value < pitch_limit:
            command_pitch = float(self.np_random.uniform(*self.command_pitch_range))
            self.set_command(0.0, 0.0, current_yaw, command_pitch)
        else:
            command_x, command_y = self._sample_translation_command()
            yaw_offset = float(self.np_random.uniform(*self.command_yaw_offset_range))
            command_pitch = float(self.np_random.uniform(*self.command_pitch_range))
            self.set_command(command_x, command_y, wrap_to_pi(current_yaw + yaw_offset), command_pitch)

    def _schedule_next_command_resample(self) -> None:
        if not self.cfg.resample_commands_during_episode:
            self.next_command_resample_step = 0
            return
        resample_time = float(self.np_random.uniform(*self.cfg.command_resample_time_range))
        self.next_command_resample_step = self.step_count + max(1, int(round(resample_time / self.policy_dt)))

    def _maybe_resample_command(self) -> None:
        if self.cfg.resample_commands_during_episode and self.step_count >= self.next_command_resample_step:
            self._sample_command()
            self._schedule_next_command_resample()

    def _update_pose_command(self) -> None:
        max_yaw_step = self.cfg.command_yaw_slew_rate * self.policy_dt
        yaw_delta = wrap_to_pi(self.requested_command_yaw - self.command_yaw)
        self.command_yaw = wrap_to_pi(self.command_yaw + float(np.clip(yaw_delta, -max_yaw_step, max_yaw_step)))

        max_pitch_step = self.cfg.command_pitch_slew_rate * self.policy_dt
        pitch_delta = self.requested_command_pitch - self.command_pitch
        self.command_pitch += float(np.clip(pitch_delta, -max_pitch_step, max_pitch_step))

    def _get_base_rpy(self) -> Tuple[float, float, float]:
        rotation = self.data.body(self.cfg.base_name).xmat.reshape(3, 3)
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return roll, pitch, yaw

    def _get_base_yaw(self) -> float:
        return self._get_base_rpy()[2]

    def _get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        velocity_world = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.base_id,
            velocity_world,
            0,
        )

        current_yaw = self._get_base_yaw()
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)

        # 只消除实时Yaw，不使用Roll/Pitch，避免俯仰时水平速度与Z轴速度耦合。
        world_to_heading = np.array([
            [cos_yaw, sin_yaw, 0.0],
            [-sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        angular_velocity = world_to_heading @ velocity_world[:3]
        linear_velocity = world_to_heading @ velocity_world[3:]
        return angular_velocity, linear_velocity

    def _get_foot_velocity(self, foot_index: int) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, int(self.foot_site_ids[foot_index]), velocity, 0)
        return velocity[3:].copy()

    def _get_foot_force(self) -> np.ndarray:
        foot_force = np.zeros(6, dtype=np.float64)
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == self.floor_geom_id:
                foot_index = self.foot_geom_to_index.get(int(contact.geom2))
            elif contact.geom2 == self.floor_geom_id:
                foot_index = self.foot_geom_to_index.get(int(contact.geom1))
            else:
                continue
            if foot_index is not None:
                mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
                foot_force[foot_index] += abs(float(contact_force[0]))
        return foot_force

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        base = self.data.body(self.cfg.base_name)
        base_rotation = base.xmat.reshape(3, 3)
        base_ang_vel, base_lin_vel = self._get_base_velocity()
        foot_force = self._get_foot_force()
        roll, pitch, yaw = self._get_base_rpy()
        return {
            "base_pos": base.xpos.copy(), "base_rpy": np.array([roll, pitch, yaw]),
            "base_yaw": np.array(yaw), "yaw_error": np.array(wrap_to_pi(yaw - self.command_yaw)),
            "pitch_error": np.array(pitch - self.command_pitch), "roll_error": np.array(roll),
            "projected_gravity": base_rotation.T @ np.array([0.0, 0.0, -1.0]),
            "base_ang_vel": base_ang_vel, "base_lin_vel": base_lin_vel,
            "q": self.data.qpos[self.joint_qpos_adr].copy(), "dq": self.data.qvel[self.joint_dof_adr].copy(),
            "foot_force": foot_force, "foot_contact": (foot_force > self.cfg.contact_force_threshold).astype(np.float32),
        }

    def _get_obs(self, state: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        state = self._get_robot_state() if state is None else state
        # Yaw目标使用误差sin/cos表达，Command的第3维固定留给Pitch目标，最终观测维度保持95。
        command = np.array([self.command_x / self.cfg.command_velocity_scale, self.command_y / self.cfg.command_velocity_scale, self.command_pitch / self.cfg.command_pitch_scale], dtype=np.float64)
        yaw_error = float(state["yaw_error"])
        joint_obs_scale = np.maximum(self.action_scale, 0.20)
        obs = np.concatenate([
            state["projected_gravity"], state["base_ang_vel"] / 5.0, state["base_lin_vel"], command,
            np.array([(state["base_pos"][2] - self.cfg.target_height) / 0.05]), np.array([np.sin(yaw_error), np.cos(yaw_error)]),
            (state["q"] - self.q_nominal) / joint_obs_scale, state["dq"] / 10.0, state["foot_contact"], self.filtered_action,
            np.array([np.sin(self.phase), np.cos(self.phase)]),
        ])
        if obs.shape != (self.observation_dim,) or not np.isfinite(obs).all():
            raise FloatingPointError(f"Observation is invalid: expected {self.observation_dim}, got {obs.shape}")
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _get_gait_groups(self) -> Tuple[np.ndarray, np.ndarray]:
        tripod_a = np.array([0, 2, 4], dtype=np.int32)
        tripod_b = np.array([1, 3, 5], dtype=np.int32)
        return (tripod_a, tripod_b) if np.sin(self.phase) >= 0.0 else (tripod_b, tripod_a)

    def _compute_tripod_reward(self, contact: np.ndarray) -> float:
        stance_feet, swing_feet = self._get_gait_groups()
        return float(0.7 * np.mean(contact[stance_feet]) + 0.3 * np.mean(1.0 - contact[swing_feet]))

    def _is_standing_command(self, state: Dict[str, np.ndarray]) -> bool:
        command_speed = float(np.hypot(self.command_x, self.command_y))
        requested_yaw_delta = wrap_to_pi(self.requested_command_yaw - self.command_yaw)
        requested_pitch_delta = self.requested_command_pitch - self.command_pitch
        return bool(
            command_speed <= 1e-6
            and abs(float(state["yaw_error"])) <= self.reward_cfg.standing_pose_tolerance
            and abs(float(state["pitch_error"])) <= self.reward_cfg.standing_pose_tolerance
            and abs(requested_yaw_delta) <= self.reward_cfg.standing_pose_tolerance
            and abs(requested_pitch_delta) <= self.reward_cfg.standing_pose_tolerance
        )

    def _compute_pose_motion_gate(self, state: Dict[str, np.ndarray]) -> float:
        requested_yaw_delta = wrap_to_pi(self.requested_command_yaw - self.command_yaw)
        requested_pitch_delta = self.requested_command_pitch - self.command_pitch
        yaw_motion_error = max(abs(float(state["yaw_error"])), abs(requested_yaw_delta))
        pitch_motion_error = max(abs(float(state["pitch_error"])), abs(requested_pitch_delta))
        yaw_gate = float(np.clip(yaw_motion_error / self.reward_cfg.yaw_motion_gate_angle, 0.0, 1.0))
        pitch_gate = float(np.clip(pitch_motion_error / self.reward_cfg.pitch_motion_gate_angle, 0.0, 1.0))
        return max(yaw_gate, pitch_gate)

    def _compute_gait_frequency(self, command_speed: float) -> float:
        speed_min, speed_max = self.cfg.gait_frequency_speed_range
        frequency_min, frequency_max = self.cfg.gait_frequency_range
        speed_ratio = float(np.clip((command_speed - speed_min) / (speed_max - speed_min), 0.0, 1.0))
        return float(frequency_min + speed_ratio * (frequency_max - frequency_min))

    def _advance_gait_phase(self, state: Dict[str, np.ndarray], executed_steps: int) -> None:
        command_speed = float(np.hypot(self.command_x, self.command_y))
        translation_command_gate = float(command_speed > 1e-6)
        pose_motion_gate = self._compute_pose_motion_gate(state)

        # 完全静止时冻结相位；移动Command始终推进相位，原地姿态调整则由姿态误差平滑门控。
        self.gait_phase_gate = 0.0 if self._is_standing_command(state) else max(translation_command_gate, pose_motion_gate)
        self.gait_frequency = self._compute_gait_frequency(command_speed)
        elapsed_time = self.model.opt.timestep * executed_steps
        self.phase = float((self.phase + 2.0 * np.pi * self.gait_frequency * elapsed_time * self.gait_phase_gate) % (2.0 * np.pi))

    def _compute_clearance_reward(self, command_speed: float, contact: np.ndarray) -> Tuple[float, float, float]:
        _, swing_feet = self._get_gait_groups()
        speed_min, speed_max = self.reward_cfg.swing_clearance_speed_range
        speed_ratio = float(np.clip((command_speed - speed_min) / max(speed_max - speed_min, 1e-6), 0.0, 1.0))
        peak_clearance = self.reward_cfg.swing_clearance_min + speed_ratio * (self.reward_cfg.swing_clearance_max - self.reward_cfg.swing_clearance_min)
        swing_height_ratio = abs(np.sin(self.phase))
        desired_clearance = peak_clearance * swing_height_ratio

        # 起步和落脚阶段平滑关闭触地惩罚，避免要求摆动腿在相位切换瞬间离地。
        gate_ratio = float(np.clip(
            (swing_height_ratio - self.reward_cfg.swing_contact_gate_start)
            / (self.reward_cfg.swing_contact_gate_end - self.reward_cfg.swing_contact_gate_start),
            0.0,
            1.0,
        ))
        swing_contact_gate = gate_ratio * gate_ratio * (3.0 - 2.0 * gate_ratio)

        clearance_rewards = []
        clearance_deficits = []
        for foot_index in swing_feet:
            geom_id = self.foot_geom_id_array[foot_index]
            foot_bottom_z = self.data.geom_xpos[geom_id, 2] - self.foot_radii[foot_index]
            clearance_rewards.append(np.exp(-((foot_bottom_z - desired_clearance) / self.reward_cfg.clearance_sigma) ** 2))
            clearance_deficit = max(desired_clearance - foot_bottom_z, 0.0)
            clearance_deficits.append((clearance_deficit / self.reward_cfg.clearance_deficit_scale) ** 2)

        r_clearance = float(np.mean(clearance_rewards))
        p_clearance_deficit = float(min(np.mean(clearance_deficits), 4.0))
        p_swing_contact = float(np.mean(contact[swing_feet]) * swing_contact_gate)
        return r_clearance, p_clearance_deficit, p_swing_contact

    def _geom_label(self, geom_id: int) -> str:
        geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is not None:
            return geom_name
        body_id = int(self.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return f"{body_name or 'body'}:geom_{geom_id}"

    def _get_forbidden_ground_contacts(self) -> Tuple[str, ...]:
        names = set()
        contact_force = np.zeros(6, dtype=np.float64)

        for index in range(self.data.ncon):
            contact = self.data.contact[index]

            if contact.geom1 == self.floor_geom_id:
                other = int(contact.geom2)
            elif contact.geom2 == self.floor_geom_id:
                other = int(contact.geom1)
            else:
                continue

            if other in self.foot_geom_ids:
                continue

            mujoco.mj_contactForce(self.model, self.data, index, contact_force)
            if abs(float(contact_force[0])) < self.cfg.forbidden_contact_force_threshold:
                continue

            names.add(self._geom_label(other))

        return tuple(sorted(names))

    def _physics_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all() and np.isfinite(self.data.qacc).all() and np.isfinite(self.data.ctrl).all())

    def _compute_reward(self, state: Dict[str, np.ndarray], filtered_action: np.ndarray, old_filtered_action: np.ndarray, older_filtered_action: np.ndarray, mean_torque_sq: float, mean_power: float, forbidden_contact: bool) -> Tuple[float, Dict[str, float]]:
        cfg = self.reward_cfg
        lin_vel, ang_vel, contact = state["base_lin_vel"], state["base_ang_vel"], state["foot_contact"]
        contact_count = int(np.sum(contact > 0.5))
        all_feet_contact = float(contact_count == contact.size)
        support_ratio = float(np.clip(contact_count / 3.0, 0.0, 1.0))
        command_xy = np.array([self.command_x, self.command_y], dtype=np.float64)
        command_speed = float(np.linalg.norm(command_xy))
        velocity_error = lin_vel[:2] - command_xy
        velocity_error_sq = float(np.dot(velocity_error, velocity_error))

        # 移动与静止Command分开计算，零速度时不产生除零、进度奖励或停滞惩罚。
        if command_speed > 1e-6:
            command_direction = command_xy / command_speed
            direction_velocity = float(np.dot(lin_vel[:2], command_direction))
            progress_ratio = direction_velocity / command_speed
            movement_gate = float(np.clip(progress_ratio, 0.0, 1.0))
            r_velocity = float(np.exp(-velocity_error_sq / cfg.velocity_sigma ** 2) * movement_gate * (0.6 + 0.4 * support_ratio))
            r_progress = float(np.clip(progress_ratio, -1.0, 1.0))
            p_stall = float(1.0 - movement_gate)
        else:
            progress_ratio = 0.0
            movement_gate = 0.0
            r_velocity = float(np.exp(-velocity_error_sq / cfg.standing_velocity_sigma ** 2))
            r_progress = 0.0
            p_stall = 0.0

        yaw_error = float(state["yaw_error"])
        pitch_error = float(state["pitch_error"])
        roll_error = float(state["roll_error"])
        r_yaw_pose = float(np.exp(-(yaw_error / cfg.yaw_pose_sigma) ** 2))
        r_pitch_pose = float(np.exp(-(pitch_error / cfg.pitch_pose_sigma) ** 2))
        r_roll_pose = float(np.exp(-(roll_error / cfg.roll_pose_sigma) ** 2))

        # 由姿态误差生成期望角速度，使策略既能到达目标姿态，也能在到达后稳定保持。
        target_yaw_rate = float(np.clip(-cfg.yaw_error_to_rate_gain * yaw_error, -cfg.max_target_yaw_rate, cfg.max_target_yaw_rate))
        target_pitch_rate = float(np.clip(-cfg.pitch_error_to_rate_gain * pitch_error, -cfg.max_target_pitch_rate, cfg.max_target_pitch_rate))
        r_pose_rate = float(np.exp(-((ang_vel[2] - target_yaw_rate) / cfg.yaw_rate_sigma) ** 2 - ((ang_vel[1] - target_pitch_rate) / cfg.pitch_rate_sigma) ** 2))

        standing_command = self._is_standing_command(state)
        if standing_command:
            planar_speed_sq = float(np.dot(lin_vel[:2], lin_vel[:2]))
            angular_speed_sq = float(np.dot(ang_vel, ang_vel))
            r_standing = float(np.exp(-planar_speed_sq / cfg.standing_velocity_sigma ** 2 - angular_speed_sq / cfg.standing_angular_velocity_sigma ** 2))
            # 只有六只脚均与地面接触时，才获得完整的静止支撑奖励。
            r_standing_contact = all_feet_contact
        else:
            r_standing = 0.0
            r_standing_contact = 0.0

        r_height = float(np.exp(-((state["base_pos"][2] - self.cfg.target_height) / cfg.height_sigma) ** 2))
        p_flight = float(max(0.0, (3.0 - contact_count) / 3.0))
        p_vertical = float(min((lin_vel[2] / cfg.vertical_velocity_scale) ** 2, 4.0))
        stance_slip = []
        for foot_index in range(6):
            if contact[foot_index] > 0.5:
                velocity = self._get_foot_velocity(foot_index)
                stance_slip.append(min(float(np.dot(velocity[:2], velocity[:2])) / cfg.slip_velocity_scale ** 2, 4.0))
        p_slip = float(np.mean(stance_slip)) if stance_slip else 0.0
        # 平滑惩罚作用于实际送入PD的滤波动作，而不是SAC的原始探索动作。
        p_action_rate = float(np.mean((filtered_action - old_filtered_action) ** 2))
        p_action_acceleration = float(np.mean((filtered_action - 2.0 * old_filtered_action + older_filtered_action) ** 2))
        p_joint_velocity = float(min(np.mean((state["dq"] / cfg.joint_velocity_scale) ** 2), 4.0))
        joint_acceleration = (state["dq"] - self.previous_dq) / self.policy_dt
        p_joint_acceleration = float(min(np.mean((joint_acceleration / cfg.joint_acceleration_scale) ** 2), 4.0))
        torque_limit = max(float(np.max(np.abs(self.ctrl_lower))), float(np.max(np.abs(self.ctrl_upper))), 1.0)
        p_torque = float(min(mean_torque_sq / torque_limit ** 2, 4.0))
        p_power = float(min(mean_power / cfg.power_scale, 4.0))

        pose_motion_gate = self._compute_pose_motion_gate(state)
        gait_gate = 0.0 if standing_command else max(movement_gate, pose_motion_gate)
        r_tripod = self._compute_tripod_reward(contact) * gait_gate if self.cfg.use_tripod_gait_reward else 0.0
        if self.cfg.use_tripod_gait_reward:
            r_clearance, p_clearance_deficit, p_swing_contact = self._compute_clearance_reward(command_speed, contact)
            r_clearance *= gait_gate
            p_clearance_deficit *= gait_gate
            p_swing_contact *= gait_gate
        else:
            r_clearance = 0.0
            p_clearance_deficit = 0.0
            p_swing_contact = 0.0
        reward = (
            cfg.velocity_weight * r_velocity
            + cfg.progress_weight * r_progress
            + cfg.yaw_pose_weight * r_yaw_pose
            + cfg.pitch_pose_weight * r_pitch_pose
            + cfg.roll_pose_weight * r_roll_pose
            + cfg.pose_rate_weight * r_pose_rate
            + cfg.standing_weight * r_standing
            + cfg.standing_contact_weight * r_standing_contact
            + cfg.height_weight * r_height
            + cfg.tripod_weight * r_tripod
            + cfg.clearance_weight * r_clearance
            - cfg.clearance_deficit_weight * p_clearance_deficit
            - cfg.swing_contact_weight * p_swing_contact
            - cfg.stall_weight * p_stall
            - cfg.flight_weight * p_flight
            - cfg.vertical_velocity_weight * p_vertical
            - cfg.slip_weight * p_slip
            - cfg.action_rate_weight * p_action_rate
            - cfg.action_acceleration_weight * p_action_acceleration
            - cfg.torque_weight * p_torque
            - cfg.power_weight * p_power
            - cfg.joint_velocity_weight * p_joint_velocity
            - cfg.joint_acceleration_weight * p_joint_acceleration
            - cfg.forbidden_contact_weight * float(forbidden_contact)
        )
        terms = {
            "total": float(reward),
            "velocity": r_velocity,
            "progress": r_progress,
            "yaw_pose": r_yaw_pose,
            "pitch_pose": r_pitch_pose,
            "roll_pose": r_roll_pose,
            "pose_rate": r_pose_rate,
            "standing": r_standing,
            "standing_contact": r_standing_contact,
            "height": r_height,
            "tripod": r_tripod,
            "clearance": r_clearance,
            "clearance_deficit_penalty": p_clearance_deficit,
            "swing_contact_penalty": p_swing_contact,
            "stall_penalty": p_stall,
            "flight_penalty": p_flight,
            "vertical_velocity_penalty": p_vertical,
            "slip_penalty": p_slip,
            "action_rate_penalty": p_action_rate,
            "action_acceleration_penalty": p_action_acceleration,
            "torque_penalty": p_torque,
            "power_penalty": p_power,
            "joint_velocity_penalty": p_joint_velocity,
            "joint_acceleration_penalty": p_joint_acceleration,
            "forbidden_contact": float(forbidden_contact),
            "contact_count": float(contact_count),
            "all_feet_contact": all_feet_contact,
            "standing_command": float(standing_command),
            "vx": float(lin_vel[0]),
            "vy": float(lin_vel[1]),
            "vz": float(lin_vel[2]),
            "yaw_error": yaw_error,
            "pitch_error": pitch_error,
            "roll_error": roll_error,
            "target_yaw_rate": target_yaw_rate,
            "target_pitch_rate": target_pitch_rate,
            "movement_gate": movement_gate,
            "pose_motion_gate": pose_motion_gate,
            "gait_gate": gait_gate,
            "gait_frequency": self.gait_frequency,
            "gait_phase_gate": self.gait_phase_gate,
            "mean_torque_sq": float(mean_torque_sq),
            "mean_power": float(mean_power),
        }
        return float(reward), terms

    def _check_termination(self, state: Dict[str, np.ndarray], forbidden_contact: bool) -> Tuple[bool, str]:
        base_z = float(state["base_pos"][2])
        if base_z < self.cfg.min_base_height:
            return True, "base_too_low"
        if base_z > self.cfg.max_base_height:
            return True, "base_too_high"
        if state["projected_gravity"][2] > -np.cos(np.deg2rad(self.cfg.max_tilt_deg)):
            return True, "excessive_tilt"
        if forbidden_contact and self.cfg.terminate_on_forbidden_contact:
            return True, "forbidden_ground_contact"
        return False, ""

    def _align_lowest_foot_with_ground(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        foot_center_z = self.data.geom_xpos[self.foot_geom_id_array, 2]
        foot_bottom_z = foot_center_z - self.foot_radii
        lowest_bottom = float(np.min(foot_bottom_z))
        self.data.qpos[self.free_qpos_adr + 2] += self.cfg.initial_ground_clearance - lowest_bottom
        mujoco.mj_forward(self.model, self.data)

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
            self.data.qpos[self.joint_qpos_adr] += self.np_random.uniform(-self.cfg.joint_position_noise, self.cfg.joint_position_noise, self.action_dim)
            self.data.qvel[self.joint_dof_adr] = self.np_random.uniform(-self.cfg.joint_velocity_noise, self.cfg.joint_velocity_noise, self.action_dim)
            self.data.qpos[qadr:qadr + 2] += self.np_random.uniform(-self.cfg.base_position_xy_noise, self.cfg.base_position_xy_noise, 2)
            max_tilt = np.deg2rad(self.cfg.initial_roll_pitch_deg)
            yaw = self.np_random.uniform(*self.cfg.initial_yaw_range)
            noise_quat = euler_to_quat(self.np_random.uniform(-max_tilt, max_tilt), self.np_random.uniform(-max_tilt, max_tilt), yaw)
            self.data.qpos[qadr + 3:qadr + 7] = quat_multiply(noise_quat, self.data.qpos[qadr + 3:qadr + 7].copy())
            self.data.qvel[dadr:dadr + 3] = self.np_random.uniform(-self.cfg.base_linear_velocity_noise, self.cfg.base_linear_velocity_noise, 3)
            self.data.qvel[dadr + 3:dadr + 6] = self.np_random.uniform(-self.cfg.base_angular_velocity_noise, self.cfg.base_angular_velocity_noise, 3)
        self._align_lowest_foot_with_ground()
        _, _, current_yaw = self._get_base_rpy()
        self.command_yaw = current_yaw
        self.requested_command_yaw = current_yaw
        self.command_pitch = 0.0
        self.requested_command_pitch = 0.0
        self._sample_command()
        self.phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.filtered_action.fill(0.0)
        self.previous_filtered_action.fill(0.0)
        self.previous_dq[:] = self.data.qvel[self.joint_dof_adr]
        self.step_count = 0
        self.episode_command_progress = 0.0
        self._schedule_next_command_resample()
        self._episode_ended = False
        self.forbidden_contact_count = 0
        state = self._get_robot_state()
        self.gait_frequency = self._compute_gait_frequency(self.command_speed)
        self.gait_phase_gate = 0.0 if self._is_standing_command(state) else max(float(self.command_speed > 1e-6), self._compute_pose_motion_gate(state))
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, {}, "", (), np.zeros(self.action_dim), self.q_nominal.copy(), 0.0, 0.0)
        self.render()
        return obs, info

    def _make_info(self, state: Dict[str, np.ndarray], reward_terms: Dict[str, float], termination_reason: str, forbidden_contacts: Tuple[str, ...], torque: np.ndarray, q_des: np.ndarray, mean_torque_sq: float, mean_power: float) -> Dict[str, object]:
        return {
            "reward_terms": reward_terms, "termination_reason": termination_reason,
            "base_position": state["base_pos"].copy(), "base_height": float(state["base_pos"][2]),
            "base_linear_velocity": state["base_lin_vel"].copy(), "base_angular_velocity": state["base_ang_vel"].copy(),
            "base_rpy": state["base_rpy"].copy(), "base_yaw": float(state["base_yaw"]),
            "target_yaw": self.command_yaw, "target_pitch": self.command_pitch,
            "yaw_error": float(state["yaw_error"]), "pitch_error": float(state["pitch_error"]), "roll_error": float(state["roll_error"]),
            "command_progress": float(self.episode_command_progress),
            "foot_force": state["foot_force"].copy(), "foot_contact": state["foot_contact"].copy(),
            "forbidden_contact_geoms": forbidden_contacts,
            "command": np.array([self.command_x, self.command_y, self.command_yaw, self.command_pitch], dtype=np.float32),
            "requested_command": np.array([self.command_x, self.command_y, self.requested_command_yaw, self.requested_command_pitch], dtype=np.float32),
            "command_speed": float(self.command_speed), "command_direction": float(self.command_direction),
            "command_direction_deg": float(np.rad2deg(self.command_direction)),
            "torque": torque.copy(), "mean_torque_sq": mean_torque_sq, "mean_power": mean_power,
            "q_des": q_des.copy(), "filtered_action": self.filtered_action.copy(), "phase": self.phase,
            "gait_frequency": self.gait_frequency, "gait_phase_gate": self.gait_phase_gate,
        }

    def step(self, action: np.ndarray):
        if self._episode_ended:
            raise RuntimeError("step() called after episode end; call reset() first")
        raw_action = np.asarray(action, dtype=np.float64)
        if raw_action.shape != (self.action_dim,) or not np.isfinite(raw_action).all():
            raise ValueError(f"action must be finite with shape ({self.action_dim},)")
        raw_action = np.clip(raw_action, -1.0, 1.0)
        self._maybe_resample_command()
        self._update_pose_command()
        old_filtered_action = self.filtered_action.copy()
        older_filtered_action = self.previous_filtered_action.copy()
        filter_delta = self.cfg.action_filter_alpha * (raw_action - self.filtered_action)
        max_filter_delta = self.cfg.action_rate_limit * self.policy_dt
        self.filtered_action += np.clip(filter_delta, -max_filter_delta, max_filter_delta)
        margin = self.cfg.joint_limit_margin
        q_des_start = np.clip(self.q_nominal + self.action_scale * old_filtered_action, self.joint_lower + margin, self.joint_upper - margin)
        q_des_target = np.clip(self.q_nominal + self.action_scale * self.filtered_action, self.joint_lower + margin, self.joint_upper - margin)
        q_des = q_des_start.copy()
        torque = np.zeros(self.action_dim, dtype=np.float64)
        torque_sq_sum = power_sum = 0.0
        executed_steps = 0
        invalid_state = False
        for substep in range(self.frame_skip):
            # 在1000Hz物理子步中连续插值关节目标，消除100Hz策略边界的PD目标跳变。
            interpolation = float(substep + 1) / self.frame_skip
            q_des = q_des_start + interpolation * (q_des_target - q_des_start)
            q = self.data.qpos[self.joint_qpos_adr]
            dq = self.data.qvel[self.joint_dof_adr]
            torque = np.clip(self.kp * (q_des - q) - self.kd * dq, self.ctrl_lower, self.ctrl_upper)
            self.data.ctrl[self.actuator_ids] = torque
            torque_sq_sum += float(np.mean(torque ** 2))
            power_sum += float(np.mean(np.abs(torque * dq)))
            mujoco.mj_step(self.model, self.data)
            executed_steps += 1
            if not self._physics_is_finite():
                invalid_state = True
                break
        self.step_count += 1
        mean_torque_sq = torque_sq_sum / max(executed_steps, 1)
        mean_power = power_sum / max(executed_steps, 1)
        if invalid_state:
            self._episode_ended = True
            reward = -self.reward_cfg.invalid_physics_penalty
            info = {"termination_reason": "invalid_physics", "reward_terms": {"total": reward, "invalid_physics": 1.0}, "command": np.array([self.command_x, self.command_y, self.command_yaw, self.command_pitch]), "torque": torque.copy(), "q_des": q_des.copy()}
            return self._last_valid_obs.copy(), reward, True, False, info

        state = self._get_robot_state()
        self._advance_gait_phase(state, executed_steps)
        command_xy = np.array([self.command_x, self.command_y], dtype=np.float64)
        command_speed = float(np.linalg.norm(command_xy))
        if command_speed > 1e-6:
            # 速度已在实时Yaw坐标系中，按每一步的瞬时目标方向累计实际行进量。
            direction_velocity = float(np.dot(state["base_lin_vel"][:2], command_xy / command_speed))
            self.episode_command_progress += direction_velocity * self.model.opt.timestep * executed_steps
        forbidden_contacts = self._get_forbidden_ground_contacts()
        raw_forbidden_contact = bool(forbidden_contacts)

        if self.step_count <= self.cfg.forbidden_contact_grace_steps:
            self.forbidden_contact_count = 0
        elif raw_forbidden_contact:
            self.forbidden_contact_count += 1
        else:
            self.forbidden_contact_count = 0

        forbidden_contact = self.forbidden_contact_count >= self.cfg.forbidden_contact_persistence_steps
        reward, reward_terms = self._compute_reward(state, self.filtered_action, old_filtered_action, older_filtered_action, mean_torque_sq, mean_power, forbidden_contact)
        terminated, termination_reason = self._check_termination(state, forbidden_contact)
        if terminated:
            reward -= self.reward_cfg.termination_penalty
            reward_terms["termination_penalty"] = self.reward_cfg.termination_penalty
            reward_terms["total"] = float(reward)
        truncated = bool(not terminated and self.step_count >= self.max_episode_steps)
        self._episode_ended = bool(terminated or truncated)
        self.previous_filtered_action[:] = old_filtered_action
        self.previous_dq[:] = state["dq"]
        obs = self._get_obs(state)
        self._last_valid_obs = obs.copy()
        info = self._make_info(state, reward_terms, termination_reason, forbidden_contacts, torque, q_des_target, mean_torque_sq, mean_power)
        self.render()
        return obs, float(reward), terminated, truncated, info

    def update_command_arrow(self) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            return

        command_xy = np.array([self.command_x, self.command_y], dtype=np.float64)
        command_speed = float(np.linalg.norm(command_xy))

        body_pos = self.data.xpos[self.base_id].copy()
        arrow_start = body_pos.copy()
        arrow_start[2] += 0.5

        if command_speed > 1e-6:
            current_yaw = self._get_base_yaw()
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            heading_to_world = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
            command_direction_world_xy = heading_to_world @ (command_xy / command_speed)
            command_direction_world = np.array([command_direction_world_xy[0], command_direction_world_xy[1], 0.0], dtype=np.float64)
            arrow_length = 0.15 + 0.25 * command_speed
            arrow_end = arrow_start + arrow_length * command_direction_world
            arrow_color = np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32)
        else:
            arrow_end = arrow_start + np.array([0.0, 0.0, 0.001], dtype=np.float64)
            arrow_color = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)

        with self.viewer.lock():
            self.viewer.user_scn.ngeom = max(self.viewer.user_scn.ngeom, 1)
            arrow_geom = self.viewer.user_scn.geoms[0]
            mujoco.mjv_connector(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, arrow_start, arrow_end)
            arrow_geom.rgba[:] = arrow_color

    def render(self):
        if self.viewer is not None and self.viewer.is_running():
            self.update_command_arrow()
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
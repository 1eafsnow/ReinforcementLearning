import os
import signal
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import mujoco
import mujoco.viewer
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    SCRIPT_DIR.parent / "snake_description" / "mjcf" / "scene.xml",
    SCRIPT_DIR / "mjcf" / "scene.xml",
    Path("mjcf/scene.xml"),
]

MOTOR_CONFIGS = [
    ("f1_motor", "front_joint1", 0.0, 0.0, 45.0, 5.0, 0.0, 36.0),
    ("f2_motor", "front_joint2", 0.0, 0.0, 45.0, 5.0, 0.0, 36.0),
    ("b1_motor", "back_joint1", 0.0, 0.0, 45.0, 5.0, 0.0, 36.0),
    ("b2_motor", "back_joint2", 0.0, 0.0, 45.0, 5.0, 0.0, 36.0),
    ("front_track_motor", "front_track_drive_joint", 0.0, 0.0, 0.0, 8.0, 0.0, 40.0),
    ("back_track_motor", "back_track_drive_joint", 0.0, 0.0, 0.0, 8.0, 0.0, 40.0),
]

LIDAR_SITE_NAME = "lidar_site"
LIDAR_ROWS = 4
LIDAR_COLS = 120
LIDAR_HFOV_DEG = 120.0
LIDAR_VFOV_DEG = 90.0
LIDAR_MAX_RANGE = 30.0
LIDAR_SCAN_HZ = 10.0


@dataclass
class MitCommand:
    q_des: float = 0.0
    dq_des: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tau_ff: float = 0.0
    tau_limit: float = 0.0
    enabled: bool = True


@dataclass
class MotorState:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0


class SnakeMitController:
    def __init__(self):
        self.model_path = self._find_model()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.qpos_ids = []
        self.dof_ids = []
        for actuator_name, joint_name, *_ in MOTOR_CONFIGS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            if actuator_id < 0:
                raise RuntimeError(f"Actuator not found: {actuator_name}")
            self.qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_ids.append(int(self.model.jnt_dofadr[joint_id]))

        self.commands = [MitCommand(q_des=q, dq_des=dq, kp=kp, kd=kd, tau_ff=ff, tau_limit=limit) for _, _, q, dq, kp, kd, ff, limit in MOTOR_CONFIGS]
        self.states = [MotorState() for _ in MOTOR_CONFIGS]

        self.lidar_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, LIDAR_SITE_NAME)
        if self.lidar_site_id < 0:
            raise RuntimeError(f"LiDAR site not found: {LIDAR_SITE_NAME}")
        self.lidar_enabled = False
        self.lidar_error = ""
        self.lidar_scan = np.full((LIDAR_ROWS, LIDAR_COLS), LIDAR_MAX_RANGE, dtype=np.float32)
        self.lidar_dirs_local = self._build_lidar_directions()
        self.lidar_nray = LIDAR_ROWS * LIDAR_COLS
        self.lidar_geomid = np.empty(self.lidar_nray, dtype=np.int32)
        self.lidar_dist = np.empty(self.lidar_nray, dtype=np.float64)
        self.lidar_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.lidar_next_time = 0.0
        self.lidar_has_normal_arg = self._mujoco_version_at_least(3, 8, 0)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.process_exit_event = threading.Event()
        self.process_exit_ready = threading.Event()
        self.emergency_stop = False
        self.sim_thread = None

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _find_model():
        for path in MODEL_CANDIDATES:
            if path.exists():
                return path.resolve()
        searched = "\n".join(str(path) for path in MODEL_CANDIDATES)
        raise FileNotFoundError(f"Could not find scene.xml. Searched:\n{searched}")

    @staticmethod
    def _mujoco_version_at_least(major, minor, patch):
        numbers = []
        for part in mujoco.__version__.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            numbers.append(int(digits) if digits else 0)
        while len(numbers) < 3:
            numbers.append(0)
        return tuple(numbers[:3]) >= (major, minor, patch)

    @staticmethod
    def _build_lidar_directions():
        h_step = np.deg2rad(LIDAR_HFOV_DEG / LIDAR_COLS)
        v_step = np.deg2rad(LIDAR_VFOV_DEG / LIDAR_ROWS)
        azimuth = (np.arange(LIDAR_COLS, dtype=np.float64) - (LIDAR_COLS - 1) * 0.5) * h_step
        elevation = (np.arange(LIDAR_ROWS, dtype=np.float64) - (LIDAR_ROWS - 1) * 0.5) * v_step
        az, el = np.meshgrid(azimuth, elevation)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        return np.ascontiguousarray(np.stack((x, y, z), axis=-1).reshape(-1, 3), dtype=np.float64)

    def start(self):
        if self.sim_thread is not None and self.sim_thread.is_alive():
            return
        self.stop_event.clear()
        self.process_exit_event.clear()
        self.process_exit_ready.clear()
        self.sim_thread = threading.Thread(target=self._simulation_loop, name="mujoco-simulation", daemon=True)
        self.sim_thread.start()

    def request_process_exit(self):
        with self.lock:
            self.emergency_stop = True
            self.lidar_enabled = False
            self.states = [MotorState(q=s.q, dq=s.dq, tau=0.0) for s in self.states]
        self.process_exit_event.set()

    def request_graceful_stop(self):
        with self.lock:
            self.emergency_stop = True
            self.lidar_enabled = False
        self.stop_event.set()

    def reset(self):
        if not self.stop_event.is_set() and not self.process_exit_event.is_set():
            self.reset_event.set()

    def set_emergency_stop(self, enabled):
        with self.lock:
            self.emergency_stop = enabled

    def set_commands(self, commands):
        with self.lock:
            self.commands = commands

    def get_commands(self):
        with self.lock:
            return [MitCommand(**vars(cmd)) for cmd in self.commands]

    def get_states(self):
        with self.lock:
            return [MotorState(**vars(state)) for state in self.states]

    def set_lidar_enabled(self, enabled):
        with self.lock:
            self.lidar_enabled = bool(enabled)
            if enabled:
                self.lidar_error = ""

    def get_lidar_scan(self):
        with self.lock:
            return self.lidar_scan.copy()

    def get_lidar_state(self):
        with self.lock:
            return self.lidar_enabled, self.lidar_scan.copy(), self.lidar_error

    def _zero_output(self):
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

    def _scan_lidar(self):
        origin = self.data.site_xpos[self.lidar_site_id].copy()
        rotation = self.data.site_xmat[self.lidar_site_id].reshape(3, 3)
        dirs_world = np.ascontiguousarray(self.lidar_dirs_local @ rotation.T, dtype=np.float64).reshape(-1)

        if self.lidar_has_normal_arg:
            mujoco.mj_multiRay(self.model, self.data, origin, dirs_world, self.lidar_geomgroup, 1, -1, self.lidar_geomid, self.lidar_dist, None, self.lidar_nray, LIDAR_MAX_RANGE)
        else:
            mujoco.mj_multiRay(self.model, self.data, origin, dirs_world, self.lidar_geomgroup, 1, -1, self.lidar_geomid, self.lidar_dist, self.lidar_nray, LIDAR_MAX_RANGE)

        ranges = self.lidar_dist.copy()
        ranges[ranges < 0.0] = LIDAR_MAX_RANGE
        np.clip(ranges, 0.0, LIDAR_MAX_RANGE, out=ranges)
        scan = ranges.reshape(LIDAR_ROWS, LIDAR_COLS).astype(np.float32)
        with self.lock:
            self.lidar_scan = scan
            self.lidar_error = ""

    def _simulation_loop(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while not self.stop_event.is_set():
                    if not viewer.is_running():
                        break

                    step_start = time.perf_counter()

                    if self.reset_event.is_set():
                        mujoco.mj_resetData(self.model, self.data)
                        mujoco.mj_forward(self.model, self.data)
                        self.lidar_next_time = 0.0
                        self.reset_event.clear()

                    with self.lock:
                        commands = [MitCommand(**vars(cmd)) for cmd in self.commands]
                        emergency_stop = self.emergency_stop
                        lidar_enabled = self.lidar_enabled

                    self._zero_output()
                    new_states = []

                    for i, command in enumerate(commands):
                        q = float(self.data.qpos[self.qpos_ids[i]])
                        dq = float(self.data.qvel[self.dof_ids[i]])
                        if emergency_stop or not command.enabled:
                            tau = 0.0
                        else:
                            tau = command.kp * (command.q_des - q) + command.kd * (command.dq_des - dq) + command.tau_ff
                            tau_limit = max(0.0, command.tau_limit)
                            tau = float(np.clip(tau, -tau_limit, tau_limit))
                        self.data.qfrc_applied[self.dof_ids[i]] = tau
                        new_states.append(MotorState(q=q, dq=dq, tau=tau))

                    with self.lock:
                        self.states = new_states

                    if self.process_exit_event.is_set():
                        self.process_exit_ready.set()

                    mujoco.mj_step(self.model, self.data)

                    if lidar_enabled and self.data.time >= self.lidar_next_time:
                        try:
                            self._scan_lidar()
                        except Exception as exc:
                            with self.lock:
                                self.lidar_error = str(exc)
                            print(f"LiDAR scan error: {exc}")
                        self.lidar_next_time = self.data.time + 1.0 / LIDAR_SCAN_HZ

                    if self.stop_event.is_set():
                        break

                    try:
                        viewer.sync()
                    except Exception as exc:
                        print(f"MuJoCo viewer stopped: {exc}")
                        break

                    sleep_time = self.model.opt.timestep - (time.perf_counter() - step_start)
                    if sleep_time > 0.0:
                        self.stop_event.wait(sleep_time)
        finally:
            self._zero_output()
            with self.lock:
                self.emergency_stop = True
                self.lidar_enabled = False
                self.states = [MotorState(q=s.q, dq=s.dq, tau=0.0) for s in self.states]
            self.process_exit_ready.set()
            self.stop_event.set()


class SnakeControlUi:
    UPDATE_PERIOD_MS = 50
    EXIT_POLL_MS = 10
    EXIT_ZERO_TIMEOUT_S = 0.5
    EXIT_GRACE_MS = 80

    def __init__(self, root, controller, signal_stop_event):
        self.root = root
        self.controller = controller
        self.signal_stop_event = signal_stop_event
        self.entries = []
        self.state_labels = []
        self.enable_vars = []
        self.closing = False
        self.exit_started_at = None

        self.root.title("Snake Robot MIT Control")
        self.root.protocol("WM_DELETE_WINDOW", self._safe_exit)
        self.root.bind("<Control-c>", lambda _event: self._safe_exit())

        self._build_ui()
        self.controller.start()
        self.root.after(self.UPDATE_PERIOD_MS, self._refresh_state)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        headers = ["Motor", "Enable", "q_des [rad]", "dq_des [rad/s]", "Kp", "Kd", "tau_ff [Nm]", "tau_limit [Nm]", "q", "dq", "tau"]
        for col, text in enumerate(headers):
            ttk.Label(main, text=text).grid(row=0, column=col, padx=4, pady=4)

        for row, ((motor_name, joint_name, *_), command) in enumerate(zip(MOTOR_CONFIGS, self.controller.get_commands()), start=1):
            ttk.Label(main, text=f"{motor_name}\n{joint_name}").grid(row=row, column=0, padx=4, pady=3)

            enable_var = tk.BooleanVar(value=True)
            self.enable_vars.append(enable_var)
            ttk.Checkbutton(main, variable=enable_var).grid(row=row, column=1, padx=4, pady=3)

            row_entries = []
            for offset, value in enumerate([command.q_des, command.dq_des, command.kp, command.kd, command.tau_ff, command.tau_limit]):
                entry = ttk.Entry(main, width=10)
                entry.insert(0, f"{value:g}")
                entry.grid(row=row, column=2 + offset, padx=3, pady=3)
                row_entries.append(entry)
            self.entries.append(row_entries)

            labels = []
            for col in range(8, 11):
                label = ttk.Label(main, text="0.000", width=10, anchor="e")
                label.grid(row=row, column=col, padx=3, pady=3)
                labels.append(label)
            self.state_labels.append(labels)

        controls = ttk.Frame(main)
        controls.grid(row=len(MOTOR_CONFIGS) + 1, column=0, columnspan=11, sticky="w", pady=(12, 0))
        ttk.Button(controls, text="Apply MIT Parameters", command=self._apply).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls, text="Hold Current Position", command=self._hold_current).grid(row=0, column=1, padx=6)
        ttk.Button(controls, text="Zero Targets", command=self._zero_targets).grid(row=0, column=2, padx=6)
        ttk.Button(controls, text="Reset Simulation", command=self.controller.reset).grid(row=0, column=3, padx=6)

        self.estop_button = ttk.Button(controls, text="Emergency Stop", command=self._toggle_estop)
        self.estop_button.grid(row=0, column=4, padx=6)

        self.lidar_var = tk.BooleanVar(value=False)
        self.lidar_toggle = ttk.Checkbutton(controls, text="LiDAR", variable=self.lidar_var, command=self._toggle_lidar)
        self.lidar_toggle.grid(row=0, column=5, padx=(18, 6))

        self.exit_button = ttk.Button(controls, text="Safe Exit", command=self._safe_exit)
        self.exit_button.grid(row=0, column=6, padx=(18, 0))

        self.lidar_status_var = tk.StringVar(value="LiDAR: OFF | 4x120 | FOV 120 x 90 deg | max 30 m | 10 Hz")
        ttk.Label(main, textvariable=self.lidar_status_var).grid(row=len(MOTOR_CONFIGS) + 2, column=0, columnspan=11, sticky="w", pady=(10, 0))

        self.status_var = tk.StringVar(value=f"Model: {self.controller.model_path}")
        ttk.Label(main, textvariable=self.status_var).grid(row=len(MOTOR_CONFIGS) + 3, column=0, columnspan=11, sticky="w", pady=(4, 0))

        tip = "MIT: tau = Kp*(q_des-q) + Kd*(dq_des-dq) + tau_ff. For track speed control, keep Kp=0 and set dq_des/Kd."
        ttk.Label(main, text=tip).grid(row=len(MOTOR_CONFIGS) + 4, column=0, columnspan=11, sticky="w", pady=(4, 0))

    def _read_commands_from_ui(self):
        commands = []
        for i, row_entries in enumerate(self.entries):
            try:
                q_des, dq_des, kp, kd, tau_ff, tau_limit = [float(entry.get()) for entry in row_entries]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in row {i + 1}") from exc
            if kp < 0.0 or kd < 0.0 or tau_limit < 0.0:
                raise ValueError(f"Kp, Kd and tau_limit must be >= 0 in row {i + 1}")
            commands.append(MitCommand(q_des, dq_des, kp, kd, tau_ff, tau_limit, self.enable_vars[i].get()))
        return commands

    def _apply(self):
        if self.closing:
            return
        try:
            self.controller.set_commands(self._read_commands_from_ui())
            self.status_var.set("MIT parameters applied.")
        except ValueError as exc:
            messagebox.showerror("Invalid MIT parameter", str(exc))

    def _hold_current(self):
        if self.closing:
            return
        for i, state in enumerate(self.controller.get_states()[:4]):
            self.entries[i][0].delete(0, tk.END)
            self.entries[i][0].insert(0, f"{state.q:.6f}")
            self.entries[i][1].delete(0, tk.END)
            self.entries[i][1].insert(0, "0")
        self._apply()

    def _zero_targets(self):
        if self.closing:
            return
        for row_entries in self.entries:
            for index in (0, 1, 4):
                row_entries[index].delete(0, tk.END)
                row_entries[index].insert(0, "0")
        self._apply()

    def _toggle_estop(self):
        if self.closing:
            return
        active = self.estop_button.cget("text") == "Emergency Stop"
        self.controller.set_emergency_stop(active)
        self.estop_button.configure(text="Release Emergency Stop" if active else "Emergency Stop")
        self.status_var.set("Emergency stop active." if active else "Emergency stop released.")

    def _toggle_lidar(self):
        if self.closing:
            return
        enabled = self.lidar_var.get()
        self.controller.set_lidar_enabled(enabled)
        self.status_var.set("LiDAR enabled." if enabled else "LiDAR disabled.")

    def _refresh_state(self):
        if self.signal_stop_event.is_set() and not self.closing:
            self._safe_exit()
            return

        for labels, state in zip(self.state_labels, self.controller.get_states()):
            labels[0].configure(text=f"{state.q:.3f}")
            labels[1].configure(text=f"{state.dq:.3f}")
            labels[2].configure(text=f"{state.tau:.3f}")

        lidar_enabled, lidar_scan, lidar_error = self.controller.get_lidar_state()
        if lidar_error:
            self.lidar_status_var.set(f"LiDAR ERROR: {lidar_error}")
        elif lidar_enabled:
            self.lidar_status_var.set(f"LiDAR: ON | 4x120 | min {float(np.min(lidar_scan)):.2f} m | max 30 m | 10 Hz")
        else:
            self.lidar_status_var.set("LiDAR: OFF | 4x120 | FOV 120 x 90 deg | max 30 m | 10 Hz")

        if not self.closing:
            self.root.after(self.UPDATE_PERIOD_MS, self._refresh_state)

    def _safe_exit(self):
        if self.closing:
            return

        self.closing = True
        self.exit_started_at = time.monotonic()
        self.controller.request_process_exit()

        self.exit_button.configure(state=tk.DISABLED)
        self.estop_button.configure(state=tk.DISABLED)
        self.lidar_toggle.configure(state=tk.DISABLED)
        self.status_var.set("Safe exit: requesting zero torque before process termination...")
        self.root.after(self.EXIT_POLL_MS, self._poll_exit_zero)

    def _poll_exit_zero(self):
        ready = self.controller.process_exit_ready.is_set()
        timed_out = time.monotonic() - self.exit_started_at >= self.EXIT_ZERO_TIMEOUT_S

        if ready or timed_out:
            self.status_var.set("Torque zeroed. Closing without GLFW/X11 teardown...")
            self.root.update_idletasks()
            self.root.after(self.EXIT_GRACE_MS, self._hard_process_exit)
            return

        self.root.after(self.EXIT_POLL_MS, self._poll_exit_zero)

    @staticmethod
    def _hard_process_exit():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)


def main():
    signal_stop_event = threading.Event()

    def handle_signal(_signum, _frame):
        signal_stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    controller = SnakeMitController()
    root = tk.Tk()
    SnakeControlUi(root, controller, signal_stop_event)

    try:
        root.mainloop()
    finally:
        controller.request_process_exit()
        time.sleep(0.05)
        os._exit(0)


if __name__ == "__main__":
    main()
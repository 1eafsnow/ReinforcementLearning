import signal
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

        self.joint_ids = []
        self.qpos_ids = []
        self.dof_ids = []
        self.actuator_ids = []

        for actuator_name, joint_name, *_ in MOTOR_CONFIGS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            if actuator_id < 0:
                raise RuntimeError(f"Actuator not found: {actuator_name}")

            self.joint_ids.append(joint_id)
            self.qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
            self.actuator_ids.append(actuator_id)

        self.commands = []
        for _, _, q_des, dq_des, kp, kd, tau_ff, tau_limit in MOTOR_CONFIGS:
            self.commands.append(
                MitCommand(
                    q_des=q_des,
                    dq_des=dq_des,
                    kp=kp,
                    kd=kd,
                    tau_ff=tau_ff,
                    tau_limit=tau_limit,
                    enabled=True,
                )
            )

        self.states = [MotorState() for _ in MOTOR_CONFIGS]
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.emergency_stop = False
        self.sim_thread = None
        self.viewer = None

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _find_model():
        for path in MODEL_CANDIDATES:
            if path.exists():
                return path.resolve()

        searched = "\n".join(str(path) for path in MODEL_CANDIDATES)
        raise FileNotFoundError(f"Could not find scene.xml. Searched:\n{searched}")

    def start(self):
        if self.sim_thread is not None and self.sim_thread.is_alive():
            return

        self.stop_event.clear()
        self.sim_thread = threading.Thread(target=self._simulation_loop, name="mujoco-simulation", daemon=True)
        self.sim_thread.start()

    def stop(self):
        self.request_stop()

    def request_stop(self):
        with self.lock:
            self.emergency_stop = True
            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            self.states = [MotorState(q=s.q, dq=s.dq, tau=0.0) for s in self.states]

        self.stop_event.set()

    def shutdown(self, timeout=3.0):
        self.request_stop()

        thread = self.sim_thread
        if thread is None or thread is threading.current_thread():
            return True

        thread.join(timeout=timeout)
        if not thread.is_alive():
            return True

        self.close_viewer()
        thread.join(timeout=1.0)
        return not thread.is_alive()

    def close_viewer(self):
        with self.lock:
            viewer = self.viewer

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    def reset(self):
        if not self.stop_event.is_set():
            self.reset_event.set()

    def set_emergency_stop(self, enabled):
        with self.lock:
            self.emergency_stop = enabled
            if enabled:
                self.data.ctrl[:] = 0.0
                self.data.qfrc_applied[:] = 0.0

    def set_commands(self, commands):
        with self.lock:
            self.commands = commands

    def get_commands(self):
        with self.lock:
            return [
                MitCommand(
                    q_des=cmd.q_des,
                    dq_des=cmd.dq_des,
                    kp=cmd.kp,
                    kd=cmd.kd,
                    tau_ff=cmd.tau_ff,
                    tau_limit=cmd.tau_limit,
                    enabled=cmd.enabled,
                )
                for cmd in self.commands
            ]

    def get_states(self):
        with self.lock:
            return [MotorState(q=state.q, dq=state.dq, tau=state.tau) for state in self.states]

    def _zero_output(self):
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

    def _simulation_loop(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                with self.lock:
                    self.viewer = viewer

                while viewer.is_running() and not self.stop_event.is_set():
                    step_start = time.perf_counter()

                    if self.reset_event.is_set():
                        mujoco.mj_resetData(self.model, self.data)
                        mujoco.mj_forward(self.model, self.data)
                        self.reset_event.clear()

                    with self.lock:
                        commands = [
                            MitCommand(
                                q_des=cmd.q_des,
                                dq_des=cmd.dq_des,
                                kp=cmd.kp,
                                kd=cmd.kd,
                                tau_ff=cmd.tau_ff,
                                tau_limit=cmd.tau_limit,
                                enabled=cmd.enabled,
                            )
                            for cmd in self.commands
                        ]
                        emergency_stop = self.emergency_stop

                    self._zero_output()
                    new_states = []

                    for i, command in enumerate(commands):
                        q = float(self.data.qpos[self.qpos_ids[i]])
                        dq = float(self.data.qvel[self.dof_ids[i]])

                        if emergency_stop or not command.enabled:
                            tau = 0.0
                        else:
                            tau = (
                                command.kp * (command.q_des - q)
                                + command.kd * (command.dq_des - dq)
                                + command.tau_ff
                            )

                            tau_limit = max(0.0, command.tau_limit)
                            tau = float(np.clip(tau, -tau_limit, tau_limit))

                        self.data.qfrc_applied[self.dof_ids[i]] = tau
                        new_states.append(MotorState(q=q, dq=dq, tau=tau))

                    with self.lock:
                        self.states = new_states

                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    elapsed = time.perf_counter() - step_start
                    sleep_time = self.model.opt.timestep - elapsed

                    if sleep_time > 0.0:
                        self.stop_event.wait(sleep_time)
        finally:
            with self.lock:
                self._zero_output()
                self.emergency_stop = True
                self.viewer = None
                self.states = [MotorState(q=s.q, dq=s.dq, tau=0.0) for s in self.states]

            self.stop_event.set()


class SnakeControlUi:
    UPDATE_PERIOD_MS = 50

    def __init__(self, root, controller, signal_stop_event):
        self.root = root
        self.controller = controller
        self.signal_stop_event = signal_stop_event
        self.entries = []
        self.state_labels = []
        self.enable_vars = []
        self.closing = False

        self.root.title("Snake Robot MIT Control")
        self.root.protocol("WM_DELETE_WINDOW", self._safe_exit)
        self.root.bind("<Control-c>", lambda _event: self._safe_exit())

        self._build_ui()
        self.controller.start()
        self.root.after(self.UPDATE_PERIOD_MS, self._refresh_state)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        headers = [
            "Motor",
            "Enable",
            "q_des [rad]",
            "dq_des [rad/s]",
            "Kp",
            "Kd",
            "tau_ff [Nm]",
            "tau_limit [Nm]",
            "q",
            "dq",
            "tau",
        ]

        for col, text in enumerate(headers):
            ttk.Label(main, text=text).grid(row=0, column=col, padx=4, pady=4)

        commands = self.controller.get_commands()

        for row, ((motor_name, joint_name, *_), command) in enumerate(zip(MOTOR_CONFIGS, commands), start=1):
            ttk.Label(main, text=f"{motor_name}\n{joint_name}").grid(row=row, column=0, padx=4, pady=3)

            enable_var = tk.BooleanVar(value=True)
            self.enable_vars.append(enable_var)
            ttk.Checkbutton(main, variable=enable_var).grid(row=row, column=1, padx=4, pady=3)

            values = [
                command.q_des,
                command.dq_des,
                command.kp,
                command.kd,
                command.tau_ff,
                command.tau_limit,
            ]

            row_entries = []

            for offset, value in enumerate(values):
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

        self.exit_button = ttk.Button(controls, text="Safe Exit", command=self._safe_exit)
        self.exit_button.grid(row=0, column=5, padx=(18, 0))

        self.status_var = tk.StringVar(value=f"Model: {self.controller.model_path}")
        ttk.Label(main, textvariable=self.status_var).grid(
            row=len(MOTOR_CONFIGS) + 2,
            column=0,
            columnspan=11,
            sticky="w",
            pady=(10, 0),
        )

        tip = (
            "MIT: tau = Kp*(q_des-q) + Kd*(dq_des-dq) + tau_ff. "
            "For track speed control, keep Kp=0 and set dq_des/Kd. "
            "Use Safe Exit to stop torque, close the viewer, and exit cleanly."
        )
        ttk.Label(main, text=tip).grid(
            row=len(MOTOR_CONFIGS) + 3,
            column=0,
            columnspan=11,
            sticky="w",
            pady=(4, 0),
        )

    def _read_commands_from_ui(self):
        commands = []

        for i, row_entries in enumerate(self.entries):
            try:
                q_des, dq_des, kp, kd, tau_ff, tau_limit = [float(entry.get()) for entry in row_entries]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in row {i + 1}") from exc

            if kp < 0.0:
                raise ValueError(f"Kp must be >= 0 in row {i + 1}")
            if kd < 0.0:
                raise ValueError(f"Kd must be >= 0 in row {i + 1}")
            if tau_limit < 0.0:
                raise ValueError(f"tau_limit must be >= 0 in row {i + 1}")

            commands.append(
                MitCommand(
                    q_des=q_des,
                    dq_des=dq_des,
                    kp=kp,
                    kd=kd,
                    tau_ff=tau_ff,
                    tau_limit=tau_limit,
                    enabled=self.enable_vars[i].get(),
                )
            )

        return commands

    def _apply(self):
        if self.closing:
            return

        try:
            commands = self._read_commands_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid MIT parameter", str(exc))
            return

        self.controller.set_commands(commands)
        self.status_var.set("MIT parameters applied.")

    def _hold_current(self):
        if self.closing:
            return

        states = self.controller.get_states()

        for i, state in enumerate(states):
            if i >= 4:
                continue

            self.entries[i][0].delete(0, tk.END)
            self.entries[i][0].insert(0, f"{state.q:.6f}")
            self.entries[i][1].delete(0, tk.END)
            self.entries[i][1].insert(0, "0")

        self._apply()

    def _zero_targets(self):
        if self.closing:
            return

        for row_entries in self.entries:
            row_entries[0].delete(0, tk.END)
            row_entries[0].insert(0, "0")
            row_entries[1].delete(0, tk.END)
            row_entries[1].insert(0, "0")
            row_entries[4].delete(0, tk.END)
            row_entries[4].insert(0, "0")

        self._apply()

    def _toggle_estop(self):
        if self.closing:
            return

        if self.estop_button.cget("text") == "Emergency Stop":
            self.controller.set_emergency_stop(True)
            self.estop_button.configure(text="Release Emergency Stop")
            self.status_var.set("Emergency stop active: all commanded torques are zero.")
        else:
            self.controller.set_emergency_stop(False)
            self.estop_button.configure(text="Emergency Stop")
            self.status_var.set("Emergency stop released.")

    def _refresh_state(self):
        if self.signal_stop_event.is_set() and not self.closing:
            self._safe_exit()
            return

        states = self.controller.get_states()

        for labels, state in zip(self.state_labels, states):
            labels[0].configure(text=f"{state.q:.3f}")
            labels[1].configure(text=f"{state.dq:.3f}")
            labels[2].configure(text=f"{state.tau:.3f}")

        if not self.closing and not self.controller.stop_event.is_set():
            self.root.after(self.UPDATE_PERIOD_MS, self._refresh_state)

    def _safe_exit(self):
        if self.closing:
            return

        self.closing = True
        self.status_var.set("Safe shutdown: zeroing torque and closing MuJoCo viewer...")
        self.exit_button.configure(state=tk.DISABLED)
        self.estop_button.configure(state=tk.DISABLED)
        self.root.update_idletasks()

        clean_shutdown = self.controller.shutdown(timeout=3.0)

        if not clean_shutdown:
            self.status_var.set("Viewer did not stop in time; forcing viewer close...")
            self.root.update_idletasks()
            self.controller.close_viewer()
            time.sleep(0.1)

        self.root.after(0, self.root.destroy)


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
        controller.shutdown(timeout=1.0)


if __name__ == "__main__":
    main()

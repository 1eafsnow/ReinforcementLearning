import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = "mjcf/scene.xml"

JOINT_NAMES = [
    "front_joint1",
    "front_joint2",
    "back_joint1",
    "back_joint2",
]

JOINT_ACTUATOR_NAMES = [
    "f1_motor",
    "f2_motor",
    "b1_motor",
    "b2_motor",
]

FRONT_TRACK_ACTUATOR = "front_track_motor"
BACK_TRACK_ACTUATOR = "back_track_motor"

KP = np.array([45.0, 45.0, 45.0, 45.0], dtype=np.float64)
KD = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
MAX_TORQUE = np.array([36.0, 36.0, 36.0, 36.0], dtype=np.float64)


def get_joint_indices(model):
    qpos_ids = []
    qvel_ids = []

    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {name}")

        qpos_ids.append(model.jnt_qposadr[joint_id])
        qvel_ids.append(model.jnt_dofadr[joint_id])

    return np.array(qpos_ids, dtype=np.int32), np.array(qvel_ids, dtype=np.int32)


def get_actuator_ids(model):
    ids = []

    for name in JOINT_ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        if actuator_id < 0:
            raise RuntimeError(f"Actuator not found: {name}")

        ids.append(actuator_id)

    front_track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, FRONT_TRACK_ACTUATOR)
    back_track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, BACK_TRACK_ACTUATOR)

    if front_track_id < 0:
        raise RuntimeError(f"Actuator not found: {FRONT_TRACK_ACTUATOR}")

    if back_track_id < 0:
        raise RuntimeError(f"Actuator not found: {BACK_TRACK_ACTUATOR}")

    return np.array(ids, dtype=np.int32), front_track_id, back_track_id


def pd_control(target_pos, current_pos, current_vel):
    torque = KP * (target_pos - current_pos) - KD * current_vel
    torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
    return torque


def run(target_joint_pos, front_track_speed, back_track_speed):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    qpos_ids, qvel_ids = get_joint_indices(model)
    joint_actuator_ids, front_track_id, back_track_id = get_actuator_ids(model)

    target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)

    if target_joint_pos.shape != (4,):
        raise ValueError("target_joint_pos must contain exactly 4 values")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            current_pos = data.qpos[qpos_ids].copy()
            current_vel = data.qvel[qvel_ids].copy()

            torque = pd_control(target_joint_pos, current_pos, current_vel)

            for i in range(4):
                data.ctrl[joint_actuator_ids[i]] = torque[i]

            #data.ctrl[front_track_id] = front_track_speed
            #data.ctrl[back_track_id] = back_track_speed

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="PD control for four snake joints plus two track velocity actuators."
    )

    parser.add_argument("front_joint1", type=float, help="Target angle of front_joint1 [rad]")
    parser.add_argument("front_joint2", type=float, help="Target angle of front_joint2 [rad]")
    parser.add_argument("back_joint1", type=float, help="Target angle of back_joint1 [rad]")
    parser.add_argument("back_joint2", type=float, help="Target angle of back_joint2 [rad]")
    parser.add_argument("front_track_speed", type=float, help="Front track target speed [rad/s]")
    parser.add_argument("back_track_speed", type=float, help="Back track target speed [rad/s]")

    args = parser.parse_args()

    target_joint_pos = [
        args.front_joint1,
        args.front_joint2,
        args.back_joint1,
        args.back_joint2,
    ]

    run(
        target_joint_pos,
        args.front_track_speed,
        args.back_track_speed,
    )


if __name__ == "__main__":
    main()

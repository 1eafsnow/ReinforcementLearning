import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import FrozenSet, Optional

import numpy as np

from config import ENV_CONFIG, SAC_CONFIG
from env import HexapodWalkEnv
from sac import SACAgent
from train import CHECKPOINT_DIR


@dataclass(frozen=True)
class KeyboardSnapshot:
    pressed: FrozenSet[str]
    reset_requested: bool
    quit_requested: bool
    error: Optional[str]


class WslKeyboard:
    """通过WSLg的X11会话读取键盘按下与释放事件。"""

    def __init__(self):
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "未检测到DISPLAY，pynput无法连接WSLg。请先在Windows PowerShell执行 "
                "wsl --update 和 wsl --shutdown，然后重新进入WSL。"
            )
        os.environ.setdefault("PYNPUT_BACKEND_KEYBOARD", "xorg")
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError("缺少pynput，请先执行: python -m pip install pynput") from exc
        except Exception as exc:
            raise RuntimeError(f"无法连接WSLg的X11键盘会话，DISPLAY={os.environ.get('DISPLAY')}: {exc}") from exc

        self.keyboard = keyboard
        self._pressed = set()
        self._down = set()
        self._reset_requested = False
        self._quit_requested = False
        self._error = None
        self._lock = threading.Lock()
        self._started = False
        self._listener = self.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )

    @property
    def description(self) -> str:
        return f"pynput X11监听器 (DISPLAY={os.environ.get('DISPLAY')})"

    def start(self) -> None:
        self._listener.start()
        self._listener.wait()
        self._started = True
        if not self._listener.running:
            raise RuntimeError("pynput键盘监听器启动失败")

    def _key_name(self, key) -> Optional[str]:
        if key == self.keyboard.Key.up:
            return "w"
        if key == self.keyboard.Key.down:
            return "s"
        if key == self.keyboard.Key.left:
            return "a"
        if key == self.keyboard.Key.right:
            return "d"
        if key == self.keyboard.Key.space:
            return "space"
        if key == self.keyboard.Key.esc:
            return "esc"

        char = getattr(key, "char", None)
        if isinstance(char, str) and char.lower() == "r":
            return "r"
        return None

    def _on_press(self, key):
        key_name = self._key_name(key)
        if key_name is None:
            return None
        with self._lock:
            first_press = key_name not in self._down
            self._down.add(key_name)
            if key_name in {"w", "a", "s", "d", "space"}:
                self._pressed.add(key_name)
            elif key_name == "r" and first_press:
                self._reset_requested = True
            elif key_name == "esc" and first_press:
                self._quit_requested = True
                return False
        return None

    def _on_release(self, key) -> None:
        key_name = self._key_name(key)
        if key_name is None:
            return
        with self._lock:
            self._down.discard(key_name)
            self._pressed.discard(key_name)

    def poll(self) -> KeyboardSnapshot:
        with self._lock:
            if self._started and not self._listener.running and not self._quit_requested:
                self._error = "pynput键盘监听器意外停止"
            snapshot = KeyboardSnapshot(
                pressed=frozenset(self._pressed),
                reset_requested=self._reset_requested,
                quit_requested=self._quit_requested,
                error=self._error,
            )
            self._reset_requested = False
            return snapshot

    def close(self) -> None:
        if self._started:
            self._listener.stop()
            self._listener.join(timeout=0.50)
            self._started = False


def parse_args():
    parser = argparse.ArgumentParser(description="使用WASD实时控制训练好的六足机器人策略")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "sac_hexapod_best.pt"))
    parser.add_argument("--speed", type=float, default=0.10, help="WASD目标平移速度，单位m/s")
    parser.add_argument("--acceleration", type=float, default=0.80, help="有方向输入时的Command变化率，单位m/s^2")
    parser.add_argument("--deceleration", type=float, default=1.20, help="松开方向键后的Command减速率，单位m/s^2")
    parser.add_argument("--episodes", type=int, default=0, help="完成指定回合后退出，0表示持续运行")
    parser.add_argument("--max-episode-steps", type=int, default=0, help="每回合最大策略步数，0表示仅在终止时重置")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="使用随机策略动作，默认使用确定性动作")
    parser.add_argument("--fixed-reset", action="store_true", help="关闭reset时的初始状态随机化")
    parser.add_argument("--status-interval", type=float, default=0.50, help="终端状态刷新间隔，0表示关闭")
    return parser.parse_args()


def validate_args(args) -> None:
    if not np.isfinite(args.speed) or args.speed <= 0.0:
        raise ValueError("--speed必须是正数")
    if not np.isfinite(args.acceleration) or args.acceleration <= 0.0:
        raise ValueError("--acceleration必须是正数")
    if not np.isfinite(args.deceleration) or args.deceleration <= 0.0:
        raise ValueError("--deceleration必须是正数")
    if args.episodes < 0 or args.max_episode_steps < 0:
        raise ValueError("--episodes和--max-episode-steps不能为负数")
    if not np.isfinite(args.status_interval) or args.status_interval < 0.0:
        raise ValueError("--status-interval不能为负数")


def target_command_from_keys(pressed: FrozenSet[str], speed: float) -> np.ndarray:
    if "space" in pressed:
        return np.zeros(2, dtype=np.float64)
    command_direction = np.array([
        float("w" in pressed) - float("s" in pressed),
        float("a" in pressed) - float("d" in pressed),
    ], dtype=np.float64)
    direction_norm = float(np.linalg.norm(command_direction))
    if direction_norm > 1.0:
        command_direction /= direction_norm
    return speed * command_direction


def move_towards(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm <= max_delta or delta_norm <= 1e-12:
        return target.copy()
    return current + delta * (max_delta / delta_norm)


def update_command_in_observation(env: HexapodWalkEnv, obs: np.ndarray, command: np.ndarray) -> np.ndarray:
    """将本策略周期的新平移Command写入95维观测，消除一帧输入延迟。"""
    updated_obs = obs.copy()
    updated_obs[9] = float(np.clip(command[0] / env.cfg.command_velocity_scale, -10.0, 10.0))
    updated_obs[10] = float(np.clip(command[1] / env.cfg.command_velocity_scale, -10.0, 10.0))
    updated_obs[11] = 0.0
    return updated_obs


def reset_environment(env: HexapodWalkEnv, seed: Optional[int], randomize: bool):
    obs, info = env.reset(seed=seed, options={"randomize": randomize})
    hold_yaw = float(info["base_yaw"])
    env.set_command(0.0, 0.0, hold_yaw, 0.0)
    return obs, info, hold_yaw


def print_status(command: np.ndarray, info: dict, pressed: FrozenSet[str]) -> None:
    actual_velocity = np.asarray(info.get("base_linear_velocity", np.zeros(3)), dtype=np.float64)
    movement_keys = "".join(key.upper() for key in "wasd" if key in pressed) or "-"
    message = (
        f"keys={movement_keys:<4}  command=({command[0]:+0.3f}, {command[1]:+0.3f}) m/s  "
        f"velocity=({actual_velocity[0]:+0.3f}, {actual_velocity[1]:+0.3f}) m/s"
    )
    print(message, end="\r", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint}")

    max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else 2_147_483_647
    env_config = replace(
        ENV_CONFIG,
        max_episode_steps=max_episode_steps,
        command_speed_range=(0.0, 0.0),
        command_direction_range=(0.0, 0.0),
        command_yaw_offset_range=(0.0, 0.0),
        command_pitch_range=(0.0, 0.0),
        resample_commands_during_episode=False,
        standing_command_probability=1.0,
        turn_in_place_probability=0.0,
        pitch_in_place_probability=0.0,
    )

    env = None
    keyboard = None
    try:
        env = HexapodWalkEnv(env_config, render_mode="human")
        keyboard = WslKeyboard()
        agent_config = replace(
            SAC_CONFIG,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        agent = SACAgent(agent_config)
        metadata = agent.load(str(checkpoint), load_optimizers=False)
        print(f"已加载 {checkpoint.name}，训练步数 {metadata['step']}，checkpoint版本 {metadata['version']}")
        print(f"键盘输入: {keyboard.description}")
        print("控制: W前进，S后退，A左平移，D右平移，组合键斜向移动，Space立即停止，R重置，Esc退出")
        print("请先点击MuJoCo窗口使其获得键盘焦点。")
        print("Yaw固定为每次reset时的朝向，本程序不生成旋转Command。")

        keyboard.start()
        randomize_reset = not args.fixed_reset
        obs, info, hold_yaw = reset_environment(env, args.seed, randomize_reset)
        command = np.zeros(2, dtype=np.float64)
        episode_reward = 0.0
        episode_length = 0
        completed_episodes = 0
        next_status_time = time.perf_counter()
        next_step_time = time.perf_counter()

        while env.viewer is not None and env.viewer.is_running():
            if args.episodes > 0 and completed_episodes >= args.episodes:
                break

            snapshot = keyboard.poll()
            if snapshot.error is not None:
                raise RuntimeError(snapshot.error)
            if snapshot.quit_requested:
                break
            if snapshot.reset_requested:
                print()
                obs, info, hold_yaw = reset_environment(env, None, randomize_reset)
                command.fill(0.0)
                episode_reward = 0.0
                episode_length = 0
                next_step_time = time.perf_counter()

            target_command = target_command_from_keys(snapshot.pressed, args.speed)
            if "space" in snapshot.pressed:
                command.fill(0.0)
            else:
                rate = args.deceleration if np.linalg.norm(target_command) <= 1e-12 else args.acceleration
                command = move_towards(command, target_command, rate * env.policy_dt)

            env.set_command(float(command[0]), float(command[1]), hold_yaw, 0.0)
            policy_obs = update_command_in_observation(env, obs, command)
            action = agent.select_action(policy_obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            now = time.perf_counter()
            if args.status_interval > 0.0 and now >= next_status_time:
                print_status(command, info, snapshot.pressed)
                next_status_time = now + args.status_interval

            if terminated or truncated:
                completed_episodes += 1
                reason = info.get("termination_reason", "time_limit" if truncated else "")
                print(
                    f"\nEpisode {completed_episodes}: reward={episode_reward:.3f}, "
                    f"length={episode_length}, reason={reason}"
                )
                if args.episodes > 0 and completed_episodes >= args.episodes:
                    break
                obs, info, hold_yaw = reset_environment(env, None, randomize_reset)
                command.fill(0.0)
                episode_reward = 0.0
                episode_length = 0
                next_step_time = time.perf_counter()

            next_step_time += env.policy_dt
            remaining = next_step_time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -env.policy_dt:
                next_step_time = time.perf_counter()
    finally:
        print()
        if keyboard is not None:
            keyboard.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
import argparse
import csv
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from config import ENV_CONFIG, SAC_CONFIG, TRAIN_CONFIG
from env import HexapodWalkEnv
from sac import ReplayBuffer, SACAgent


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"
EPISODE_LOG_PATH = LOG_DIR / "episodes_pose_command.csv"
EVALUATION_LOG_PATH = LOG_DIR / "evaluation_pose_command.csv"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def curriculum_speed_range(step: int) -> Tuple[float, float]:
    selected = TRAIN_CONFIG.velocity_curriculum[0][1:]
    for start_step, low, high in TRAIN_CONFIG.velocity_curriculum:
        if step < start_step:
            break
        selected = (low, high)
    return float(selected[0]), float(selected[1])


def make_env(command_speed_range: Tuple[float, float], render_mode=None) -> HexapodWalkEnv:
    env_config = replace(ENV_CONFIG, command_speed_range=command_speed_range)
    return HexapodWalkEnv(config=env_config, render_mode=render_mode)


def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(agent: SACAgent, step: int, episodes: int, seed: int) -> Dict[str, float]:
    env = make_env(curriculum_speed_range(step), render_mode=None)
    rewards = []
    lengths = []
    distances = []
    mean_velocities_x = []
    mean_velocities_y = []
    yaw_errors = []
    pitch_errors = []
    falls = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            velocity_x_sum = 0.0
            velocity_y_sum = 0.0
            yaw_error_sum = 0.0
            pitch_error_sum = 0.0
            last_progress = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1
                last_progress = float(info.get("command_progress", last_progress))
                if "base_linear_velocity" in info:
                    velocity_x_sum += float(info["base_linear_velocity"][0])
                    velocity_y_sum += float(info["base_linear_velocity"][1])
                yaw_error_sum += abs(float(info.get("yaw_error", 0.0)))
                pitch_error_sum += abs(float(info.get("pitch_error", 0.0)))
            rewards.append(total_reward)
            lengths.append(length)
            distances.append(last_progress)
            mean_velocities_x.append(velocity_x_sum / max(length, 1))
            mean_velocities_y.append(velocity_y_sum / max(length, 1))
            yaw_errors.append(np.rad2deg(yaw_error_sum / max(length, 1)))
            pitch_errors.append(np.rad2deg(pitch_error_sum / max(length, 1)))
            falls.append(float(terminated))
    finally:
        env.close()
    return {
        "reward": float(np.mean(rewards)),
        "length": float(np.mean(lengths)),
        "distance": float(np.mean(distances)),
        "mean_vx": float(np.mean(mean_velocities_x)),
        "mean_vy": float(np.mean(mean_velocities_y)),
        "mean_abs_yaw_error_deg": float(np.mean(yaw_errors)),
        "mean_abs_pitch_error_deg": float(np.mean(pitch_errors)),
        "fall_rate": float(np.mean(falls)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 24-DOF hexapod with Soft Actor-Critic")
    parser.add_argument("--total-steps", type=int, default=TRAIN_CONFIG.total_steps)
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG.seed)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--checkpoint-every", type=int, default=TRAIN_CONFIG.checkpoint_every)
    parser.add_argument("--eval-every", type=int, default=TRAIN_CONFIG.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=TRAIN_CONFIG.eval_episodes)
    parser.add_argument("--render", action="store_true", help="Render training; much slower")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps < 1 or args.eval_episodes < 1:
        raise ValueError("total-steps and eval-episodes must be positive")
    set_global_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(curriculum_speed_range(0), render_mode="human" if args.render else None)
    env.action_space.seed(args.seed)
    cfg = replace(SAC_CONFIG, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], total_steps=args.total_steps, device=args.device)
    agent = SACAgent(cfg)
    replay_buffer = ReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.replay_size, seed=args.seed)
    start_step = 0
    best_eval_reward = -np.inf
    if args.resume is not None:
        metadata = agent.load(args.resume, load_optimizers=True)
        start_step = metadata["step"]
        best_eval_reward = float(metadata["extra"].get("best_eval_reward", -np.inf))
        print(f"Resumed from step {start_step}: {args.resume}")
    if start_step >= args.total_steps:
        raise ValueError(f"Checkpoint step {start_step} is not below total steps {args.total_steps}")

    env.set_command_speed_range(curriculum_speed_range(start_step))
    obs, reset_info = env.reset(seed=args.seed)
    episode = 0
    episode_reward = 0.0
    episode_length = 0
    episode_velocity_x_sum = 0.0
    episode_velocity_y_sum = 0.0
    episode_reward_terms = defaultdict(float)
    episode_last_progress = 0.0
    last_metrics = None
    last_step = start_step

    try:
        for step in range(start_step + 1, cfg.total_steps + 1):
            last_step = step
            collected_steps = step - start_step
            if start_step == 0 and collected_steps <= cfg.start_steps:
                limit = TRAIN_CONFIG.warmup_action_limit
                action = np.random.uniform(-limit, limit, cfg.action_dim).astype(np.float32)
            else:
                action = agent.select_action(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, float(terminated))
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            episode_last_progress = float(info.get("command_progress", episode_last_progress))
            if "base_linear_velocity" in info:
                episode_velocity_x_sum += float(info["base_linear_velocity"][0])
                episode_velocity_y_sum += float(info["base_linear_velocity"][1])
            for name, value in info.get("reward_terms", {}).items():
                if np.isscalar(value):
                    episode_reward_terms[name] += float(value)

            if replay_buffer.size >= max(cfg.update_after, cfg.batch_size):
                for _ in range(cfg.updates_per_step):
                    last_metrics = agent.update(replay_buffer)

            if terminated or truncated:
                episode += 1
                row = {
                    "step": step,
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "distance": episode_last_progress,
                    "mean_vx": episode_velocity_x_sum / max(episode_length, 1),
                    "mean_vy": episode_velocity_y_sum / max(episode_length, 1),
                    "command_x": float(info.get("command", reset_info["command"])[0]),
                    "command_y": float(info.get("command", reset_info["command"])[1]),
                    "command_yaw": float(info.get("command", reset_info["command"])[2]),
                    "command_pitch": float(info.get("command", reset_info["command"])[3]),
                    "terminated": int(terminated),
                    "termination_reason": info.get("termination_reason", ""),
                    "final_yaw_error_deg": np.rad2deg(float(info.get("yaw_error", 0.0))),
                    "final_pitch_error_deg": np.rad2deg(float(info.get("pitch_error", 0.0))),
                    "mean_tracking_reward": episode_reward_terms["velocity"] / max(episode_length, 1),
                    "mean_stall_penalty": episode_reward_terms["stall_penalty"] / max(episode_length, 1),
                    "mean_slip_penalty": episode_reward_terms["slip_penalty"] / max(episode_length, 1),
                    "mean_torque_sq": episode_reward_terms["mean_torque_sq"] / max(episode_length, 1),
                }
                append_csv(EPISODE_LOG_PATH, row)
                print(f"Episode {episode:5d} | Step {step:8d} | Reward {episode_reward:9.3f} | Distance {row['distance']:+.3f} m | MeanVx {row['mean_vx']:+.3f} m/s |  MeanVy {row['mean_vy']:+.3f} m/s | YawErr {row['final_yaw_error_deg']:+.1f} deg | Length {episode_length:4d} | {row['termination_reason']}")
                env.set_command_speed_range(curriculum_speed_range(step))
                obs, reset_info = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_velocity_x_sum = 0.0
                episode_velocity_y_sum = 0.0
                episode_reward_terms.clear()
                episode_last_progress = 0.0

            if step % TRAIN_CONFIG.log_updates_every == 0 and last_metrics is not None:
                append_csv(LOG_DIR / "updates.csv", {"step": step, **last_metrics})
                print(f"[SAC] step={step} actor={last_metrics['actor_loss']:.4f} critic={last_metrics['critic_loss']:.4f} alpha={last_metrics['alpha']:.4f} entropy={last_metrics['entropy']:.4f} Q={last_metrics['q']:.4f}")

            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(agent, step, args.eval_episodes, args.seed + 10_000)
                append_csv(EVALUATION_LOG_PATH, {"step": step, **eval_metrics})
                print(f"[Eval] step={step} reward={eval_metrics['reward']:.3f} distance={eval_metrics['distance']:.3f} mean_vx={eval_metrics['mean_vx']:.3f} mean_vy={eval_metrics['mean_vy']:.3f} yaw_error={eval_metrics['mean_abs_yaw_error_deg']:.2f}deg pitch_error={eval_metrics['mean_abs_pitch_error_deg']:.2f}deg fall_rate={eval_metrics['fall_rate']:.2%}")
                if eval_metrics["reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["reward"]
                    agent.save(str(CHECKPOINT_DIR / "sac_hexapod_best.pt"), step, {"best_eval_reward": best_eval_reward})

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                agent.save(str(CHECKPOINT_DIR / f"sac_hexapod_{step}.pt"), step, {"best_eval_reward": best_eval_reward})

        agent.save(str(CHECKPOINT_DIR / "sac_hexapod_final.pt"), cfg.total_steps, {"best_eval_reward": best_eval_reward})
    except KeyboardInterrupt:
        interrupted_path = CHECKPOINT_DIR / "sac_hexapod_interrupted.pt"
        agent.save(str(interrupted_path), last_step, {"best_eval_reward": best_eval_reward})
        print(f"Training interrupted; checkpoint saved to {interrupted_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

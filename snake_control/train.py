import argparse
import csv
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from config import ENV_CONFIG, SAC_CONFIG, TRAIN_CONFIG
from env import SnakeAvoidEnv
from sac import ReplayBuffer, SACAgent


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"
EPISODE_LOG_PATH = LOG_DIR / "episodes_avoidance.csv"
EVALUATION_LOG_PATH = LOG_DIR / "evaluation_avoidance.csv"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def curriculum_obstacle_offset(step: int) -> float:
    selected = float(TRAIN_CONFIG.obstacle_offset_curriculum[0][1])
    for start_step, max_abs_offset in TRAIN_CONFIG.obstacle_offset_curriculum:
        if step < start_step:
            break
        selected = float(max_abs_offset)
    return selected


def make_env(obstacle_offset: float, render_mode=None) -> SnakeAvoidEnv:
    env = SnakeAvoidEnv(config=ENV_CONFIG, render_mode=render_mode)
    env.set_obstacle_offset_limit(obstacle_offset)
    return env


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
    env = make_env(curriculum_obstacle_offset(step), render_mode=None)
    rewards = []
    lengths = []
    final_distances = []
    progresses = []
    path_lengths = []
    successes = []
    collisions = []
    mean_lidar_min = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            lidar_min_sum = 0.0
            length = 0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                lidar_min_sum += float(info.get("lidar_min", env.cfg.lidar_max_range))
                length += 1
            rewards.append(total_reward)
            lengths.append(length)
            final_distances.append(float(info.get("goal_distance", np.nan)))
            progresses.append(float(info.get("episode_progress", 0.0)))
            path_lengths.append(float(info.get("path_length", 0.0)))
            successes.append(float(info.get("success", False)))
            collisions.append(float(info.get("collision", False)))
            mean_lidar_min.append(lidar_min_sum / max(length, 1))
    finally:
        env.close()
    return {
        "reward": float(np.mean(rewards)),
        "length": float(np.mean(lengths)),
        "final_goal_distance": float(np.nanmean(final_distances)),
        "progress": float(np.mean(progresses)),
        "path_length": float(np.mean(path_lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "mean_lidar_min": float(np.mean(mean_lidar_min)),
        "obstacle_offset_limit": curriculum_obstacle_offset(step),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the snake tracked robot for point-goal obstacle avoidance with Soft Actor-Critic")
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

    env = make_env(curriculum_obstacle_offset(0), render_mode="human" if args.render else None)
    env.action_space.seed(args.seed)
    cfg = replace(SAC_CONFIG, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], total_steps=args.total_steps, device=args.device)
    agent = SACAgent(cfg)
    replay_buffer = ReplayBuffer(cfg.obs_dim, cfg.action_dim, cfg.replay_size, seed=args.seed)
    start_step = 0
    best_eval_reward = -np.inf
    best_success_rate = 0.0
    if args.resume is not None:
        metadata = agent.load(args.resume, load_optimizers=True)
        start_step = metadata["step"]
        best_eval_reward = float(metadata["extra"].get("best_eval_reward", -np.inf))
        best_success_rate = float(metadata["extra"].get("best_success_rate", 0.0))
        print(f"Resumed from step {start_step}: {args.resume}")
    if start_step >= args.total_steps:
        raise ValueError(f"Checkpoint step {start_step} is not below total steps {args.total_steps}")

    env.set_obstacle_offset_limit(curriculum_obstacle_offset(start_step))
    obs, _ = env.reset(seed=args.seed)
    episode = 0
    episode_reward = 0.0
    episode_length = 0
    episode_reward_terms = defaultdict(float)
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
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "termination_reason": info.get("termination_reason", "timeout" if truncated else ""),
                    "goal_distance": float(info.get("goal_distance", np.nan)),
                    "progress": float(info.get("episode_progress", 0.0)),
                    "path_length": float(info.get("path_length", 0.0)),
                    "lidar_min": float(info.get("lidar_min", env.cfg.lidar_max_range)),
                    "heading_error_deg": np.rad2deg(float(info.get("heading_error", 0.0))),
                    "obstacle_offset_limit": curriculum_obstacle_offset(step),
                    "mean_progress_reward": episode_reward_terms["progress"] / max(episode_length, 1),
                    "mean_clearance_penalty": episode_reward_terms["clearance_penalty"] / max(episode_length, 1),
                    "mean_action_rate_penalty": episode_reward_terms["action_rate_penalty"] / max(episode_length, 1),
                }
                append_csv(EPISODE_LOG_PATH, row)
                print(f"Episode {episode:5d} | Step {step:8d} | Reward {episode_reward:9.3f} | GoalDist {row['goal_distance']:.3f} m | Progress {row['progress']:+.3f} m | Success {row['success']} | Collision {row['collision']} | Length {episode_length:4d} | {row['termination_reason']}")
                env.set_obstacle_offset_limit(curriculum_obstacle_offset(step))
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_reward_terms.clear()

            if step % TRAIN_CONFIG.log_updates_every == 0 and last_metrics is not None:
                append_csv(LOG_DIR / "updates.csv", {"step": step, **last_metrics})
                print(f"[SAC] step={step} actor={last_metrics['actor_loss']:.4f} critic={last_metrics['critic_loss']:.4f} alpha={last_metrics['alpha']:.4f} entropy={last_metrics['entropy']:.4f} Q={last_metrics['q']:.4f}")

            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_metrics = evaluate(agent, step, args.eval_episodes, args.seed + 10_000)
                append_csv(EVALUATION_LOG_PATH, {"step": step, **eval_metrics})
                print(f"[Eval] step={step} reward={eval_metrics['reward']:.3f} success={eval_metrics['success_rate']:.1%} collision={eval_metrics['collision_rate']:.1%} final_dist={eval_metrics['final_goal_distance']:.3f} m progress={eval_metrics['progress']:.3f} m")
                improved = eval_metrics["success_rate"] > best_success_rate or (np.isclose(eval_metrics["success_rate"], best_success_rate) and eval_metrics["reward"] > best_eval_reward)
                if improved:
                    best_success_rate = eval_metrics["success_rate"]
                    best_eval_reward = eval_metrics["reward"]
                    agent.save(str(CHECKPOINT_DIR / "sac_snake_best.pt"), step, {"best_eval_reward": best_eval_reward, "best_success_rate": best_success_rate})

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                agent.save(str(CHECKPOINT_DIR / f"sac_snake_{step}.pt"), step, {"best_eval_reward": best_eval_reward, "best_success_rate": best_success_rate})

        agent.save(str(CHECKPOINT_DIR / "sac_snake_final.pt"), cfg.total_steps, {"best_eval_reward": best_eval_reward, "best_success_rate": best_success_rate})
    except KeyboardInterrupt:
        interrupted_path = CHECKPOINT_DIR / "sac_snake_interrupted.pt"
        agent.save(str(interrupted_path), last_step, {"best_eval_reward": best_eval_reward, "best_success_rate": best_success_rate})
        print(f"Training interrupted; checkpoint saved to {interrupted_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

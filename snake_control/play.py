import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from config import ENV_CONFIG, SAC_CONFIG
from env import SnakeAvoidEnv
from sac import SACAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained snake obstacle-avoidance SAC policy")
    parser.add_argument("--checkpoint", type=str, default=str(Path(__file__).resolve().parent / "checkpoints" / "sac_snake_best.pt"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--obstacle-offset", type=float, default=0.15, help="Maximum obstacle lateral offset from the direct start-goal path")
    parser.add_argument("--no-randomize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SnakeAvoidEnv(config=ENV_CONFIG, render_mode="human")
    env.set_obstacle_offset_limit(args.obstacle_offset)
    cfg = replace(SAC_CONFIG, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=args.device)
    agent = SACAgent(cfg)
    metadata = agent.load(args.checkpoint, load_optimizers=False)
    print(f"Loaded {args.checkpoint} at training step {metadata['step']}")

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode, options={"randomize": not args.no_randomize})
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                if env.viewer is not None and not env.viewer.is_running():
                    return
                start = time.perf_counter()
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                sleep_time = env.policy_dt - (time.perf_counter() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
            print(f"Episode {episode + 1:3d} | Reward {total_reward:9.3f} | Success {int(info.get('success', False))} | Collision {int(info.get('collision', False))} | GoalDist {float(info.get('goal_distance', 0.0)):.3f} m | {info.get('termination_reason', 'timeout')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

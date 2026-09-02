import argparse
from dataclasses import replace

import numpy as np

from config import ENV_CONFIG
from env import HexapodWalkEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a finite-state and standing smoke test")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--randomized-reset", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    env_config = replace(ENV_CONFIG, command_speed_range=(0.0, 0.0), standing_command_probability=1.0, max_episode_steps=100000000)
    env = HexapodWalkEnv(env_config, render_mode="human" if args.render else None)
    env.action_space.seed(args.seed)
    reset_options = {"randomize": args.random_actions or args.randomized_reset}
    obs, reset_info = env.reset(seed=args.seed, options=reset_options)
    phase_at_reset = env.phase
    frequency_samples = np.array([env._compute_gait_frequency(speed) for speed in (0.0, 0.03, 0.07, 0.15, 0.25)], dtype=np.float64)
    episodes = 0
    try:
        assert env.observation_space.contains(obs)
        assert env.observation_space.shape == (95,)
        assert np.allclose(reset_info["command"][:2], 0.0)
        assert np.all(np.diff(frequency_samples) >= 0.0)
        for step in range(1, args.steps + 1):
            action = env.action_space.sample() if args.random_actions else np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if not env.observation_space.contains(obs) or not np.isfinite(reward):
                raise AssertionError(f"Invalid transition at step {step}")
            if terminated or truncated:
                episodes += 1
                if not args.random_actions and terminated:
                    geoms = ", ".join(info.get("forbidden_contact_geoms", ()))
                    details = f" ({geoms})" if geoms else ""
                    raise AssertionError(f"Zero-action standing failed at step {step}: {info.get('termination_reason', '')}{details}")
                obs, _ = env.reset(options=reset_options)
        if not args.random_actions and not args.randomized_reset:
            reward_terms = info.get("reward_terms", {})
            assert reward_terms.get("standing_command") == 1.0
            assert reward_terms.get("all_feet_contact") == 1.0
            assert reward_terms.get("standing_contact") == 1.0
            assert reward_terms.get("gait_gate") == 0.0
            assert info.get("gait_phase_gate") == 0.0
            assert env.phase == phase_at_reset
        print(f"Smoke test passed: steps={args.steps}, completed_episodes={episodes}, final_height={info.get('base_height', float('nan')):.4f}, yaw_error_deg={np.rad2deg(info.get('yaw_error', 0.0)):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

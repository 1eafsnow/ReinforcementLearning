import numpy as np

from env import SnakeTraverseEnv


def main() -> None:
    env = SnakeTraverseEnv(render_mode=None)
    env.set_obstacle_height_limit(min(0.06, env.cfg.obstacle_max_supported_height))
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert env.cfg.obstacle_min_height <= info["obstacle_height"] <= env.obstacle_height_limit
        print(f"obs_dim={obs.size}, action_dim={env.action_space.shape[0]}, obstacle_height={info['obstacle_height']:.3f} m, ramp={info['obstacle_ramp_length']:.3f} m, platform={info['obstacle_platform_length']:.3f} m, lidar_min={info['lidar_min']:.3f} m")
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise FloatingPointError(f"Non-finite result at step {step}")
            if terminated or truncated:
                print(f"episode ended at step {step + 1}: {info.get('termination_reason', 'timeout')}")
                obs, info = env.reset()
        print("SnakeTraverseEnv smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()

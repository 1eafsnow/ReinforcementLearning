import numpy as np

from env import SnakeAvoidEnv


def main() -> None:
    env = SnakeAvoidEnv(render_mode=None)
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        print(f"obs_dim={obs.size}, action_dim={env.action_space.shape[0]}, lidar_min={info['lidar_min']:.3f} m")
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise FloatingPointError(f"Non-finite result at step {step}")
            if terminated or truncated:
                print(f"episode ended at step {step + 1}: {info.get('termination_reason', 'timeout')}")
                obs, info = env.reset()
        print("SnakeAvoidEnv smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()

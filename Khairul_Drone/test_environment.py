import numpy as np

from environments.airsim_env import AirSimDroneEnv


def main() -> None:
    env = AirSimDroneEnv()
    print("Testing environment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    obs = env.reset()
    print(f"[OK] Reset successful, observation shape: {obs.shape}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {step + 1}: reward={reward:.2f}, done={done}, info={info}")
        if done:
            break

    env.close()
    print("[OK] Environment test complete")


if __name__ == "__main__":
    main()


import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from algorithms.replay_buffer import ReplayBuffer
from algorithms.td3 import TD3
from environments.airsim_env import AirSimDroneEnv
from models.actor import Actor
from models.critic import Critic
from models.feature_extractor import DepthFeatureExtractor


EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 128
BUFFER_SIZE = 50_000
EXPLORATION_NOISE = 0.3
EVAL_FREQ = 50


def fill_replay_buffer(env: AirSimDroneEnv, buffer: ReplayBuffer, steps: int) -> None:
    obs = env.reset()
    depth_image = env.get_latest_depth_image()
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        next_depth_image = env.get_latest_depth_image()
        buffer.add(
            depth_image,
            obs[25:],
            action,
            reward,
            next_depth_image,
            next_obs[25:],
            done,
        )
        if done:
            obs = env.reset()
            depth_image = env.get_latest_depth_image()
        else:
            obs = next_obs
            depth_image = next_depth_image


def evaluate_agent(env: AirSimDroneEnv, agent: TD3, episodes: int = 10) -> float:
    rewards: List[float] = []
    for _ in range(episodes):
        obs = env.reset()
        depth_image = env.get_latest_depth_image()
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action = agent.select_action(depth_image, obs[25:], noise=0.0)
            obs, reward, done, info = env.step(action)
            depth_image = env.get_latest_depth_image()
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    return float(np.mean(rewards))


def plot_rewards(rewards: List[float], output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Baseline Training Progress")
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    os.makedirs("data/checkpoints", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)

    env = AirSimDroneEnv()
    feature_extractor = DepthFeatureExtractor()
    actor = Actor()
    critic = Critic()
    agent = TD3(feature_extractor, actor, critic)
    replay_buffer = ReplayBuffer(max_size=BUFFER_SIZE)

    episode_rewards: List[float] = []
    success_count = 0

    print("Device:", agent.device)
    print("Filling replay buffer with random exploration...")
    fill_replay_buffer(env, replay_buffer, 2000)
    print(f"Buffer populated with {len(replay_buffer)} transitions")

    for episode in range(EPISODES):
        obs = env.reset()
        depth_image = env.get_latest_depth_image()
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            state = obs[25:]
            action = agent.select_action(depth_image, state, noise=EXPLORATION_NOISE)

            next_obs, reward, done, info = env.step(action)
            next_depth_image = env.get_latest_depth_image()
            episode_reward += reward

            replay_buffer.add(
                depth_image,
                state,
                action,
                reward,
                next_depth_image,
                next_obs[25:],
                done,
            )

            if len(replay_buffer) > BATCH_SIZE:
                agent.train(replay_buffer, BATCH_SIZE)

            obs = next_obs
            depth_image = next_depth_image
            if done:
                if info.get("termination") == "success":
                    success_count += 1
                break

        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            success_rate = success_count / (episode + 1)
            print(
                f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                f"Success Rate = {success_rate:.2%}"
            )

        if episode % EVAL_FREQ == 0:
            mean_reward = evaluate_agent(env, agent, episodes=10)
            print(f"Evaluation at episode {episode}: Avg Reward = {mean_reward:.2f}")
            checkpoint_path = f"data/checkpoints/baseline_ep{episode}.pt"
            agent.save(checkpoint_path)

    env.close()

    plot_rewards(episode_rewards, "data/plots/baseline_training.png")
    print("Training complete.")
    print(f"Final success rate: {success_count / EPISODES:.2%}")


if __name__ == "__main__":
    main()


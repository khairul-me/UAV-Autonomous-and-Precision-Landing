from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from algorithms.td3 import TD3
from attacks.fgsm import FGSM
from attacks.motion_blur import MotionBlur
from attacks.pgd import PGD
from environments.airsim_env import AirSimDroneEnv
from models.actor import Actor
from models.critic import Critic
from models.feature_extractor import DepthFeatureExtractor


def evaluate_under_attack(
    env: AirSimDroneEnv, agent: TD3, attack, episodes: int = 20
) -> Dict[str, float]:
    results = {"success": 0, "collision": 0, "rewards": []}

    for _ in range(episodes):
        obs = env.reset()
        depth_image = env.get_latest_depth_image()
        total_reward = 0.0

        for _ in range(500):
            state = obs[25:]
            if attack is not None:
                depth_used = attack.attack(depth_image, state)
            else:
                depth_used = depth_image

            action = agent.select_action(depth_used, state, noise=0.0)

            obs, reward, done, info = env.step(action)
            depth_image = env.get_latest_depth_image()
            total_reward += reward

            if done:
                termination = info.get("termination")
                if termination == "success":
                    results["success"] += 1
                elif termination == "collision":
                    results["collision"] += 1
                break

        results["rewards"].append(total_reward)

    success_rate = results["success"] / episodes
    collision_rate = results["collision"] / episodes
    avg_reward = float(np.mean(results["rewards"]))
    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "avg_reward": avg_reward,
    }


def main() -> None:
    env = AirSimDroneEnv()
    feature_extractor = DepthFeatureExtractor()
    actor = Actor()
    critic = Critic()
    agent = TD3(feature_extractor, actor, critic)

    checkpoint_path = "data/checkpoints/baseline_ep500.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found. Train the baseline first."
        )
    agent.load(checkpoint_path)

    print("Evaluating baseline (no attack)...")
    baseline_results = evaluate_under_attack(env, agent, None)

    print("Evaluating under FGSM attack...")
    fgsm = FGSM(agent, epsilon=0.03)
    fgsm_results = evaluate_under_attack(env, agent, fgsm)

    print("Evaluating under PGD attack...")
    pgd = PGD(agent, epsilon=0.03, num_iter=10)
    pgd_results = evaluate_under_attack(env, agent, pgd)

    print("Evaluating under Motion Blur attack...")
    motion_blur = MotionBlur(kernel_size=15, angle=45.0)
    blur_results = evaluate_under_attack(env, agent, motion_blur)

    env.close()

    print("\nAttack evaluation summary:")
    headers = ["Attack", "Success Rate", "Collision Rate", "Avg Reward"]
    rows = [
        ("Baseline", baseline_results),
        ("FGSM", fgsm_results),
        ("PGD", pgd_results),
        ("Motion Blur", blur_results),
    ]
    for name, res in rows:
        print(
            f"{name:15} success={res['success_rate']:.2%} "
            f"collision={res['collision_rate']:.2%} avg_reward={res['avg_reward']:.2f}"
        )

    os.makedirs("data/plots", exist_ok=True)
    attacks = [row[0] for row in rows]
    success_rates = [row[1]["success_rate"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(attacks, success_rates, color=["green", "orange", "red", "blue"])
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.title("Impact of Adversarial Attacks on Drone Navigation")
    plt.savefig("data/plots/attack_comparison.png")
    plt.close()
    print("Plots saved to data/plots/attack_comparison.png")


if __name__ == "__main__":
    main()


from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.td3 import TD3
from attacks.fgsm import FGSM
from attacks.motion_blur import MotionBlur
from attacks.pgd import PGD
from defenses.input_sanitization import InputSanitizer
from defenses.multi_sensor_fusion import MultiSensorFusion
from defenses.temporal_consistency import TemporalConsistencyChecker
from environments.airsim_env import AirSimDroneEnv
from models.actor import Actor
from models.critic import Critic
from models.feature_extractor import DepthFeatureExtractor

MAX_STEPS = 500


def evaluate_configuration(
    env: AirSimDroneEnv,
    agent: TD3,
    attack,
    defense: Optional[str],
    episodes: int = 50,
) -> Dict[str, float]:
    outcomes = {"success": 0, "collision": 0, "timeout": 0, "rewards": [], "lengths": []}

    for ep in range(episodes):
        obs = env.reset()
        depth_image = env.get_latest_depth_image()
        total_reward = 0.0
        steps = 0

        sanitizer = InputSanitizer() if defense == "sanitizer" else None
        temp_checker = TemporalConsistencyChecker() if defense == "temporal" else None
        sensor_fusion = (
            MultiSensorFusion(env.client) if defense == "multi_sensor" else None
        )

        for step in range(MAX_STEPS):
            state = obs[25:]
            depth_for_action = depth_image.copy()

            if attack is not None:
                depth_for_action = attack.attack(depth_for_action, state)

            if sanitizer is not None:
                depth_for_action = sanitizer.sanitize(depth_for_action)

            if sensor_fusion is not None:
                detected, reason = sensor_fusion.detect_attack(depth_for_action)
                if detected:
                    action = sensor_fusion.fallback_action()
                else:
                    action = agent.select_action(depth_for_action, state, noise=0.0)
            else:
                action = agent.select_action(depth_for_action, state, noise=0.0)

            if temp_checker is not None:
                anomaly, reason = temp_checker.check_anomaly(depth_for_action, action)
                if anomaly:
                    action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

            obs, reward, done, info = env.step(action)
            depth_image = env.get_latest_depth_image()

            total_reward += reward
            steps += 1

            if done:
                termination = info.get("termination", "timeout")
                outcomes[termination] += 1
                break
        else:
            outcomes["timeout"] += 1

        outcomes["rewards"].append(total_reward)
        outcomes["lengths"].append(steps)

    return {
        "success_rate": outcomes["success"] / episodes,
        "collision_rate": outcomes["collision"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "avg_reward": float(np.mean(outcomes["rewards"])),
        "avg_length": float(np.mean(outcomes["lengths"])),
    }


def load_agent(checkpoint_path: str) -> TD3:
    feature_extractor = DepthFeatureExtractor()
    actor = Actor()
    critic = Critic()
    agent = TD3(feature_extractor, actor, critic)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found. Train the model before evaluation."
        )
    agent.load(checkpoint_path)
    return agent


def main() -> None:
    os.makedirs("data/plots", exist_ok=True)

    env = AirSimDroneEnv()

    baseline_agent = load_agent("data/checkpoints/baseline_ep500.pt")
    robust_agent = load_agent("data/checkpoints/robust_ep500.pt")

    configurations = [
        ("Baseline - No Attack", baseline_agent, None, None),
        ("Robust - No Attack", robust_agent, None, None),
        ("Baseline + FGSM", baseline_agent, FGSM(baseline_agent), None),
        ("Robust + FGSM", robust_agent, FGSM(robust_agent), None),
        (
            "Baseline + FGSM + Sanitizer",
            baseline_agent,
            FGSM(baseline_agent),
            "sanitizer",
        ),
        (
            "Robust + FGSM + Sanitizer",
            robust_agent,
            FGSM(robust_agent),
            "sanitizer",
        ),
        ("Baseline + PGD", baseline_agent, PGD(baseline_agent), None),
        ("Robust + PGD", robust_agent, PGD(robust_agent), None),
        (
            "Robust + PGD + Multi-Sensor",
            robust_agent,
            PGD(robust_agent),
            "multi_sensor",
        ),
        ("Baseline + Blur", baseline_agent, MotionBlur(), None),
        (
            "Robust + Blur + Temporal",
            robust_agent,
            MotionBlur(),
            "temporal",
        ),
    ]

    all_results: List[Dict[str, float]] = []
    print("Running comprehensive evaluation...")
    for name, agent, attack, defense in configurations:
        print(f"Evaluating: {name}")
        result = evaluate_configuration(env, agent, attack, defense)
        result["name"] = name
        print(
            f"  Success={result['success_rate']:.2%} "
            f"Collision={result['collision_rate']:.2%} "
            f"Reward={result['avg_reward']:.2f}"
        )
        all_results.append(result)

    env.close()

    df = pd.DataFrame(all_results)
    df.to_csv("data/comprehensive_results.csv", index=False)
    print("Results saved to data/comprehensive_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    indices = np.arange(len(df))

    axes[0, 0].bar(indices, df["success_rate"])
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_xticks(indices, df["name"], rotation=45, ha="right")

    axes[0, 1].bar(indices, df["collision_rate"], color="red")
    axes[0, 1].set_title("Collision Rate")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_xticks(indices, df["name"], rotation=45, ha="right")

    axes[1, 0].bar(indices, df["avg_reward"], color="green")
    axes[1, 0].set_title("Average Reward")
    axes[1, 0].set_ylabel("Reward")
    axes[1, 0].set_xticks(indices, df["name"], rotation=45, ha="right")

    axes[1, 1].bar(indices, df["avg_length"], color="purple")
    axes[1, 1].set_title("Episode Length")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_xticks(indices, df["name"], rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("data/plots/comprehensive_evaluation.png", dpi=300)
    plt.close()
    print("Plots saved to data/plots/comprehensive_evaluation.png")


if __name__ == "__main__":
    main()


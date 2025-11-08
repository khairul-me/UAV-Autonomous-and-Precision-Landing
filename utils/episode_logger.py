"""
Comprehensive logging utilities for enhanced training runs.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class EpisodeLogger:
    """Log per-episode telemetry, images, and optional sensor artefacts."""

    def __init__(self, log_dir: str = "logs/episodes"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id: Optional[int] = None
        self.episode_data: Optional[Dict[str, Any]] = None
        self.step_data: List[Dict[str, Any]] = []

        print(f"[OK] EpisodeLogger initialised at {self.log_dir}")

    def start_episode(self, episode_id: int, metadata: Dict[str, Any]):
        self.episode_id = episode_id
        self.step_data = []
        self.episode_data = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "steps": [],
        }
        print(f"[OK] Episode {episode_id} logging started")

    def log_step(self, step_num: int, data: Dict[str, Any]):
        if self.episode_data is None:
            raise RuntimeError("start_episode must be called before logging steps")

        record = {
            "step": step_num,
            "timestamp": datetime.now().isoformat(),
            "position": data.get("position", [0.0, 0.0, 0.0]),
            "velocity": data.get("velocity", [0.0, 0.0, 0.0]),
            "orientation": data.get("orientation", [0.0, 0.0, 0.0]),
            "action": data.get("action", [0.0, 0.0, 0.0, 0.0]),
            "reward": data.get("reward", 0.0),
            "goal_distance": data.get("goal_distance", 0.0),
            "closest_obstacle": data.get("closest_obstacle", 100.0),
            "sensor_health": data.get("sensor_health", {}),
            "q_values": data.get("q_values"),
            "observation_dim": len(data.get("observation", [])) if "observation" in data else 0,
        }

        self.step_data.append(record)
        self.episode_data["steps"].append(record)

        if "depth_images" in data:
            self._save_images(step_num, data["depth_images"], "depth")
        if "rgb_images" in data:
            self._save_images(step_num, data["rgb_images"], "rgb")
        if "lidar_points" in data:
            self._save_lidar(step_num, data["lidar_points"])

    def end_episode(self, success: bool, collision: bool, final_distance: float, total_reward: float):
        if self.episode_data is None or self.episode_id is None:
            print("[WARN] No active episode to end")
            return

        self.episode_data["summary"] = {
            "success": success,
            "collision": collision,
            "final_distance": final_distance,
            "total_reward": total_reward,
            "num_steps": len(self.step_data),
            "duration": self._compute_duration(),
        }
        self.episode_data["statistics"] = self._compute_statistics()

        self._save_episode()
        print(
            f"[OK] Episode {self.episode_id} logged "
            f"(success={success}, collision={collision}, reward={total_reward:.2f}, steps={len(self.step_data)})"
        )

        self.episode_id = None
        self.episode_data = None
        self.step_data = []

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _episode_dir(self) -> Path:
        if self.episode_id is None:
            raise RuntimeError("Episode directory requested without active episode")
        directory = self.log_dir / f"episode_{self.episode_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _save_images(self, step_num: int, images: Dict[str, np.ndarray], image_type: str):
        directory = self._episode_dir() / f"{image_type}_images"
        directory.mkdir(parents=True, exist_ok=True)

        for camera_name, image in images.items():
            filename = directory / f"step_{step_num:04d}_{camera_name}.png"
            if image_type == "depth":
                depth_vis = np.clip(image, 0, 100)
                depth_vis = (depth_vis / 100.0 * 255).astype(np.uint8)
                cv2.imwrite(str(filename), depth_vis)
            else:
                cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _save_lidar(self, step_num: int, points: np.ndarray):
        directory = self._episode_dir() / "lidar"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"step_{step_num:04d}.npy", points)

    def _save_episode(self):
        directory = self._episode_dir()

        json_file = directory / "episode_data.json"
        with open(json_file, "w", encoding="utf-8") as handle:
            json.dump(self.episode_data, handle, indent=2, default=str)

        pickle_file = directory / "episode_data.pkl"
        with open(pickle_file, "wb") as handle:
            pickle.dump(self.episode_data, handle)

    # ------------------------------------------------------------------ #
    # Stats helpers
    # ------------------------------------------------------------------ #
    def _compute_duration(self) -> float:
        if len(self.step_data) < 2:
            return 0.0
        start = datetime.fromisoformat(self.step_data[0]["timestamp"])
        end = datetime.fromisoformat(self.step_data[-1]["timestamp"])
        return float((end - start).total_seconds())

    def _compute_statistics(self) -> Dict[str, Any]:
        if not self.step_data:
            return {}

        positions = np.array([entry["position"] for entry in self.step_data], dtype=np.float32)
        velocities = np.array([entry["velocity"] for entry in self.step_data], dtype=np.float32)
        rewards = np.array([entry["reward"] for entry in self.step_data], dtype=np.float32)
        goal_distances = np.array([entry["goal_distance"] for entry in self.step_data], dtype=np.float32)
        obstacle_distances = np.array([entry["closest_obstacle"] for entry in self.step_data], dtype=np.float32)

        if len(positions) > 1:
            diffs = np.diff(positions, axis=0)
            path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            path_length = 0.0

        speeds = np.linalg.norm(velocities, axis=1)

        stats = {
            "path_length": path_length,
            "avg_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
            "avg_reward": float(np.mean(rewards)),
            "min_goal_distance": float(np.min(goal_distances)),
            "closest_obstacle": float(np.min(obstacle_distances)),
            "reward_sum": float(np.sum(rewards)),
            "position_range": {
                "x": [float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))],
                "y": [float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))],
                "z": [float(np.min(positions[:, 2])), float(np.max(positions[:, 2]))],
            },
        }
        return stats


class TrainingLogger:
    """Track high-level metrics across training episodes and evaluations."""

    def __init__(self, log_dir: str = "logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_data = {"start_time": datetime.now().isoformat(), "episodes": [], "evaluations": []}

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_window: List[float] = []

        print(f"[OK] TrainingLogger initialised at {self.log_dir}")

    def log_episode(self, episode: int, data: Dict[str, Any]):
        record = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "reward": data.get("reward", 0.0),
            "length": data.get("length", 0),
            "success": data.get("success", False),
            "collision": data.get("collision", False),
            "goal_distance": data.get("goal_distance", 0.0),
        }

        self.training_data["episodes"].append(record)
        self.episode_rewards.append(record["reward"])
        self.episode_lengths.append(record["length"])
        self.success_window.append(1.0 if record["success"] else 0.0)

        if len(self.success_window) > 100:
            self.success_window.pop(0)
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)

        if episode % 10 == 0:
            self.save()

    def log_evaluation(self, episode: int, eval_data: Dict[str, Any]):
        record = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "success_rate": eval_data.get("success_rate", 0.0),
            "avg_reward": eval_data.get("avg_reward", 0.0),
            "avg_length": eval_data.get("avg_length", 0),
            "collision_rate": eval_data.get("collision_rate", 0.0),
        }
        self.training_data["evaluations"].append(record)
        self.save()

        print("\n" + "=" * 60)
        print(f"EVALUATION @ Episode {episode}")
        print("=" * 60)
        print(f"Success rate   : {record['success_rate'] * 100:.1f}%")
        print(f"Avg reward     : {record['avg_reward']:.2f}")
        print(f"Avg length     : {record['avg_length']:.1f}")
        print(f"Collision rate : {record['collision_rate'] * 100:.1f}%")
        print("=" * 60 + "\n")

    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        if not self.episode_rewards:
            return {"avg_reward": 0.0, "avg_length": 0.0, "success_rate": 0.0}

        rewards = self.episode_rewards[-window:]
        lengths = self.episode_lengths[-window:]
        successes = self.success_window[-window:]

        return {
            "avg_reward": float(np.mean(rewards)),
            "avg_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
        }

    def save(self):
        json_file = self.log_dir / "training_log.json"
        with open(json_file, "w", encoding="utf-8") as handle:
            json.dump(self.training_data, handle, indent=2, default=str)

        pickle_file = self.log_dir / "training_log.pkl"
        with open(pickle_file, "wb") as handle:
            pickle.dump(self.training_data, handle)

    def load(self):
        pickle_file = self.log_dir / "training_log.pkl"
        if not pickle_file.exists():
            print("[WARN] No existing training log to load")
            return
        with open(pickle_file, "rb") as handle:
            self.training_data = pickle.load(handle)
        print(f"[OK] Loaded training log with {len(self.training_data['episodes'])} episodes logged")


class AttackLogger:
    """Record properties of adversarial attacks encountered during training."""

    def __init__(self, log_dir: str = "logs/attacks"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.attack_data = {"attacks": []}
        print(f"[OK] AttackLogger initialised at {self.log_dir}")

    def log_attack(self, step: int, attack_type: str, data: Dict[str, Any]):
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "attack_type": attack_type,
            "epsilon": data.get("epsilon", 0.0),
            "success": data.get("success", False),
            "original_action": data.get("original_action", []),
            "attacked_action": data.get("attacked_action", []),
            "perturbation_norm": data.get("perturbation_norm", 0.0),
        }
        self.attack_data["attacks"].append(record)
        if len(self.attack_data["attacks"]) % 100 == 0:
            self.save()

    def save(self):
        json_file = self.log_dir / "attack_log.json"
        with open(json_file, "w", encoding="utf-8") as handle:
            json.dump(self.attack_data, handle, indent=2, default=str)


# ---------------------------------------------------------------------- #
# Diagnostic test harness
# ---------------------------------------------------------------------- #
def test_loggers():
    print("Testing logging utilities...\n")

    # Episode logger
    episode_logger = EpisodeLogger(log_dir="outputs/test_logs/episodes")
    episode_logger.start_episode(episode_id=1, metadata={"goal": [50, 30, -5], "num_obstacles": 70})
    for step in range(5):
        episode_logger.log_step(
            step,
            {
                "position": [step * 2.0, step * 1.5, -5.0],
                "velocity": [2.0, 1.5, 0.0],
                "orientation": [0.0, 0.0, np.pi / 4],
                "action": [2.0, 1.5, 0.0, 0.0],
                "reward": 0.5,
                "goal_distance": 50 - step * 5,
                "closest_obstacle": 10.0,
                "sensor_health": {"camera": True, "lidar": True},
            },
        )
    episode_logger.end_episode(success=True, collision=False, final_distance=5.0, total_reward=2.5)
    print("[OK] EpisodeLogger test complete\n")

    # Training logger
    training_logger = TrainingLogger(log_dir="outputs/test_logs/training")
    for episode in range(10):
        training_logger.log_episode(
            episode,
            {
                "reward": np.random.randn() * 2 + 5,
                "length": int(np.random.randint(100, 250)),
                "success": bool(np.random.rand() > 0.5),
                "collision": bool(np.random.rand() > 0.8),
                "goal_distance": float(np.random.rand() * 10),
            },
        )
    training_logger.log_evaluation(
        10, {"success_rate": 0.7, "avg_reward": 6.1, "avg_length": 180, "collision_rate": 0.2}
    )
    print(f"[OK] Recent stats: {training_logger.get_recent_stats()}\n")

    # Attack logger
    attack_logger = AttackLogger(log_dir="outputs/test_logs/attacks")
    for step in range(5):
        attack_logger.log_attack(
            step,
            "fgsm",
            {
                "epsilon": 0.03,
                "success": bool(np.random.rand() > 0.5),
                "original_action": [1.0, 0.0, 0.0, 0.0],
                "attacked_action": [0.5, 0.5, 0.0, 0.2],
                "perturbation_norm": 0.15,
            },
        )
    attack_logger.save()
    print("[OK] AttackLogger test complete\n")

    print("[OK] All logger tests passed.")


if __name__ == "__main__":
    import numpy as np

    test_loggers()


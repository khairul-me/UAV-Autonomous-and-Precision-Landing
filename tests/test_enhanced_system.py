"""
Comprehensive regression test for the enhanced navigation stack.
"""
from __future__ import annotations

import sys

sys.path.append(".")

from pathlib import Path

import airsim
import cv2
import numpy as np
import torch
import time

from environments.advanced_obstacles import AdvancedObstacleGenerator
from environments.observations import ObservationBuilder
from utils.episode_logger import AttackLogger, EpisodeLogger, TrainingLogger
from utils.multi_camera import MultiCameraFeatureExtractor, MultiCameraManager
from utils.training_monitor import TrainingMonitor


def test_setup():
    print("\n" + "=" * 60)
    print("TEST 1: AirSim connection")
    print("=" * 60)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    print("[OK] Connected to AirSim")
    return client


def test_multi_camera(client: airsim.MultirotorClient):
    print("\n" + "=" * 60)
    print("TEST 2: Multi-camera capture")
    print("=" * 60)
    manager = MultiCameraManager(client)
    images = manager.capture_all()
    print(f"[OK] Captured streams: { {cam: list(streams.keys()) for cam, streams in images.items()} }")
    depth_images = manager.get_depth_images()
    distances = manager.compute_obstacle_distances(depth_images)
    print(f"[OK] Obstacle distances: {distances}")
    return manager, images, depth_images


def test_feature_extractor(depth_images):
    print("\n" + "=" * 60)
    print("TEST 3: Feature extraction")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = MultiCameraFeatureExtractor().to(device)

    depth_tensors = {}
    for cam_name, depth in depth_images.items():
        depth_norm = np.clip(depth, 0, 100) / 100.0
        if cam_name == "front_center":
            resized = cv2.resize(depth_norm, (240, 120))
        elif cam_name in ("front_left", "front_right"):
            resized = cv2.resize(depth_norm, (160, 80))
        elif cam_name == "bottom":
            resized = cv2.resize(depth_norm, (160, 160))
        else:
            resized = depth_norm
        tensor = torch.from_numpy(resized).float().unsqueeze(0).to(device)
        depth_tensors[cam_name] = tensor

    with torch.no_grad():
        features = extractor(depth_tensors)
    print(f"[OK] Feature tensor shape: {features.shape}")
    return extractor


def test_observation_builder(client, manager, extractor):
    print("\n" + "=" * 60)
    print("TEST 4: Observation builder")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = ObservationBuilder(client, manager, extractor, device)
    goal = np.array([50.0, 30.0, -5.0], dtype=np.float32)
    builder.reset(goal)
    obs = builder.build(current_time=1.0)
    print(f"[OK] Observation vector dimension: {obs.dim}")
    return builder


def test_logging_system():
    print("\n" + "=" * 60)
    print("TEST 5: Logging utilities")
    print("=" * 60)
    EpisodeLogger(log_dir="outputs/test_logs/episodes")
    TrainingLogger(log_dir="outputs/test_logs/training")
    AttackLogger(log_dir="outputs/test_logs/attacks")
    print("[OK] Logger classes initialised")


def test_obstacle_generation(client):
    print("\n" + "=" * 60)
    print("TEST 6: Advanced obstacle generator")
    print("=" * 60)
    generator = AdvancedObstacleGenerator(client)
    for scenario in ["dprl", "corridor", "sparse"]:
        obstacles = generator.generate_scenario(
            scenario,
            save_path=f"outputs/test_obstacles_{scenario}.json",
        )
        print(f"[OK] Scenario '{scenario}' generated ({len(obstacles)} obstacles)")


def test_training_monitor():
    print("\n" + "=" * 60)
    print("TEST 7: Training monitor")
    print("=" * 60)
    monitor = TrainingMonitor(update_interval=0.5)
    monitor.set_obstacles([{"x": 0.0, "y": 0.0, "radius": 2.0}])
    monitor.start()
    try:
        for episode in range(3):
            for step in range(10):
                monitor.update_step({"position": [step * 0.1, step * 0.05, -5], "action": [0.1, 0.0, 0.0, 0.0]})
            monitor.update_episode(
                episode,
                {
                    "reward": episode * 1.0,
                    "length": 10,
                    "success": True,
                    "collision": False,
                    "q_value": 10.0,
                    "loss": 0.1,
                },
            )
            time.sleep(0.2)
    finally:
        monitor.stop()
    print("[OK] Training monitor smoke test complete")


def run_all_tests():
    Path("outputs").mkdir(exist_ok=True)

    client = test_setup()
    manager, images, depth_images = test_multi_camera(client)

    extractor = test_feature_extractor(depth_images)
    builder = test_observation_builder(client, manager, extractor)

    test_logging_system()
    test_obstacle_generation(client)
    test_training_monitor()

    client.armDisarm(False)
    client.enableApiControl(False)

    print("\n" + "=" * 60)
    print("ALL ENHANCED SYSTEM TESTS PASSED")
    print("=" * 60)
    print("Ready for enhanced training runs.")


if __name__ == "__main__":
    import time

    run_all_tests()


"""
Advanced obstacle generator supporting multiple scenario layouts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import airsim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


class AdvancedObstacleGenerator:
    """Generate and persist configurable obstacle layouts."""

    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        print("[OK] AdvancedObstacleGenerator initialised")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_scenario(self, scenario_type: str, save_path: Optional[str] = None) -> List[Dict[str, float]]:
        scenario_type = scenario_type.lower()
        if scenario_type == "dprl":
            obstacles = self._generate_dprl()
        elif scenario_type == "corridor":
            obstacles = self._generate_corridor()
        elif scenario_type == "forest":
            obstacles = self._generate_forest()
        elif scenario_type == "urban":
            obstacles = self._generate_urban()
        elif scenario_type == "sparse":
            obstacles = self._generate_sparse()
        elif scenario_type == "dense":
            obstacles = self._generate_dense()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        print(f"[OK] Generated scenario '{scenario_type}' with {len(obstacles)} obstacles")

        if save_path:
            self._save_obstacles(obstacles, save_path)
            print(f"[OK] Saved obstacle configuration to {save_path}")

        return obstacles

    def load_obstacles(self, path: str) -> List[Dict[str, float]]:
        with open(path, "r", encoding="utf-8") as handle:
            obstacles = json.load(handle)
        print(f"[OK] Loaded {len(obstacles)} obstacles from {path}")
        return obstacles

    def visualize_scenario(self, obstacles: List[Dict[str, float]], save_path: Optional[str] = None):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        for obs in obstacles:
            if obs["type"] == "cylinder":
                ax.add_patch(Circle((obs["x"], obs["y"]), obs["radius"], color="red", alpha=0.4))
            elif obs["type"] == "box":
                ax.add_patch(
                    Rectangle(
                        (obs["x"] - obs["width"] / 2, obs["y"] - obs["length"] / 2),
                        obs["width"],
                        obs["length"],
                        color="blue",
                        alpha=0.3,
                    )
                )

        ax.plot(0, 0, "go", markersize=10, label="Start")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Obstacle layout ({len(obstacles)} obstacles)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[OK] Saved visualisation to {save_path}")
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------ #
    # Scenario generators
    # ------------------------------------------------------------------ #
    def _generate_dprl(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        num_obstacles = 70
        ring_radius = 60.0

        for idx in range(num_obstacles):
            angle = 2 * np.pi * idx / num_obstacles
            radius = float(np.random.uniform(1.5, 3.5))
            height = float(np.random.uniform(10.0, 20.0))
            obstacles.append(
                {
                    "x": float(ring_radius * np.cos(angle)),
                    "y": float(ring_radius * np.sin(angle)),
                    "z": -height / 2,
                    "radius": radius,
                    "height": height,
                    "type": "cylinder",
                }
            )
        return obstacles

    def _generate_corridor(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        length = 100.0
        width = 15.0
        sections = 10

        for idx in range(sections):
            centre = idx * (length / sections)
            for _ in range(3):
                obstacles.append(
                    {
                        "x": float(centre + np.random.uniform(-2, 2)),
                        "y": float(-width / 2 + np.random.uniform(-1, 1)),
                        "z": -float(np.random.uniform(8, 15) / 2),
                        "radius": float(np.random.uniform(1.5, 2.5)),
                        "height": float(np.random.uniform(8, 15)),
                        "type": "cylinder",
                    }
                )
                obstacles.append(
                    {
                        "x": float(centre + np.random.uniform(-2, 2)),
                        "y": float(width / 2 + np.random.uniform(-1, 1)),
                        "z": -float(np.random.uniform(8, 15) / 2),
                        "radius": float(np.random.uniform(1.5, 2.5)),
                        "height": float(np.random.uniform(8, 15)),
                        "type": "cylinder",
                    }
                )
        return obstacles

    def _generate_forest(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        num_obstacles = 150
        area = 80.0

        for _ in range(num_obstacles):
            obstacles.append(
                {
                    "x": float(np.random.uniform(-area / 2, area / 2)),
                    "y": float(np.random.uniform(-area / 2, area / 2)),
                    "z": -float(np.random.uniform(12, 25) / 2),
                    "radius": float(np.random.uniform(0.5, 2.0)),
                    "height": float(np.random.uniform(12, 25)),
                    "type": "cylinder",
                }
            )
        return obstacles

    def _generate_urban(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        grid_size = 4
        spacing = 20.0

        for ix in range(-grid_size, grid_size + 1):
            for iy in range(-grid_size, grid_size + 1):
                if abs(ix) < 2 and abs(iy) < 2:
                    continue
                width = float(np.random.uniform(8, 15))
                length = float(np.random.uniform(8, 15))
                height = float(np.random.uniform(15, 40))
                obstacles.append(
                    {
                        "x": float(ix * spacing),
                        "y": float(iy * spacing),
                        "z": -height / 2,
                        "width": width,
                        "length": length,
                        "height": height,
                        "type": "box",
                    }
                )
        return obstacles

    def _generate_sparse(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        area = 70.0

        for _ in range(20):
            radius = float(np.random.uniform(2.0, 4.0))
            height = float(np.random.uniform(10.0, 15.0))
            obstacles.append(
                {
                    "x": float(np.random.uniform(-area / 2, area / 2)),
                    "y": float(np.random.uniform(-area / 2, area / 2)),
                    "z": -height / 2,
                    "radius": radius,
                    "height": height,
                    "type": "cylinder",
                }
            )
        return obstacles

    def _generate_dense(self) -> List[Dict[str, float]]:
        obstacles: List[Dict[str, float]] = []
        area = 80.0

        for _ in range(200):
            radius = float(np.random.uniform(1.0, 3.0))
            height = float(np.random.uniform(8.0, 20.0))
            obstacles.append(
                {
                    "x": float(np.random.uniform(-area / 2, area / 2)),
                    "y": float(np.random.uniform(-area / 2, area / 2)),
                    "z": -height / 2,
                    "radius": radius,
                    "height": height,
                    "type": "cylinder",
                }
            )
        return obstacles

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _save_obstacles(obstacles: List[Dict[str, float]], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obstacles, handle, indent=2)


# ---------------------------------------------------------------------- #
# Diagnostic script
# ---------------------------------------------------------------------- #
def test_scenarios():
    print("Testing AdvancedObstacleGenerator...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    generator = AdvancedObstacleGenerator(client)
    scenarios = ["dprl", "corridor", "forest", "urban", "sparse", "dense"]

    Path("outputs").mkdir(exist_ok=True)

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario} ---")
        obstacles = generator.generate_scenario(scenario, save_path=f"outputs/obstacles_{scenario}.json")
        generator.visualize_scenario(obstacles, save_path=f"outputs/obstacles_{scenario}_layout.png")

    print("\n[OK] Scenario generation tests completed")


if __name__ == "__main__":
    test_scenarios()


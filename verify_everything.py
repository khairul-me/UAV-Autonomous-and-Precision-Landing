"""
COMPREHENSIVE SYSTEM VERIFICATION

Run this script to exercise the enhanced AirSim navigation stack end-to-end.
It produces a detailed report (JSON + Markdown) under ./verification_output.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import airsim
except ImportError:
    airsim = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class SystemVerifier:
    """Runs staged checks and collates outputs into a summary report."""

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
        }
        self.output_dir = PROJECT_ROOT / "verification_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(" " * 22 + "COMPREHENSIVE SYSTEM VERIFICATION")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Timestamp       : {self.results['timestamp']}\n")

        self.client = None
        self.manager = None
        self.extractor = None

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_all_tests(self) -> None:
        self.test_airsim_connection()
        self.test_configuration()
        self.test_multi_camera()
        self.test_feature_extraction()
        self.test_observation_builder()
        self.test_logging_system()
        self.test_obstacle_generation()
        self.test_file_structure()
        self.generate_report()

    # ------------------------------------------------------------------ #
    # Individual tests
    # ------------------------------------------------------------------ #
    def test_airsim_connection(self):
        name = "airsim_connection"
        result = self._init_result()
        print(self._header("TEST 1: AirSim Connection"))

        if airsim is None:
            result["status"] = "FAILED"
            result["errors"].append("airsim package not installed")
        else:
            try:
                client = airsim.MultirotorClient()
                client.confirmConnection()
                client.enableApiControl(True)
                state = client.getMultirotorState()

                result["details"] = {
                    "connected": True,
                    "api_control": True,
                    "position": [
                        state.kinematics_estimated.position.x_val,
                        state.kinematics_estimated.position.y_val,
                        state.kinematics_estimated.position.z_val,
                    ],
                    "landed_state": str(state.landed_state),
                }
                self.client = client
                print("[OK] AirSim connection established")
                result["status"] = "PASSED"
            except Exception as exc:  # pylint: disable=broad-except
                result["status"] = "FAILED"
                result["errors"].append(str(exc))
                print(f"[FAIL] {exc}")
                traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_configuration(self):
        name = "configuration"
        result = self._init_result()
        print(self._header("TEST 2: Configuration Files"))

        candidate_paths = [
            Path.home() / "Documents" / "AirSim" / "settings.json",
            PROJECT_ROOT / "settings_enhanced.json",
            PROJECT_ROOT / "settings.json",
        ]

        try:
            for path in candidate_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as handle:
                        settings = json.load(handle)
                    result["details"]["settings_path"] = str(path)
                    result["details"]["vehicles"] = list(settings.get("Vehicles", {}).keys())
                    result["details"]["num_cameras"] = sum(
                        len(vehicle.get("Cameras", {})) for vehicle in settings.get("Vehicles", {}).values()
                    )
                    result["details"]["num_sensors"] = sum(
                        len(vehicle.get("Sensors", {})) for vehicle in settings.get("Vehicles", {}).values()
                    )
                    print(f"[OK] Found settings file at {path}")
                    result["status"] = "PASSED"
                    break
            else:
                raise FileNotFoundError("No settings file found in expected locations")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_multi_camera(self):
        name = "multi_camera"
        result = self._init_result()
        print(self._header("TEST 3: Multi-Camera System"))

        if self.client is None:
            result["status"] = "SKIPPED"
            result["errors"].append("AirSim connection unavailable")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        if cv2 is None:
            result["status"] = "FAILED"
            result["errors"].append("opencv-python not installed")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        try:
            from utils.multi_camera import MultiCameraManager

            manager = MultiCameraManager(self.client)
            images = manager.capture_all()
            depth_images = manager.get_depth_images()
            distances = manager.compute_obstacle_distances(depth_images)

            result["details"]["cameras"] = {
                cam: list(streams.keys()) for cam, streams in images.items()
            }
            result["details"]["obstacle_distances"] = distances

            # Save sample imagery
            for cam_name, streams in images.items():
                for stream_name, image in streams.items():
                    filename = self.output_dir / f"sample_{cam_name}_{stream_name}.png"
                    if stream_name == "depth":
                        depth_vis = np.clip(image, 0, 100)
                        depth_vis = (depth_vis / 100.0 * 255).astype(np.uint8)
                        cv2.imwrite(str(filename), depth_vis)
                    elif stream_name == "rgb":
                        cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(str(filename), image)

            composite = manager.visualize_multi_view(images)
            cv2.imwrite(str(self.output_dir / "multi_camera_composite.png"), composite)

            self.manager = manager
            result["status"] = "PASSED"
            print("[OK] Multi-camera capture successful")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")
            traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_feature_extraction(self):
        name = "feature_extraction"
        result = self._init_result()
        print(self._header("TEST 4: Feature Extraction"))

        if self.manager is None:
            result["status"] = "SKIPPED"
            result["errors"].append("MultiCameraManager unavailable")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        if torch is None:
            result["status"] = "FAILED"
            result["errors"].append("PyTorch not installed")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        try:
            from utils.multi_camera import MultiCameraFeatureExtractor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = MultiCameraFeatureExtractor().to(device)
            depth_images = self.manager.get_depth_images()

            depth_tensors = {}
            for cam, depth in depth_images.items():
                depth_norm = np.clip(depth, 0, 100) / 100.0
                tensor = torch.from_numpy(depth_norm).float().unsqueeze(0).to(device)
                depth_tensors[cam] = tensor

            with torch.no_grad():
                features = extractor(depth_tensors)

            result["details"]["feature_shape"] = list(features.shape)
            result["details"]["feature_stats"] = {
                "min": float(features.min()),
                "max": float(features.max()),
                "mean": float(features.mean()),
                "std": float(features.std()),
            }

            self.extractor = extractor
            result["status"] = "PASSED"
            print("[OK] Feature extraction succeeded")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")
            traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_observation_builder(self):
        name = "observation_builder"
        result = self._init_result()
        print(self._header("TEST 5: Observation Builder"))

        if any(getattr(self, attr) is None for attr in ("client", "manager", "extractor")):
            result["status"] = "SKIPPED"
            result["errors"].append("Prerequisite components unavailable")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        try:
            from environments.observations import ObservationBuilder

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            builder = ObservationBuilder(self.client, self.manager, self.extractor, device)
            goal = np.array([50.0, 30.0, -5.0], dtype=np.float32)
            builder.reset(goal)
            obs = builder.build(current_time=1.0)

            vector = obs.to_vector()
            result["details"] = {
                "goal_distance": float(obs.goal_distance),
                "goal_bearing_deg": float(np.degrees(obs.goal_bearing)),
                "sensor_health": obs.sensor_health,
                "vector_dim": obs.dim,
                "vector_min": float(vector.min()),
                "vector_max": float(vector.max()),
            }

            self.obs_builder = builder
            result["status"] = "PASSED"
            print("[OK] Observation vector built")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")
            traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_logging_system(self):
        name = "logging_system"
        result = self._init_result()
        print(self._header("TEST 6: Logging System"))

        try:
            from utils.episode_logger import AttackLogger, EpisodeLogger, TrainingLogger

            base_dir = self.output_dir / "test_logs"

            ep_logger = EpisodeLogger(log_dir=str(base_dir / "episodes"))
            ep_logger.start_episode(1, {"goal": [50, 30, -5]})
            for step in range(3):
                ep_logger.log_step(
                    step,
                    {
                        "position": [step * 1.0, step * 0.5, -5],
                        "velocity": [1.0, 0.5, 0.0],
                        "action": [0.5, 0.0, 0.0, 0.1],
                        "reward": 0.2,
                        "goal_distance": 40 - step * 5,
                        "closest_obstacle": 10.0,
                        "sensor_health": {"camera": True},
                    },
                )
            ep_logger.end_episode(True, False, 5.0, 0.6)

            training_logger = TrainingLogger(log_dir=str(base_dir / "training"))
            training_logger.log_episode(0, {"reward": 1.0, "length": 50, "success": True, "collision": False})

            attack_logger = AttackLogger(log_dir=str(base_dir / "attacks"))
            attack_logger.log_attack(0, "fgsm", {"epsilon": 0.03, "success": True, "perturbation_norm": 0.1})
            attack_logger.save()

            result["details"]["episode_logger"] = "OK"
            result["details"]["training_logger"] = "OK"
            result["details"]["attack_logger"] = "OK"
            result["status"] = "PASSED"
            print("[OK] Logging utilities verified")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")
            traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_obstacle_generation(self):
        name = "obstacle_generation"
        result = self._init_result()
        print(self._header("TEST 7: Obstacle Generation"))

        if self.client is None:
            result["status"] = "SKIPPED"
            result["errors"].append("AirSim connection unavailable")
            self.results["tests"][name] = result
            self._print_status(result["status"])
            return

        try:
            from environments.advanced_obstacles import AdvancedObstacleGenerator

            generator = AdvancedObstacleGenerator(self.client)
            scenarios = {}
            for label in ["dprl", "corridor", "sparse"]:
                obstacles = generator.generate_scenario(
                    label, save_path=str(self.output_dir / f"obstacles_{label}.json")
                )
                generator.visualize_scenario(
                    obstacles, save_path=str(self.output_dir / f"obstacles_{label}_layout.png")
                )
                scenarios[label] = {"num_obstacles": len(obstacles)}
            result["details"]["scenarios"] = scenarios
            result["status"] = "PASSED"
            print("[OK] Obstacle scenarios generated")
        except Exception as exc:  # pylint: disable=broad-except
            result["status"] = "FAILED"
            result["errors"].append(str(exc))
            print(f"[FAIL] {exc}")
            traceback.print_exc()

        self.results["tests"][name] = result
        self._print_status(result["status"])

    def test_file_structure(self):
        name = "file_structure"
        result = self._init_result()
        print(self._header("TEST 8: File Structure"))

        expected = {
            "utils/multi_camera.py": "Multi-camera manager",
            "environments/observations.py": "Observation builder",
            "utils/episode_logger.py": "Logging system",
            "utils/training_monitor.py": "Training monitor",
            "environments/advanced_obstacles.py": "Advanced obstacle generator",
            "train_enhanced.py": "Enhanced training script",
        }

        found = {}
        missing = []
        for rel_path, description in expected.items():
            path = PROJECT_ROOT / rel_path
            if path.exists():
                found[rel_path] = {"size": path.stat().st_size, "description": description}
            else:
                missing.append(rel_path)

        result["details"]["found"] = found
        result["details"]["missing"] = missing
        result["details"]["coverage"] = f"{len(found)}/{len(expected)}"

        if missing:
            result["status"] = "PARTIAL"
        else:
            result["status"] = "PASSED"

        self.results["tests"][name] = result
        self._print_status(result["status"])

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def generate_report(self):
        summary = self._summarise()
        self.results["summary"] = summary

        json_path = self.output_dir / "verification_report.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"\n[OK] JSON report written to {json_path}")

        md_path = self.output_dir / "VERIFICATION_REPORT.md"
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(self._markdown_report())
        print(f"[OK] Markdown report written to {md_path}")

        self._print_summary(summary)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_result() -> Dict[str, Any]:
        return {"status": "unknown", "details": {}, "errors": []}

    @staticmethod
    def _header(title: str) -> str:
        bar = "=" * 80
        return f"\n{bar}\n{title}\n{bar}"

    @staticmethod
    def _print_status(status: str):
        symbols = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⊘", "PARTIAL": "◐", "unknown": "?"}
        print(f"\nResult: {status}")

    def _summarise(self) -> Dict[str, Any]:
        total = len(self.results["tests"])
        passed = sum(1 for res in self.results["tests"].values() if res["status"] == "PASSED")
        failed = sum(1 for res in self.results["tests"].values() if res["status"] == "FAILED")
        skipped = sum(1 for res in self.results["tests"].values() if res["status"] == "SKIPPED")
        partial = sum(1 for res in self.results["tests"].values() if res["status"] == "PARTIAL")
        success_rate = (passed + partial * 0.5) / total * 100 if total else 0.0
        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "partial": partial,
            "success_rate": success_rate,
        }

    def _markdown_report(self) -> str:
        lines = [
            "# System Verification Report",
            "",
            f"**Generated:** {self.results['timestamp']}",
            "",
            "---",
            "",
            "## Summary",
            "",
        ]
        summary = self.results["summary"]
        lines.extend(
            [
                f"- **Total Tests:** {summary['total_tests']}",
                f"- **Passed:** {summary['passed']} ✓",
                f"- **Failed:** {summary['failed']} ✗",
                f"- **Skipped:** {summary['skipped']} ⊘",
                f"- **Partial:** {summary['partial']} ◐",
                f"- **Success Rate:** {summary['success_rate']:.1f}%",
                "",
                "---",
                "",
                "## Detailed Results",
                "",
            ]
        )

        for name, res in self.results["tests"].items():
            icon = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⊘", "PARTIAL": "◐"}.get(res["status"], "?")
            lines.append(f"### {icon} {name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"**Status:** {res['status']}")
            lines.append("")
            if res["details"]:
                lines.append("**Details:**")
                lines.append("```json")
                lines.append(json.dumps(res["details"], indent=2))
                lines.append("```")
                lines.append("")
            if res["errors"]:
                lines.append("**Errors:**")
                lines.extend([f"- {err}" for err in res["errors"]])
                lines.append("")
            lines.append("---")
            lines.append("")

        lines.append("## Generated Artefacts")
        lines.append("")
        lines.append("Artifacts saved in `verification_output/` include samples, obstacle layouts, and logs.")
        lines.append("")
        return "\n".join(lines)

    def _print_summary(self, summary: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total tests : {summary['total_tests']}")
        print(f"Passed      : {summary['passed']}")
        print(f"Failed      : {summary['failed']}")
        print(f"Skipped     : {summary['skipped']}")
        print(f"Partial     : {summary['partial']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print("=" * 80)


def main() -> bool:
    verifier = SystemVerifier()
    try:
        verifier.run_all_tests()
        return verifier.results["summary"]["success_rate"] >= 70.0
    except KeyboardInterrupt:
        print("\n[WARN] Verification interrupted by user")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[FAIL] Verification aborted: {exc}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


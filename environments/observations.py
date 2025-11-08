"""
Enhanced observation builder combining multi-camera perception, proprioception,
goal geometry, obstacle statistics, temporal context, and sensor consistency
metrics. Produces a 70-dimensional observation vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import airsim
import numpy as np
import torch

import cv2
from utils.multi_camera import MultiCameraFeatureExtractor, MultiCameraManager


@dataclass
class DroneObservation:
    """Structured representation of the enhanced observation."""

    depth_features: np.ndarray
    goal_vector: np.ndarray
    goal_distance: float
    goal_bearing: float
    velocity: np.ndarray
    angular_velocity: np.ndarray
    orientation_euler: np.ndarray
    obstacle_distances: Dict[str, float]
    closest_obstacle_bearing: float
    velocity_history: np.ndarray
    distance_traveled: float
    time_elapsed: float
    depth_lidar_agreement: float
    sensor_health: Dict[str, bool]

    def to_vector(self) -> np.ndarray:
        """Flatten all fields into a single numpy vector (70D)."""
        features = [
            np.asarray(self.depth_features),
            np.asarray(self.goal_vector),
            np.asarray([self.goal_distance]),
            np.asarray([self.goal_bearing]),
            np.asarray(self.velocity),
            np.asarray(self.angular_velocity),
            np.asarray(self.orientation_euler),
        ]

        obstacle_array = np.array(
            [
                self.obstacle_distances.get("front_center", 100.0),
                self.obstacle_distances.get("front_left", 100.0),
                self.obstacle_distances.get("front_right", 100.0),
                self.obstacle_distances.get("back", 100.0),
                self.obstacle_distances.get("bottom", 100.0),
            ],
            dtype=np.float32,
        )
        features.append(obstacle_array)
        features.append(np.asarray([self.closest_obstacle_bearing], dtype=np.float32))

        features.append(self.velocity_history.flatten())
        features.append(np.asarray([self.distance_traveled], dtype=np.float32))
        features.append(np.asarray([self.time_elapsed], dtype=np.float32))
        features.append(np.asarray([self.depth_lidar_agreement], dtype=np.float32))
        features.append(np.asarray([float(self.sensor_health.get("camera", True))], dtype=np.float32))

        flat = np.concatenate([np.atleast_1d(f).astype(np.float32) for f in features])
        return flat

    @property
    def dim(self) -> int:
        return int(self.to_vector().shape[0])


class ObservationBuilder:
    """Build enhanced observations from raw AirSim telemetry."""

    def __init__(
        self,
        client: airsim.MultirotorClient,
        camera_manager: MultiCameraManager,
        feature_extractor: MultiCameraFeatureExtractor,
        device: torch.device,
    ):
        self.client = client
        self.camera_manager = camera_manager
        self.feature_extractor = feature_extractor
        self.device = device

        self.velocity_history = np.zeros((5, 3), dtype=np.float32)
        self.position_history: list[np.ndarray] = []
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.start_position = np.zeros(3, dtype=np.float32)
        self.start_time = 0.0

        print("[OK] ObservationBuilder initialised")

    # ------------------------------------------------------------------ #
    # Episode management
    # ------------------------------------------------------------------ #
    def reset(self, goal_position: np.ndarray):
        """Reset temporal buffers and store the new goal position."""
        self.velocity_history = np.zeros((5, 3), dtype=np.float32)
        self.position_history = []
        self.goal_position = np.asarray(goal_position, dtype=np.float32)
        self.start_position = self._get_position()
        self.start_time = 0.0
        print(f"[OK] ObservationBuilder reset for goal {self.goal_position}")

    def build(self, current_time: float) -> DroneObservation:
        """Capture all sensors and construct the observation structure."""
        depth_images = self.camera_manager.get_depth_images()
        depth_features = self._extract_depth_features(depth_images)

        position = self._get_position()
        velocity_body = self._get_velocity_body()
        orientation_euler = self._get_orientation_euler()
        angular_velocity = self._get_angular_velocity()

        goal_info = self._compute_goal_info(position, orientation_euler[2])
        obstacle_distances = self.camera_manager.compute_obstacle_distances(depth_images)
        closest_obstacle_bearing = self._compute_closest_obstacle_bearing(obstacle_distances)

        self._update_temporal_state(velocity_body, position)
        distance_travelled = self._compute_distance_travelled()

        depth_lidar_agreement = self._compute_sensor_agreement(depth_images)
        sensor_health = self._check_sensor_health()

        return DroneObservation(
            depth_features=depth_features,
            goal_vector=goal_info["vector"],
            goal_distance=goal_info["distance"],
            goal_bearing=goal_info["bearing"],
            velocity=velocity_body,
            angular_velocity=angular_velocity,
            orientation_euler=orientation_euler,
            obstacle_distances=obstacle_distances,
            closest_obstacle_bearing=closest_obstacle_bearing,
            velocity_history=self.velocity_history.copy(),
            distance_traveled=distance_travelled,
            time_elapsed=current_time,
            depth_lidar_agreement=depth_lidar_agreement,
            sensor_health=sensor_health,
        )

    # ------------------------------------------------------------------ #
    # Feature extraction helpers
    # ------------------------------------------------------------------ #
    def _extract_depth_features(self, depth_images: Dict[str, np.ndarray]) -> np.ndarray:
        depth_tensors: Dict[str, torch.Tensor] = {}
        for cam_name, depth in depth_images.items():
            depth_norm = np.clip(depth, 0.0, 100.0) / 100.0

            # Resize depth to manageable sizes before feeding into CNNs.
            if cam_name == "front_center":
                resized = cv2.resize(depth_norm, (240, 120))
            elif cam_name in ("front_left", "front_right"):
                resized = cv2.resize(depth_norm, (160, 80))
            elif cam_name == "bottom":
                resized = cv2.resize(depth_norm, (160, 160))
            else:
                resized = depth_norm

            tensor = torch.from_numpy(resized).float().unsqueeze(0).to(self.device)
            depth_tensors[cam_name] = tensor

        if not depth_tensors:
            raise RuntimeError("No depth images captured for observation")

        with torch.no_grad():
            features = self.feature_extractor(depth_tensors)
        return features.cpu().numpy().flatten()

    # ------------------------------------------------------------------ #
    # Sensor helpers
    # ------------------------------------------------------------------ #
    def _get_position(self) -> np.ndarray:
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _get_velocity_body(self) -> np.ndarray:
        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        orientation = self._get_orientation_euler()
        yaw = orientation[2]

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        vx_body = vel.x_val * cos_yaw + vel.y_val * sin_yaw
        vy_body = -vel.x_val * sin_yaw + vel.y_val * cos_yaw
        vz_body = vel.z_val

        return np.array([vx_body, vy_body, vz_body], dtype=np.float32)

    def _get_orientation_euler(self) -> np.ndarray:
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        roll, pitch, yaw = airsim.to_eularian_angles(orientation)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _get_angular_velocity(self) -> np.ndarray:
        imu = self.client.getImuData()
        ang = imu.angular_velocity
        return np.array([ang.x_val, ang.y_val, ang.z_val], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Goal / obstacle helpers
    # ------------------------------------------------------------------ #
    def _compute_goal_info(self, position: np.ndarray, yaw: float) -> Dict[str, np.ndarray]:
        goal_vector = self.goal_position - position
        distance = float(np.linalg.norm(goal_vector))

        goal_angle_world = np.arctan2(goal_vector[1], goal_vector[0])
        bearing = goal_angle_world - yaw
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))

        return {"vector": goal_vector.astype(np.float32), "distance": distance, "bearing": float(bearing)}

    def _compute_closest_obstacle_bearing(self, obstacle_distances: Dict[str, float]) -> float:
        camera_angles = {
            "front_center": 0.0,
            "front_left": -np.pi / 6,
            "front_right": np.pi / 6,
            "bottom": np.pi / 2,
        }
        min_dist = float("inf")
        closest_angle = 0.0
        for cam_name, distance in obstacle_distances.items():
            if cam_name not in camera_angles:
                continue
            if distance < min_dist:
                min_dist = distance
                closest_angle = camera_angles[cam_name]
        return float(closest_angle)

    # ------------------------------------------------------------------ #
    # Temporal helpers
    # ------------------------------------------------------------------ #
    def _update_temporal_state(self, velocity: np.ndarray, position: np.ndarray):
        self.velocity_history[1:] = self.velocity_history[:-1]
        self.velocity_history[0] = velocity
        self.position_history.append(position.copy())

    def _compute_distance_travelled(self) -> float:
        if len(self.position_history) < 2:
            return 0.0
        diffs = np.diff(np.vstack(self.position_history), axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    # ------------------------------------------------------------------ #
    # Sensor consistency
    # ------------------------------------------------------------------ #
    def _compute_sensor_agreement(self, depth_images: Dict[str, np.ndarray]) -> float:
        try:
            lidar_data = self.client.getLidarData()
        except Exception:  # pylint: disable=broad-except
            return 1.0

        if not lidar_data.point_cloud:
            return 1.0
        if "front_center" not in depth_images:
            return 1.0

        depth = depth_images["front_center"]
        h, w = depth.shape
        centre_patch = depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        centre_valid = centre_patch[np.isfinite(centre_patch)]
        if centre_valid.size == 0:
            return 1.0
        depth_median = float(np.median(centre_valid))

        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        forward = points[points[:, 0] > 0]
        if forward.size == 0:
            return 1.0
        lidar_median = float(np.median(np.linalg.norm(forward, axis=1)))

        diff = abs(depth_median - lidar_median)
        agreement = np.exp(-diff / 10.0)
        return float(np.clip(agreement, 0.0, 1.0))

    def _check_sensor_health(self) -> Dict[str, bool]:
        health = {"camera": True, "lidar": True, "imu": True, "gps": True}

        try:
            _ = self.camera_manager.get_depth_images()
        except Exception:  # pylint: disable=broad-except
            health["camera"] = False

        try:
            lidar_data = self.client.getLidarData()
            if not lidar_data.point_cloud:
                health["lidar"] = False
        except Exception:  # pylint: disable=broad-except
            health["lidar"] = False

        try:
            _ = self.client.getImuData()
        except Exception:  # pylint: disable=broad-except
            health["imu"] = False

        try:
            _ = self.client.getGpsData()
        except Exception:  # pylint: disable=broad-except
            health["gps"] = False

        return health


# ---------------------------------------------------------------------- #
# Diagnostic script
# ---------------------------------------------------------------------- #
def test_observation_builder():
    import cv2

    print("Testing ObservationBuilder...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    camera_manager = MultiCameraManager(client)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = MultiCameraFeatureExtractor().to(device)
    builder = ObservationBuilder(client, camera_manager, feature_extractor, device)

    goal = np.array([50.0, 30.0, -5.0], dtype=np.float32)
    builder.reset(goal)

    obs = builder.build(current_time=1.0)
    print("[OK] Observation built")
    print(f"  Depth features: {obs.depth_features.shape}")
    print(f"  Goal distance: {obs.goal_distance:.2f} m")
    print(f"  Goal bearing: {np.degrees(obs.goal_bearing):.1f} deg")
    print(f"  Velocity (body): {obs.velocity}")
    print(f"  Sensor health: {obs.sensor_health}")
    print(f"  Observation vector dim: {obs.dim}")

    # Quick check on vector
    vector = obs.to_vector()
    print(f"  Vector min/max: {vector.min():.3f}/{vector.max():.3f}")

    # Save composite preview for manual review
    depth_images = camera_manager.get_depth_images()
    preview = camera_manager.visualize_multi_view(
        {cam: {"depth": depth} for cam, depth in depth_images.items()}
    )
    cv2.imwrite("outputs/observation_builder_preview.png", preview)
    print("[OK] Preview saved to outputs/observation_builder_preview.png")

    client.armDisarm(False)
    client.enableApiControl(False)
    print("[OK] ObservationBuilder test completed")


if __name__ == "__main__":
    test_observation_builder()


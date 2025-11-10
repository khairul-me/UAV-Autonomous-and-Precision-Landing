from __future__ import annotations

import numpy as np

import airsim


class MultiSensorFusion:
    """Fuse camera, lidar, IMU, and GPS cues to detect attacks."""

    def __init__(self, client: airsim.MultirotorClient) -> None:
        self.client = client

    def detect_attack(self, depth_image: np.ndarray) -> tuple[bool, str]:
        lidar = self.client.getLidarData()
        lidar_points = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

        gps = self.client.getGpsData()
        imu = self.client.getImuData()

        depth_min = float(np.min(depth_image)) if depth_image.size else 100.0

        if lidar_points.size > 0:
            lidar_min = float(np.min(np.linalg.norm(lidar_points, axis=1)))
        else:
            lidar_min = 100.0

        if abs(depth_min - lidar_min) > 2.0:
            return True, "Depth-LiDAR mismatch"

        angular_vel = imu.angular_velocity
        if np.linalg.norm([angular_vel.x_val, angular_vel.y_val, angular_vel.z_val]) > 2.0:
            return True, "High angular velocity anomaly"

        return False, "Nominal"

    def fallback_action(self) -> np.ndarray:
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)


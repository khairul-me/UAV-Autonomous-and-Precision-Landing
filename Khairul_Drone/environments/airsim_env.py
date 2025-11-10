import time
from typing import Dict, Optional, Tuple

import airsim
import gym
import numpy as np
from gym import spaces
import cv2


class AirSimDroneEnv(gym.Env):
    """OpenAI Gym wrapper for AirSim drone navigation."""

    metadata = {"render.modes": []}

    def __init__(self, ip: str = "127.0.0.1") -> None:
        super().__init__()
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()

        self.action_space = spaces.Box(
            low=np.array([-5, -5, -2, -1], dtype=np.float32),
            high=np.array([5, 5, 2, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )

        self.goal = np.array([65, 0, -10], dtype=np.float32)
        self.start_pos = np.array([0, 0, 0], dtype=np.float32)
        self.step_count = 0
        self.max_steps = 500
        self.prev_distance: Optional[float] = None
        self._latest_depth_image: Optional[np.ndarray] = None
        self.depth_shape = (80, 100)

    def reset(self) -> np.ndarray:
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-10, 3).join()

        self.step_count = 0
        self.prev_distance = None
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        vx, vy, vz, yaw_rate = action.astype(float)
        self.client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            0.5,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(is_rate=True, rate=yaw_rate),
        ).join()

        obs = self._get_observation()
        done, info = self._check_done()
        reward = self._calculate_reward(info)

        self.step_count += 1
        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True)]
        )
        depth_raw = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_raw = depth_raw.reshape(144, 256)
        depth_image = cv2.resize(depth_raw, (self.depth_shape[1], self.depth_shape[0]))
        self._latest_depth_image = depth_image

        depth_features = self._extract_depth_features(depth_image)

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation

        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        goal_rel = self.goal - current_pos

        yaw = airsim.to_eularian_angles(orientation)[2]
        goal_yaw = np.arctan2(goal_rel[1], goal_rel[0])
        yaw_diff = goal_yaw - yaw

        self_state = np.array(
            [
                goal_rel[0],
                goal_rel[1],
                goal_rel[2],
                vel.x_val,
                vel.y_val,
                vel.z_val,
                yaw_diff,
                state.kinematics_estimated.angular_velocity.z_val,
            ],
            dtype=np.float32,
        )

        return np.concatenate([depth_features, self_state])

    def _extract_depth_features(self, depth: np.ndarray) -> np.ndarray:
        depth_small = depth[::16, ::20]
        features = depth_small.flatten()[:25]
        features = np.nan_to_num(features, nan=100.0, posinf=100.0)
        return features.astype(np.float32)

    def _check_done(self) -> Tuple[bool, Dict]:
        info: Dict[str, str] = {}
        collision = self.client.simGetCollisionInfo()
        if collision.has_collided:
            info["termination"] = "collision"
            return True, info

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        distance = np.linalg.norm(current_pos - self.goal)

        if distance < 5.0:
            info["termination"] = "success"
            return True, info

        if self.step_count >= self.max_steps:
            info["termination"] = "timeout"
            return True, info

        info["termination"] = "ongoing"
        return False, info

    def _calculate_reward(self, info: Dict) -> float:
        reward = -0.1

        if info["termination"] == "success":
            reward += 10.0
        elif info["termination"] == "collision":
            reward -= 5.0

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        distance = np.linalg.norm(current_pos - self.goal)

        if self.prev_distance is not None:
            reward += (self.prev_distance - distance) * 0.1
        self.prev_distance = distance

        return float(reward)

    def close(self) -> None:
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        finally:
            time.sleep(0.1)

    def get_latest_depth_image(self) -> np.ndarray:
        if self._latest_depth_image is None:
            return np.zeros(self.depth_shape, dtype=np.float32)
        return self._latest_depth_image.copy()


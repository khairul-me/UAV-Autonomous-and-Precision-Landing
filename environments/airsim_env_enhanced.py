# environments/airsim_env_enhanced.py
import airsim
import numpy as np
import cv2

try:
    from gym import spaces
except ImportError:
    # Fallback if gym is not available
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from environments.obstacle_generator import ObstacleGenerator
except ImportError:
    ObstacleGenerator = None

class AirSimDroneEnvEnhanced:
    """
    Enhanced AirSim environment with:
    1. Multi-sensor support (Camera, GPS, IMU, LiDAR)
    2. Sensor noise simulation
    3. Ground truth for privileged learning
    4. Obstacle management
    """
    
    def __init__(self, 
                 ip_address="127.0.0.1",
                 vehicle_name="Drone1",
                 image_shape=(80, 100),
                 max_steps=500,
                 goal_thresh=2.0,
                 collision_thresh=1.0,
                 add_sensor_noise=False,
                 gps_noise_std=0.5,      # meters
                 imu_noise_std=0.05,     # rad/s
                 depth_noise_prob=0.01):  # probability of corrupted pixels
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        self.vehicle_name = vehicle_name
        
        # Environment parameters
        self.image_shape = image_shape
        self.max_steps = max_steps
        self.goal_thresh = goal_thresh
        self.collision_thresh = collision_thresh
        
        # Noise parameters
        self.add_sensor_noise = add_sensor_noise
        self.gps_noise_std = gps_noise_std
        self.imu_noise_std = imu_noise_std
        self.depth_noise_prob = depth_noise_prob
        
        # Obstacle manager
        if ObstacleGenerator is not None:
            self.obstacle_gen = ObstacleGenerator(self.client)
            try:
                self.obstacle_gen.load_obstacles('obstacles.json')
            except:
                print("Warning: Could not load obstacles.json, generating default obstacles")
                self.obstacle_gen.generate_random_obstacles()
                self.obstacle_gen.save_obstacles('obstacles.json')
        else:
            self.obstacle_gen = None
        
        # State/Action spaces (same as before)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, -2.0, -0.3]),
            high=np.array([3.0, 3.0, 2.0, 0.3]),
            shape=(4,),
            dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.start_pos = None
        self.goal_pos = None
        self.prev_dist_to_goal = None
    
    def reset(self, goal_distance=65.0):
        """Reset environment"""
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        self.client.moveToZAsync(-5, velocity=1, vehicle_name=self.vehicle_name).join()
        
        self.start_pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        
        # Generate random goal
        angle = np.random.uniform(0, 2 * np.pi)
        self.goal_pos = airsim.Vector3r(
            self.start_pos.x_val + goal_distance * np.cos(angle),
            self.start_pos.y_val + goal_distance * np.sin(angle),
            -5.0
        )
        
        self.step_count = 0
        self.prev_dist_to_goal = self._get_distance_to_goal()
        
        return self._get_observation()
    
    def _get_observation(self, return_clean=False):
        """
        Get observation with optional sensor noise
        
        Args:
            return_clean: If True, return both noisy and clean observations
        
        Returns:
            observation dict or (noisy_obs, clean_obs) tuple
        """
        # Get ground truth depth image
        depth_image_clean = self._get_depth_image()
        
        # Get ground truth state
        state = self.client.getMultirotorState(self.vehicle_name)
        gps_data = self.client.getGpsData(vehicle_name=self.vehicle_name)
        imu_data = self.client.getImuData(vehicle_name=self.vehicle_name)
        
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        # Compute self-state (ground truth)
        dx_clean = self.goal_pos.x_val - pos.x_val
        dy_clean = self.goal_pos.y_val - pos.y_val
        dz_clean = self.goal_pos.z_val - pos.z_val
        vx_clean = vel.x_val
        vy_clean = vel.y_val
        vz_clean = vel.z_val
        
        current_yaw = airsim.to_eularian_angles(orientation)[2]
        goal_yaw = np.arctan2(dy_clean, dx_clean)
        yaw_diff_clean = self._normalize_angle(goal_yaw - current_yaw)
        yaw_rate_clean = imu_data.angular_velocity.z_val
        
        self_state_clean = np.array([
            dx_clean, dy_clean, dz_clean,
            vx_clean, vy_clean, vz_clean,
            yaw_diff_clean, yaw_rate_clean
        ], dtype=np.float32)
        
        # Create clean observation
        obs_clean = {
            'depth_image': depth_image_clean,
            'self_state': self_state_clean
        }
        
        # Add noise if enabled
        if self.add_sensor_noise:
            # Noisy depth image
            depth_image_noisy = self._add_depth_noise(depth_image_clean)
            
            # Noisy GPS (affects position estimation)
            gps_noise = np.random.normal(0, self.gps_noise_std, 3)
            dx_noisy = dx_clean + gps_noise[0]
            dy_noisy = dy_clean + gps_noise[1]
            dz_noisy = dz_clean + gps_noise[2]
            
            # Noisy IMU (affects velocity and yaw rate)
            imu_noise = np.random.normal(0, self.imu_noise_std, 4)
            vx_noisy = vx_clean + imu_noise[0]
            vy_noisy = vy_clean + imu_noise[1]
            vz_noisy = vz_clean + imu_noise[2]
            yaw_rate_noisy = yaw_rate_clean + imu_noise[3]
            
            # Recompute yaw diff with noisy position
            goal_yaw_noisy = np.arctan2(dy_noisy, dx_noisy)
            yaw_diff_noisy = self._normalize_angle(goal_yaw_noisy - current_yaw)
            
            self_state_noisy = np.array([
                dx_noisy, dy_noisy, dz_noisy,
                vx_noisy, vy_noisy, vz_noisy,
                yaw_diff_noisy, yaw_rate_noisy
            ], dtype=np.float32)
            
            obs_noisy = {
                'depth_image': depth_image_noisy,
                'self_state': self_state_noisy
            }
            
            if return_clean:
                return obs_noisy, obs_clean
            else:
                return obs_noisy
        else:
            if return_clean:
                return obs_clean, obs_clean
            else:
                return obs_clean
    
    def _add_depth_noise(self, depth_image):
        """Add noise to depth image (salt-and-pepper + Gaussian)"""
        noisy = depth_image.copy()
        
        # Salt-and-pepper noise
        mask = np.random.random(depth_image.shape) < self.depth_noise_prob
        noisy[mask] = np.random.choice([0.0, 1.0], size=mask.sum())
        
        # Gaussian noise
        gaussian_noise = np.random.normal(0, 0.02, depth_image.shape)
        noisy = noisy + gaussian_noise
        noisy = np.clip(noisy, 0, 1)
        
        return noisy.astype(np.float32)
    
    def step(self, action):
        """Execute action (same as before)"""
        vx, vy, vz, yaw_rate = action
        # Convert numpy types to Python native types for msgpack
        vx, vy, vz, yaw_rate = float(vx), float(vy), float(vz), float(yaw_rate)
        
        duration = 0.1
        self.client.moveByVelocityAsync(
            vx, vy, vz, duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.vehicle_name
        ).join()
        
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        
        self.step_count += 1
        
        return observation, reward, done, info
    
    def _compute_reward(self):
        """
        Compute reward using GROUND TRUTH obstacle distance
        This is for privileged learning - we know true obstacle positions
        """
        collision_info = self.client.simGetCollisionInfo(self.vehicle_name)
        dist_to_goal = self._get_distance_to_goal()
        
        # Sparse rewards
        if dist_to_goal < self.goal_thresh:
            return 10.0
        if collision_info.has_collided:
            return -5.0
        
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        if abs(pos.x_val) > 85 or abs(pos.y_val) > 85 or pos.z_val > 0 or pos.z_val < -15:
            return -5.0
        
        # Continuous rewards
        progress = (self.prev_dist_to_goal - dist_to_goal) / 65.0
        self.prev_dist_to_goal = dist_to_goal
        
        dist_to_line = self._get_distance_to_line()
        path_error = np.clip(dist_to_line / 10.0, 0, 1)
        
        # Use ground truth obstacle distance (privileged info!)
        obstacle_penalty = self._get_obstacle_penalty_privileged()
        
        reward = np.clip(5.0 * progress - 0.5 * path_error - 1.0 * obstacle_penalty, -1, 1)
        
        return reward
    
    def _get_obstacle_penalty_privileged(self):
        """
        Calculate obstacle penalty using GROUND TRUTH positions
        This is PRIVILEGED INFORMATION - only available during training!
        """
        if self.obstacle_gen is None:
            # Fallback if obstacle generator not available
            return 0.0
        
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        
        # Get nearest obstacle distance from ground truth
        min_dist = self.obstacle_gen.get_nearest_obstacle_distance(pos)
        
        safe_distance = 4.0
        collision_distance = 1.0
        
        if min_dist < collision_distance:
            return 1.0
        elif min_dist < safe_distance:
            penalty = 1.0 - (min_dist - collision_distance) / (safe_distance - collision_distance)
            return penalty
        else:
            return 0.0
    
    def _get_depth_image(self):
        """Capture depth image from camera"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, 
                               pixels_as_float=True, compress=False)
        ], vehicle_name=self.vehicle_name)
        
        response = responses[0]
        depth_img = airsim.list_to_2d_float_array(
            response.image_data_float,
            response.width,
            response.height
        )
        
        depth_img = cv2.resize(depth_img, (self.image_shape[1], self.image_shape[0]))
        depth_img = np.clip(depth_img / 100.0, 0, 1)
        
        return depth_img.astype(np.float32)
    
    def _get_distance_to_goal(self):
        """Euclidean distance to goal"""
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        dx = self.goal_pos.x_val - pos.x_val
        dy = self.goal_pos.y_val - pos.y_val
        dz = self.goal_pos.z_val - pos.z_val
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _get_distance_to_line(self):
        """Distance from current position to straight line"""
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        
        v1 = np.array([
            pos.x_val - self.start_pos.x_val,
            pos.y_val - self.start_pos.y_val,
            pos.z_val - self.start_pos.z_val
        ])
        
        v2 = np.array([
            self.goal_pos.x_val - self.start_pos.x_val,
            self.goal_pos.y_val - self.start_pos.y_val,
            self.goal_pos.z_val - self.start_pos.z_val
        ])
        
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        projection = np.dot(v1, v2_norm) * v2_norm
        perpendicular = v1 - projection
        
        return np.linalg.norm(perpendicular)
    
    def _check_done(self):
        """Check termination conditions"""
        if self._get_distance_to_goal() < self.goal_thresh:
            return True
        if self.client.simGetCollisionInfo(self.vehicle_name).has_collided:
            return True
        
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        if abs(pos.x_val) > 85 or abs(pos.y_val) > 85 or pos.z_val > 0 or pos.z_val < -15:
            return True
        
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _get_info(self):
        """Return info dict"""
        pos = self.client.getMultirotorState(self.vehicle_name).kinematics_estimated.position
        
        return {
            'position': (pos.x_val, pos.y_val, pos.z_val),
            'goal': (self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val),
            'distance_to_goal': self._get_distance_to_goal(),
            'step': self.step_count,
            'collision': self.client.simGetCollisionInfo(self.vehicle_name).has_collided
        }
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def render(self):
        """Get visualization"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.vehicle_name)
        
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        return img_rgb
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)


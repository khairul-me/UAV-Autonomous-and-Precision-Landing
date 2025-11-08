# environments/airsim_env.py
import airsim
import numpy as np
import cv2

try:
    from gym import spaces
except ImportError:
    # If gym is not available, create a simple spaces module
    class spaces:
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
            def sample(self):
                return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

class AirSimDroneEnv:
    """
    Gym-like wrapper for AirSim drone environment
    Based on DPRL paper's setup
    """
    
    def __init__(self, 
                 ip_address="127.0.0.1",
                 image_shape=(80, 100),  # Resized from 240x320 (like DPRL)
                 max_steps=500,          # From DPRL paper
                 goal_thresh=2.0,        # 2m radius to reach goal
                 collision_thresh=1.0):  # 1m = collision
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        
        # Environment parameters
        self.image_shape = image_shape
        self.max_steps = max_steps
        self.goal_thresh = goal_thresh
        self.collision_thresh = collision_thresh
        
        # State space (matching DPRL paper)
        # Visual: 80x100 depth image -> CNN -> 25D
        # Self-state: [dx, dy, dz, vx, vy, vz, yaw_diff, yaw_rate] = 8D
        # Total: 33D
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(33,), 
            dtype=np.float32
        )
        
        # Action space (4D continuous, like DPRL)
        # [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-3.0, -3.0, -2.0, -0.3]),
            high=np.array([3.0, 3.0, 2.0, 0.3]),
            dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.start_pos = None
        self.goal_pos = None
        self.prev_dist_to_goal = None
    
    def reset(self, goal_distance=65.0):
        """Reset environment and generate random goal"""
        # Reset drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off to initial height
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-5, velocity=1).join()  # 5m altitude
        
        # Set start position
        self.start_pos = self.client.getMultirotorState().kinematics_estimated.position
        
        # Generate random goal on circle (like DPRL paper)
        angle = np.random.uniform(0, 2 * np.pi)
        self.goal_pos = airsim.Vector3r(
            self.start_pos.x_val + goal_distance * np.cos(angle),
            self.start_pos.y_val + goal_distance * np.sin(angle),
            -5.0  # Same altitude
        )
        
        # Initialize tracking
        self.step_count = 0
        self.prev_dist_to_goal = self._get_distance_to_goal()
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Parse action [vx, vy, vz, yaw_rate]
        vx, vy, vz, yaw_rate = action
        
        # Execute action for 0.1 seconds (10 Hz control, from DPRL)
        duration = 0.1
        
        # Method 1: Velocity control
        self.client.moveByVelocityAsync(
            vx, vy, vz, 
            duration=duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()
        
        # Get new state
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()
        
        self.step_count += 1
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """
        Get state observation (33D vector)
        Following DPRL paper's state design
        """
        # Get depth image
        depth_image = self._get_depth_image()
        
        # Get self-state
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        # Extract self-state (8D)
        dx = self.goal_pos.x_val - pos.x_val
        dy = self.goal_pos.y_val - pos.y_val
        dz = self.goal_pos.z_val - pos.z_val
        vx = vel.x_val
        vy = vel.y_val
        vz = vel.z_val
        
        # Yaw calculations
        current_yaw = airsim.to_eularian_angles(orientation)[2]
        goal_yaw = np.arctan2(dy, dx)
        yaw_diff = self._normalize_angle(goal_yaw - current_yaw)
        
        # Yaw rate from IMU
        imu_data = self.client.getImuData()
        yaw_rate = imu_data.angular_velocity.z_val
        
        self_state = np.array([dx, dy, dz, vx, vy, vz, yaw_diff, yaw_rate], 
                              dtype=np.float32)
        
        # Combine depth features (25D) + self-state (8D) = 33D
        # For now, we'll return both separately and let network handle it
        observation = {
            'depth_image': depth_image,  # (80, 100)
            'self_state': self_state      # (8,)
        }
        
        return observation
    
    def _get_depth_image(self):
        """Capture depth image from front camera"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, 
                               pixels_as_float=True, compress=False)
        ])
        
        response = responses[0]
        
        # Convert to numpy array
        depth_img = airsim.list_to_2d_float_array(
            response.image_data_float,
            response.width,
            response.height
        )
        
        # Resize to target shape (80x100, like DPRL)
        depth_img = cv2.resize(depth_img, 
                              (self.image_shape[1], self.image_shape[0]))
        
        # Normalize to [0, 1]
        depth_img = np.clip(depth_img / 100.0, 0, 1)  # Assume max depth 100m
        
        return depth_img
    
    def _compute_reward(self):
        """
        Compute reward (following DPRL paper design)
        Combination of sparse + continuous rewards
        """
        # Check terminal conditions first
        collision_info = self.client.simGetCollisionInfo()
        dist_to_goal = self._get_distance_to_goal()
        
        # Sparse rewards (end of episode)
        if dist_to_goal < self.goal_thresh:
            return 10.0  # Reached goal!
        
        if collision_info.has_collided:
            return -5.0  # Collision penalty
        
        # Out of bounds check
        pos = self.client.getMultirotorState().kinematics_estimated.position
        if abs(pos.x_val) > 85 or abs(pos.y_val) > 85 or pos.z_val > 0 or pos.z_val < -15:
            return -5.0
        
        # Continuous rewards
        # 1. Progress reward
        progress = (self.prev_dist_to_goal - dist_to_goal) / 65.0  # Normalized
        self.prev_dist_to_goal = dist_to_goal
        
        # 2. Distance error penalty (deviation from straight line)
        dist_to_line = self._get_distance_to_line()
        path_error = np.clip(dist_to_line / 10.0, 0, 1)
        
        # 3. Obstacle proximity penalty
        obstacle_penalty = self._get_obstacle_penalty()
        
        # Combined reward (like DPRL paper)
        reward = np.clip(5.0 * progress - 0.5 * path_error - 1.0 * obstacle_penalty, -1, 1)
        
        return reward
    
    def _get_obstacle_penalty(self):
        """Calculate penalty based on proximity to obstacles"""
        # Get depth image
        depth = self._get_depth_image()
        
        # Find minimum distance to obstacle
        min_dist = np.min(depth) * 100.0  # Convert back to meters
        
        safe_distance = 4.0  # From DPRL paper
        collision_distance = 1.0
        
        if min_dist < collision_distance:
            return 1.0
        elif min_dist < safe_distance:
            penalty = 1.0 - (min_dist - collision_distance) / (safe_distance - collision_distance)
            return penalty
        else:
            return 0.0
    
    def _check_done(self):
        """Check if episode should terminate"""
        # Reached goal
        if self._get_distance_to_goal() < self.goal_thresh:
            return True
        
        # Collision
        if self.client.simGetCollisionInfo().has_collided:
            return True
        
        # Out of bounds
        pos = self.client.getMultirotorState().kinematics_estimated.position
        if abs(pos.x_val) > 85 or abs(pos.y_val) > 85 or pos.z_val > 0 or pos.z_val < -15:
            return True
        
        # Max steps
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _get_distance_to_goal(self):
        """Euclidean distance to goal"""
        pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = self.goal_pos.x_val - pos.x_val
        dy = self.goal_pos.y_val - pos.y_val
        dz = self.goal_pos.z_val - pos.z_val
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _get_distance_to_line(self):
        """Distance from current position to straight line between start and goal"""
        pos = self.client.getMultirotorState().kinematics_estimated.position
        
        # Vector from start to current position
        v1 = np.array([
            pos.x_val - self.start_pos.x_val,
            pos.y_val - self.start_pos.y_val,
            pos.z_val - self.start_pos.z_val
        ])
        
        # Vector from start to goal
        v2 = np.array([
            self.goal_pos.x_val - self.start_pos.x_val,
            self.goal_pos.y_val - self.start_pos.y_val,
            self.goal_pos.z_val - self.start_pos.z_val
        ])
        
        # Distance to line
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        projection = np.dot(v1, v2_norm) * v2_norm
        perpendicular = v1 - projection
        
        return np.linalg.norm(perpendicular)
    
    def _get_info(self):
        """Return additional info for logging"""
        pos = self.client.getMultirotorState().kinematics_estimated.position
        
        return {
            'position': (pos.x_val, pos.y_val, pos.z_val),
            'goal': (self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val),
            'distance_to_goal': self._get_distance_to_goal(),
            'step': self.step_count,
            'collision': self.client.simGetCollisionInfo().has_collided
        }
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def render(self):
        """Get visualization image"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        return img_rgb
    
    def close(self):
        """Cleanup"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


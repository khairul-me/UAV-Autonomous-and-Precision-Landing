"""
Phase 1 Task 1.3: Classical PID Navigation Controller
- PID controller for waypoint navigation
- Obstacle avoidance logic (if obstacle detected → adjust trajectory)
- Maintain safe distance (>2m) from obstacles
- Test scenarios: straight line, obstacle avoidance, complex path following
Success Criteria: 95%+ success rate in clean (no attack) conditions
"""

import airsim
import numpy as np
import time
import math
from typing import List, Tuple, Optional

class PIDController:
    """PID controller for position control"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.5):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])
    
    def compute(self, current_pos: np.ndarray, target_pos: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Compute control output"""
        error = target_pos - current_pos
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset integral and error"""
        self.integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])

class ObstacleDetector:
    """Simple obstacle detector using depth data"""
    def __init__(self, safe_distance: float = 2.0):
        self.safe_distance = safe_distance  # meters
    
    def detect_obstacles(self, depth_array: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect obstacles in depth image
        Returns: (obstacle_detected, obstacle_direction)
        """
        # Simple threshold-based detection
        obstacle_mask = depth_array < self.safe_distance
        obstacle_pixels = np.where(obstacle_mask)
        
        if len(obstacle_pixels[0]) > 0:
            # Find obstacle location (center of mass)
            h, w = depth_array.shape
            y_center = np.mean(obstacle_pixels[0]) / h  # Normalized
            x_center = np.mean(obstacle_pixels[1]) / w  # Normalized
            
            # Convert to direction vector (0.5, 0.5 is center)
            direction = np.array([x_center - 0.5, y_center - 0.5, 0.0])
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            return True, direction
        
        return False, None

class PIDNavigationController:
    """PID-based navigation controller with obstacle avoidance"""
    def __init__(self, client: airsim.MultirotorClient, 
                 safe_distance: float = 2.0,
                 avoidance_force: float = 5.0):
        self.client = client
        self.pid_controller = PIDController(kp=1.5, ki=0.01, kd=0.8)
        self.obstacle_detector = ObstacleDetector(safe_distance=safe_distance)
        self.avoidance_force = avoidance_force
        self.max_velocity = 5.0  # m/s
        self.position_tolerance = 1.0  # meters
    
    def get_current_position(self) -> np.ndarray:
        """Get current drone position"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])
    
    def get_depth_data(self) -> Optional[np.ndarray]:
        """Get depth image"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ])
            if responses[0].image_data_float:
                depth_array = np.array(responses[0].image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(responses[0].height, responses[0].width)
                return depth_array
        except:
            pass
        return None
    
    def navigate_to_waypoint(self, target: np.ndarray, timeout: float = 60.0) -> bool:
        """
        Navigate to waypoint using PID control with obstacle avoidance
        Returns: True if waypoint reached, False if timeout/collision
        """
        start_time = time.time()
        self.pid_controller.reset()
        
        while time.time() - start_time < timeout:
            current_pos = self.get_current_position()
            distance = np.linalg.norm(target - current_pos)
            
            # Check if waypoint reached
            if distance < self.position_tolerance:
                # Hover at waypoint
                self.client.hoverAsync().join()
                return True
            
            # Get depth data for obstacle detection
            depth_array = self.get_depth_data()
            
            # Compute PID control
            control_velocity = self.pid_controller.compute(current_pos, target, dt=0.1)
            
            # Limit velocity
            velocity_norm = np.linalg.norm(control_velocity)
            if velocity_norm > self.max_velocity:
                control_velocity = control_velocity / velocity_norm * self.max_velocity
            
            # Obstacle avoidance
            if depth_array is not None:
                obstacle_detected, obstacle_dir = self.obstacle_detector.detect_obstacles(depth_array)
                if obstacle_detected and obstacle_dir is not None:
                    # Add avoidance force perpendicular to obstacle direction
                    avoidance = np.array([
                        -obstacle_dir[1] * self.avoidance_force,  # Perpendicular X
                        obstacle_dir[0] * self.avoidance_force,   # Perpendicular Y
                        0.0  # No vertical avoidance
                    ])
                    control_velocity += avoidance
                    
                    # Re-normalize
                    velocity_norm = np.linalg.norm(control_velocity)
                    if velocity_norm > self.max_velocity:
                        control_velocity = control_velocity / velocity_norm * self.max_velocity
                    
                    print(f"[AVOIDANCE] Obstacle detected, adjusting trajectory")
            
            # Apply velocity command
            self.client.moveByVelocityAsync(
                float(control_velocity[0]),
                float(control_velocity[1]),
                float(control_velocity[2]),
                0.5  # duration
            )
            
            time.sleep(0.1)
        
        return False  # Timeout
    
    def follow_path(self, waypoints: List[np.ndarray]) -> dict:
        """
        Follow a path of waypoints
        Returns: statistics dictionary
        """
        stats = {
            'waypoints_reached': 0,
            'waypoints_total': len(waypoints),
            'collisions': 0,
            'success': False
        }
        
        for i, waypoint in enumerate(waypoints):
            print(f"\n[Navigation] Waypoint {i+1}/{len(waypoints)}: {waypoint}")
            success = self.navigate_to_waypoint(waypoint, timeout=60.0)
            
            if success:
                stats['waypoints_reached'] += 1
                print(f"[OK] Waypoint {i+1} reached")
            else:
                print(f"[FAIL] Waypoint {i+1} not reached (timeout)")
                break
        
        stats['success'] = (stats['waypoints_reached'] == stats['waypoints_total'])
        stats['success_rate'] = stats['waypoints_reached'] / stats['waypoints_total']
        
        return stats

def test_straight_line_navigation(client: airsim.MultirotorClient):
    """Test 1: Straight line navigation"""
    print("=" * 60)
    print("TEST 1: Straight Line Navigation")
    print("=" * 60)
    
    controller = PIDNavigationController(client)
    
    # Takeoff
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Navigate forward 20m
    waypoints = [
        np.array([20.0, 0.0, -5.0])  # Forward 20m at 5m altitude
    ]
    
    stats = controller.follow_path(waypoints)
    print(f"\n[RESULTS] Success rate: {stats['success_rate']*100:.1f}%")
    
    # Land
    client.landAsync().join()
    return stats['success_rate'] >= 0.95

def test_obstacle_avoidance(client: airsim.MultirotorClient):
    """Test 2: Obstacle avoidance"""
    print("\n" + "=" * 60)
    print("TEST 2: Obstacle Avoidance")
    print("=" * 60)
    
    controller = PIDNavigationController(client, safe_distance=3.0)
    
    # Takeoff
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Path that may encounter obstacles
    waypoints = [
        np.array([10.0, 0.0, -5.0]),
        np.array([10.0, 10.0, -5.0]),
        np.array([0.0, 10.0, -5.0])
    ]
    
    stats = controller.follow_path(waypoints)
    print(f"\n[RESULTS] Success rate: {stats['success_rate']*100:.1f}%")
    
    # Land
    client.landAsync().join()
    return stats['success_rate'] >= 0.95

def test_complex_path(client: airsim.MultirotorClient):
    """Test 3: Complex path following"""
    print("\n" + "=" * 60)
    print("TEST 3: Complex Path Following")
    print("=" * 60)
    
    controller = PIDNavigationController(client)
    
    # Takeoff
    client.takeoffAsync().join()
    time.sleep(2)
    
    # Complex path
    waypoints = [
        np.array([10.0, 0.0, -5.0]),
        np.array([10.0, 10.0, -8.0]),  # Change altitude
        np.array([0.0, 10.0, -5.0]),
        np.array([0.0, 0.0, -5.0])  # Return home
    ]
    
    stats = controller.follow_path(waypoints)
    print(f"\n[RESULTS] Success rate: {stats['success_rate']*100:.1f}%")
    
    # Land
    client.landAsync().join()
    return stats['success_rate'] >= 0.95

def main():
    """Run all navigation tests"""
    print("=" * 60)
    print("PHASE 1 TASK 1.3: PID NAVIGATION CONTROLLER")
    print("=" * 60)
    
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    results = []
    
    try:
        # Run tests
        results.append(("Straight Line", test_straight_line_navigation(client)))
        time.sleep(2)
        results.append(("Obstacle Avoidance", test_obstacle_avoidance(client)))
        time.sleep(2)
        results.append(("Complex Path", test_complex_path(client)))
        
        # Summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        for test_name, success in results:
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {test_name}")
        
        overall_success = all(r[1] for r in results)
        if overall_success:
            print("\n[SUCCESS] All tests passed! 95%+ success rate achieved.")
        else:
            print("\n[WARNING] Some tests failed. Review PID parameters.")
        
    finally:
        # Cleanup
        try:
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            pass

if __name__ == "__main__":
    main()

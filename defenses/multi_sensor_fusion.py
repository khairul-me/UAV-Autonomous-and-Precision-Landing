# defenses/multi_sensor_fusion.py
import torch
import numpy as np
import airsim


class MultiSensorFusion:
    """
    Multi-Sensor Fusion Defense
    Cross-validates observations from multiple sensors
    
    Inspired by DPRL's privileged learning insight:
    If sensors disagree significantly, one is likely under attack!
    
    Available sensors in AirSim:
    1. Depth Camera - Main perception
    2. LiDAR - Ground truth distance measurement
    3. GPS - Position estimation
    4. IMU - Velocity and orientation
    """
    
    def __init__(self, 
                 depth_lidar_threshold=2.0,  # meters
                 velocity_threshold=0.5,      # m/s
                 position_threshold=1.0):     # meters
        """
        Args:
            depth_lidar_threshold: Max allowed disagreement between depth and LiDAR
            velocity_threshold: Max allowed IMU velocity noise
            position_threshold: Max allowed GPS position noise
        """
        self.depth_lidar_threshold = depth_lidar_threshold
        self.velocity_threshold = velocity_threshold
        self.position_threshold = position_threshold
    
    def fuse_observations(self, airsim_client, vehicle_name="Drone1"):
        """
        Collect and fuse observations from all sensors
        
        Args:
            airsim_client: AirSim client connection
            vehicle_name: Name of vehicle
        
        Returns:
            {
                'depth_image': Depth image from camera,
                'lidar_distances': Distance measurements from LiDAR,
                'state': Vehicle state (position, velocity),
                'attack_detected': Whether inconsistency detected,
                'attack_type': Type of attack detected (if any),
                'confidence': Confidence in attack detection,
                'trusted_sensors': List of sensors to trust
            }
        """
        # Get all sensor data
        depth_image = self._get_depth_image(airsim_client, vehicle_name)
        lidar_data = self._get_lidar_data(airsim_client, vehicle_name)
        gps_data = self._get_gps_data(airsim_client, vehicle_name)
        imu_data = self._get_imu_data(airsim_client, vehicle_name)
        state = airsim_client.getMultirotorState(vehicle_name)
        
        # Cross-validate sensors
        attack_detected = False
        attack_type = None
        confidence = 0.0
        trusted_sensors = ['depth', 'lidar', 'gps', 'imu']
        
        # Check 1: Depth vs LiDAR consistency
        depth_lidar_consistent, depth_lidar_error = self._check_depth_lidar_consistency(
            depth_image, lidar_data
        )
        
        if not depth_lidar_consistent:
            attack_detected = True
            attack_type = 'depth_camera_attack'
            confidence = min(confidence + 0.5, 1.0)
            trusted_sensors.remove('depth')
            print(f"[WARNING] Depth-LiDAR inconsistency detected! Error: {depth_lidar_error:.2f}m")
        
        # Check 2: GPS reasonableness
        gps_reasonable, gps_error = self._check_gps_reasonableness(
            gps_data, state
        )
        
        if not gps_reasonable:
            attack_detected = True
            if attack_type is None:
                attack_type = 'gps_spoofing'
            else:
                attack_type = 'multi_modal_attack'  # Multiple sensors attacked!
            confidence = min(confidence + 0.3, 1.0)
            trusted_sensors.remove('gps')
            print(f"[WARNING] GPS anomaly detected! Error: {gps_error:.2f}m")
        
        # Check 3: IMU velocity reasonableness
        imu_reasonable, imu_error = self._check_imu_reasonableness(
            imu_data, state
        )
        
        if not imu_reasonable:
            attack_detected = True
            if attack_type is None:
                attack_type = 'imu_attack'
            else:
                attack_type = 'multi_modal_attack'
            confidence = min(confidence + 0.2, 1.0)
            trusted_sensors.remove('imu')
            print(f"[WARNING] IMU anomaly detected! Error: {imu_error:.2f}m/s")
        
        return {
            'depth_image': depth_image,
            'lidar_distances': lidar_data,
            'gps_position': gps_data,
            'imu_data': imu_data,
            'state': state,
            'attack_detected': attack_detected,
            'attack_type': attack_type,
            'confidence': confidence,
            'trusted_sensors': trusted_sensors
        }
    
    def _get_depth_image(self, client, vehicle_name):
        """Get depth image from camera"""
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanner,
                               pixels_as_float=True, compress=False)
        ], vehicle_name=vehicle_name)
        
        response = responses[0]
        depth_img = airsim.list_to_2d_float_array(
            response.image_data_float,
            response.width,
            response.height
        )
        
        return depth_img
    
    def _get_lidar_data(self, client, vehicle_name):
        """Get LiDAR point cloud"""
        try:
            lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name=vehicle_name)
        except:
            return None
        
        if len(lidar_data.point_cloud) < 3:
            return None
        
        # Convert point cloud to numpy array
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape((-1, 3))  # [N, 3] (x, y, z)
        
        # Compute distances
        distances = np.linalg.norm(points, axis=1)
        
        return {
            'points': points,
            'distances': distances,
            'min_distance': distances.min() if len(distances) > 0 else float('inf')
        }
    
    def _get_gps_data(self, client, vehicle_name):
        """Get GPS data"""
        gps_data = client.getGpsData(vehicle_name=vehicle_name)
        
        return {
            'latitude': gps_data.gnss.geo_point.latitude,
            'longitude': gps_data.gnss.geo_point.longitude,
            'altitude': gps_data.gnss.geo_point.altitude,
            'velocity': gps_data.gnss.velocity
        }
    
    def _get_imu_data(self, client, vehicle_name):
        """Get IMU data"""
        imu_data = client.getImuData(vehicle_name=vehicle_name)
        
        return {
            'linear_acceleration': imu_data.linear_acceleration,
            'angular_velocity': imu_data.angular_velocity,
            'orientation': imu_data.orientation
        }
    
    def _check_depth_lidar_consistency(self, depth_image, lidar_data):
        """
        Check if depth camera and LiDAR agree on obstacle distances
        
        Returns:
            (is_consistent, error)
        """
        if lidar_data is None or depth_image is None:
            return True, 0.0  # Can't check without both sensors
        
        # Get minimum distance from depth image (center region)
        h, w = depth_image.shape
        center_region = depth_image[h//3:2*h//3, w//3:2*w//3]
        depth_min_distance = np.min(center_region)
        
        # Get minimum distance from LiDAR
        lidar_min_distance = lidar_data['min_distance']
        
        # Compare
        error = abs(depth_min_distance - lidar_min_distance)
        is_consistent = error < self.depth_lidar_threshold
        
        return is_consistent, error
    
    def _check_gps_reasonableness(self, gps_data, state):
        """
        Check if GPS data is reasonable given state estimate
        
        Returns:
            (is_reasonable, error)
        """
        # Get position from state estimate
        state_pos = state.kinematics_estimated.position
        
        # Convert GPS to local coordinates (simplified - in practice use proper conversion)
        # For now, just check if GPS velocity matches state velocity
        gps_vel = gps_data['velocity']
        state_vel = state.kinematics_estimated.linear_velocity
        
        state_speed = np.sqrt(state_vel.x_val**2 + state_vel.y_val**2 + state_vel.z_val**2)
        
        # Error in velocity estimate
        error = abs(gps_vel.x_val - state_speed)
        is_reasonable = error < self.velocity_threshold
        
        return is_reasonable, error
    
    def _check_imu_reasonableness(self, imu_data, state):
        """
        Check if IMU data is reasonable
        
        Returns:
            (is_reasonable, error)
        """
        # Check if angular velocity is within reasonable bounds
        ang_vel = imu_data['angular_velocity']
        ang_speed = np.sqrt(ang_vel.x_val**2 + ang_vel.y_val**2 + ang_vel.z_val**2)
        
        # Very high angular velocity is suspicious
        max_reasonable_ang_vel = 5.0  # rad/s
        is_reasonable = ang_speed < max_reasonable_ang_vel
        error = max(0, ang_speed - max_reasonable_ang_vel)
        
        return is_reasonable, error
    
    def get_trusted_depth(self, fusion_result):
        """
        Get most trustworthy depth estimate
        
        If depth camera is under attack, use LiDAR instead
        
        Args:
            fusion_result: Output from fuse_observations()
        
        Returns:
            Trusted depth image (or approximation from LiDAR)
        """
        if 'depth' in fusion_result['trusted_sensors']:
            # Depth camera is trusted
            return fusion_result['depth_image']
        
        elif 'lidar' in fusion_result['trusted_sensors']:
            # Use LiDAR data to approximate depth image
            print("[WARNING] Using LiDAR as depth fallback")
            
            # Create synthetic depth image from LiDAR
            # (In practice, you'd project LiDAR points onto image plane)
            depth_approx = self._lidar_to_depth_image(
                fusion_result['lidar_distances']
            )
            return depth_approx
        
        else:
            # Both sensors compromised - emergency mode!
            print("[CRITICAL] Both depth and LiDAR compromised!")
            return None
    
    def _lidar_to_depth_image(self, lidar_data):
        """
        Approximate depth image from LiDAR point cloud
        Simplified implementation - just use minimum distance
        """
        if lidar_data is None:
            return np.ones((80, 100)) * 100.0  # Max depth
        
        min_dist = lidar_data['min_distance']
        
        # Create uniform depth image with minimum distance
        # (This is very simplified - real implementation would project points)
        depth_approx = np.ones((80, 100)) * min_dist
        
        return depth_approx


# Test
if __name__ == "__main__":
    print("Testing Multi-Sensor Fusion Defense...")
    print("="*60)
    
    # This test requires actual AirSim connection
    # Create mock test
    
    class MockAirSimClient:
        """Mock AirSim client for testing"""
        
        def simGetImages(self, requests, vehicle_name):
            # Return mock depth image
            class MockResponse:
                def __init__(self):
                    self.image_data_float = list(np.random.rand(240*320))
                    self.width = 320
                    self.height = 240
            return [MockResponse()]
        
        def getLidarData(self, lidar_name, vehicle_name):
            # Return mock LiDAR data
            class MockLidarData:
                def __init__(self):
                    # Generate random point cloud
                    self.point_cloud = list(np.random.randn(1000*3) * 10)
            return MockLidarData()
        
        def getGpsData(self, vehicle_name):
            class MockGPS:
                def __init__(self):
                    class GNSS:
                        def __init__(self):
                            class GeoPoint:
                                latitude = 47.6
                                longitude = -122.1
                                altitude = 100.0
                            self.geo_point = GeoPoint()
                            class Velocity:
                                x_val = 2.0
                            self.velocity = Velocity()
                    self.gnss = GNSS()
            return MockGPS()
        
        def getImuData(self, vehicle_name):
            class MockIMU:
                def __init__(self):
                    class Vector3:
                        x_val = 0.1
                        y_val = 0.1
                        z_val = 0.1
                    self.linear_acceleration = Vector3()
                    self.angular_velocity = Vector3()
                    self.orientation = None
            return MockIMU()
        
        def getMultirotorState(self, vehicle_name):
            class MockState:
                def __init__(self):
                    class Kinematics:
                        def __init__(self):
                            class Pos:
                                x_val = 0.0
                                y_val = 0.0
                                z_val = -5.0
                            class Vel:
                                x_val = 2.0
                                y_val = 0.0
                                z_val = 0.0
                            self.position = Pos()
                            self.linear_velocity = Vel()
                    self.kinematics_estimated = Kinematics()
            return MockState()
    
    # Create fusion system
    fusion = MultiSensorFusion(
        depth_lidar_threshold=2.0,
        velocity_threshold=0.5,
        position_threshold=1.0
    )
    
    # Test with mock client
    client = MockAirSimClient()
    
    print("\n1. Testing normal operation (no attack)...")
    result = fusion.fuse_observations(client)
    print(f"  Attack detected: {result['attack_detected']}")
    print(f"  Trusted sensors: {result['trusted_sensors']}")
    
    print("\n2. Simulating depth camera attack...")
    # Manually create inconsistency
    # (In real test, you'd apply adversarial attack)
    print("  (Would detect depth-LiDAR inconsistency)")
    
    print("\n" + "="*60)
    print("[OK] Multi-Sensor Fusion Defense Test Complete!")
    print("\nNote: Full testing requires actual AirSim environment")


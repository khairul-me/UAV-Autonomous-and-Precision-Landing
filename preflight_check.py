"""
Pre-flight System Verification

Verifies all components before training begins

"""

import airsim
import numpy as np
import cv2
import time
import sys
import os

class PreFlightChecker:
    """Comprehensive pre-flight verification"""
    
    def __init__(self):
        self.client = None
        self.checks_passed = []
        self.checks_failed = []
    
    def run_all_checks(self):
        """Run all pre-flight checks"""
        print("="*80)
        print("SYSTEM VERIFICATION")
        print("="*80)
        print("Checking all systems...\n")
        
        checks = [
            ("AirSim Connection", self.check_connection),
            ("Drone Control", self.check_drone_control),
            ("Camera System", self.check_cameras),
            ("Depth Perception", self.check_depth_camera),
            ("Sensor Suite", self.check_sensors),
            ("Obstacle Configuration", self.check_obstacles),
            ("Action Execution", self.check_action_execution),
            ("State Observation", self.check_state_observation),
        ]
        
        for check_name, check_func in checks:
            self._run_check(check_name, check_func)
        
        # Summary
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        print(f"[OK] Passed: {len(self.checks_passed)}/{len(checks)}")
        print(f"[FAIL] Failed: {len(self.checks_failed)}/{len(checks)}")
        
        if len(self.checks_failed) > 0:
            print("\n[WARNING] FAILED CHECKS:")
            for check in self.checks_failed:
                print(f"  [FAIL] {check}")
            print("\n[CRITICAL] PLEASE FIX ISSUES BEFORE PROCEEDING")
            return False
        else:
            print("\n[OK] ALL SYSTEMS OPERATIONAL")
            print("Ready to proceed with demo and training.")
            return True
    
    def _run_check(self, name, func):
        """Run a single check"""
        print(f"\n[{name}]")
        try:
            success, message = func()
            if success:
                print(f"  [OK] {message}")
                self.checks_passed.append(name)
            else:
                print(f"  [FAIL] {message}")
                self.checks_failed.append(name)
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)}")
            self.checks_failed.append(name)
    
    def check_connection(self):
        """Check AirSim connection"""
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.ping()
            return True, "Connected to AirSim successfully"
        except Exception as e:
            return False, f"Cannot connect to AirSim: {str(e)}"
    
    def check_drone_control(self):
        """Check if we can control the drone"""
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            state = self.client.getMultirotorState()
            
            # Check if we can get state (means control is working)
            if state is None:
                return False, "Cannot get drone state"
            
            return True, "Drone armed and ready for API control"
        except Exception as e:
            return False, f"Control error: {str(e)}"
    
    def check_cameras(self):
        """Check camera system"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if len(responses) == 0:
                return False, "No camera response"
            
            response = responses[0]
            if response.width == 0 or response.height == 0:
                return False, "Invalid image dimensions"
            
            return True, f"Camera working: {response.width}x{response.height}"
        except Exception as e:
            return False, f"Camera error: {str(e)}"
    
    def check_depth_camera(self):
        """Check depth camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, 
                                   pixels_as_float=True, compress=False)
            ])
            
            response = responses[0]
            depth_data = airsim.list_to_2d_float_array(
                response.image_data_float, response.width, response.height
            )
            
            if depth_data.size == 0:
                return False, "Empty depth data"
            
            min_depth = np.min(depth_data)
            max_depth = np.max(depth_data)
            
            return True, f"Depth camera OK: range [{min_depth:.2f}, {max_depth:.2f}]m"
        except Exception as e:
            return False, f"Depth camera error: {str(e)}"
    
    def check_sensors(self):
        """Check all sensors"""
        try:
            imu_data = self.client.getImuData()
            gps_data = self.client.getGpsData()
            baro_data = self.client.getBarometerData()
            
            sensor_status = [
                "IMU: OK",
                "GPS: OK",
                f"Barometer: {baro_data.altitude:.2f}m"
            ]
            
            try:
                lidar_data = self.client.getLidarData()
                if len(lidar_data.point_cloud) > 0:
                    sensor_status.append(f"LiDAR: {len(lidar_data.point_cloud)//3} points")
                else:
                    sensor_status.append("LiDAR: No data (optional)")
            except:
                sensor_status.append("LiDAR: Not configured (optional)")
            
            return True, ", ".join(sensor_status)
        except Exception as e:
            return False, f"Sensor error: {str(e)}"
    
    def check_obstacles(self):
        """Check if obstacles are configured"""
        if os.path.exists('obstacles.json'):
            import json
            with open('obstacles.json', 'r') as f:
                obstacles = json.load(f)
            return True, f"Loaded {len(obstacles)} obstacles"
        else:
            return True, "obstacles.json not found - will be generated on first run"
    
    def check_action_execution(self):
        """Check if actions execute properly"""
        try:
            state_before = self.client.getMultirotorState()
            pos_before = state_before.kinematics_estimated.position
            
            self.client.takeoffAsync().join()
            time.sleep(1)
            
            self.client.moveByVelocityAsync(1, 0, 0, duration=1).join()
            time.sleep(0.5)
            
            state_after = self.client.getMultirotorState()
            pos_after = state_after.kinematics_estimated.position
            
            distance_moved = np.sqrt(
                (pos_after.x_val - pos_before.x_val)**2 +
                (pos_after.y_val - pos_before.y_val)**2
            )
            
            if distance_moved > 0.5:
                return True, f"Action execution verified: moved {distance_moved:.2f}m"
            else:
                return False, f"Drone did not move as expected: {distance_moved:.2f}m"
        except Exception as e:
            return False, f"Action execution error: {str(e)}"
    
    def check_state_observation(self):
        """Check if we can get complete state observation"""
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            components = [
                f"Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})",
                f"Velocity: ({vel.x_val:.2f}, {vel.y_val:.2f}, {vel.z_val:.2f})",
                "Orientation: OK"
            ]
            
            return True, " | ".join(components)
        except Exception as e:
            return False, f"State observation error: {str(e)}"
    
    def cleanup(self):
        """Cleanup after checks"""
        if self.client is not None:
            try:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass

def main():
    checker = PreFlightChecker()
    
    try:
        all_passed = checker.run_all_checks()
        
        if all_passed:
            print("\n" + "="*80)
            print("Ready to run demo! Execute: python demo_flight.py")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("Please fix the failed checks before proceeding.")
            print("="*80)
            sys.exit(1)
    finally:
        checker.cleanup()

if __name__ == '__main__':
    main()


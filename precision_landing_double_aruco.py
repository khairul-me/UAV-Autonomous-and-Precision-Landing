# =================================================================
# Precision Landing System Using Double ArUco Markers
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program implements an advanced precision landing system using two ArUco markers
of different sizes. The system switches between markers based on altitude to maintain
optimal tracking throughout the landing sequence.

Key Features:
- Dual ArUco marker detection
- Automatic marker switching based on altitude
- Dynamic marker size adjustment
- Enhanced landing accuracy through marker fusion
- Real-time performance monitoring
"""

import time
import math
import argparse
from typing import Tuple, Optional, List, Dict

import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.video import WebcamVideoStream
import imutils

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from pymavlink import mavutil

class DualMarkerLandingSystem:
    def __init__(self):
        """Initialize the dual marker precision landing system"""
        # Marker configurations
        self.marker_configs = {
            'high_altitude': {
                'id': 129,
                'size': 40,  # cm
                'height_threshold': 7  # meters
            },
            'low_altitude': {
                'id': 72,
                'size': 19,  # cm
                'height_threshold': 4  # meters
            }
        }
        
        self.takeoff_height = 10  # meters
        self.velocity = 0.5      # m/s

        # Camera configuration
        self.camera_config = {
            'resolution': (640, 480),
            'horizontal_fov': 62.2 * (math.pi / 180),  # Pi cam V2
            'vertical_fov': 48.8 * (math.pi / 180)     # Pi cam V2
        }

        # Initialize ArUco detection
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters_create()

        # Initialize camera
        self.cap = WebcamVideoStream(
            src=0,
            width=self.camera_config['resolution'][0],
            height=self.camera_config['resolution'][1]
        ).start()

        # Load camera calibration
        self._load_camera_calibration()

        # Performance metrics
        self.metrics = {
            'found_count': 0,
            'notfound_count': 0,
            'first_run': True,
            'start_time': 0
        }

        # Connect to drone
        self.vehicle = self._connect_vehicle()
        self._configure_precision_landing()

    def _load_camera_calibration(self):
        """Load camera calibration parameters from files"""
        try:
            calib_path = "/home/pi/video2calibration/calibrationFiles/"
            self.camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
            self.camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')
        except Exception as e:
            raise Exception(f"Failed to load camera calibration files: {str(e)}")

    def _connect_vehicle(self):
        """Establish connection with the drone"""
        parser = argparse.ArgumentParser(description='Dual Marker Precision Landing')
        parser.add_argument('--connect', 
                          default='127.0.0.1:14550',
                          help='Vehicle connection target string')
        args = parser.parse_args()
        
        return connect(args.connect, wait_ready=True)

    def _configure_precision_landing(self):
        """Configure drone parameters for precision landing"""
        self.vehicle.parameters['PLND_ENABLED'] = 1
        self.vehicle.parameters['PLND_TYPE'] = 1
        self.vehicle.parameters['PLND_EST_TYPE'] = 0
        self.vehicle.parameters['LAND_SPEED'] = 20

    def get_active_marker_config(self) -> Dict:
        """
        Determine which marker to track based on current altitude
        
        Returns:
            dict: Active marker configuration
        """
        altitude = self.vehicle.location.global_relative_frame.alt
        
        if altitude > self.marker_configs['low_altitude']['height_threshold']:
            return self.marker_configs['high_altitude']
        return self.marker_configs['low_altitude']

    def detect_marker(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect the appropriate ArUco marker based on current altitude
        
        Returns:
            tuple: (x_angle, y_angle, distance) or (None, None, None) if not detected
        """
        frame = self.cap.read()
        frame = cv2.resize(frame, self.camera_config['resolution'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None:
            return None, None, None
            
        active_config = self.get_active_marker_config()
        marker_id = active_config['id']
        marker_size = active_config['size']
        
        # Find the correct marker in detected markers
        for idx, detected_id in enumerate(ids):
            if detected_id == marker_id:
                marker_corners = [corners[idx]]
                
                # Estimate pose
                ret = aruco.estimatePoseSingleMarkers(
                    marker_corners, marker_size,
                    self.camera_matrix, self.camera_distortion
                )
                rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
                
                # Calculate center point
                corner_array = np.array(marker_corners[0][0])
                center_x = np.mean(corner_array[:, 0])
                center_y = np.mean(corner_array[:, 1])
                
                # Calculate angles
                x_angle = (center_x - self.camera_config['resolution'][0]/2) * \
                         (self.camera_config['horizontal_fov']/self.camera_config['resolution'][0])
                y_angle = (center_y - self.camera_config['resolution'][1]/2) * \
                         (self.camera_config['vertical_fov']/self.camera_config['resolution'][1])
                
                return x_angle, y_angle, tvec[2]
        
        return None, None, None

    def send_landing_target(self, x_angle: float, y_angle: float) -> None:
        """
        Send landing target message to drone
        
        Args:
            x_angle (float): Angle to target in x-axis
            y_angle (float): Angle to target in y-axis
        """
        msg = self.vehicle.message_factory.landing_target_encode(
            0,  # time since system boot
            0,  # target num
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            x_angle,
            y_angle,
            0,  # distance to target
            0,  # Target x-axis size
            0   # Target y-axis size
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def arm_and_takeoff(self, target_height: float) -> None:
        """
        Arm the drone and take off to specified height
        
        Args:
            target_height (float): Target altitude in meters
        """
        print("Starting takeoff sequence...")
        
        # Check if vehicle is armable
        while not self.vehicle.is_armable:
            print("Waiting for vehicle to become armable...")
            time.sleep(1)
        print("Vehicle is now armable")

        # Switch to GUIDED mode
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode != 'GUIDED':
            print("Waiting for GUIDED mode...")
            time.sleep(1)
        print("Vehicle now in GUIDED mode")

        # Arm the vehicle
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)
        print("Vehicle armed")

        # Take off
        self.vehicle.simple_takeoff(target_height)
        while True:
            altitude = self.vehicle.location.global_relative_frame.alt
            print(f"Current Altitude: {altitude:.1f}m")
            if altitude >= target_height * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def run_landing_sequence(self) -> None:
        """Execute the precision landing sequence"""
        print("Starting precision landing sequence...")
        
        if self.metrics['first_run']:
            self.metrics['start_time'] = time.time()
            self.metrics['first_run'] = False

        try:
            while self.vehicle.armed:
                # Ensure we're in LAND mode
                if self.vehicle.mode != 'LAND':
                    self.vehicle.mode = VehicleMode("LAND")
                    while self.vehicle.mode != 'LAND':
                        print("Switching to LAND mode...")
                        time.sleep(1)
                    print("Vehicle now in LAND mode")

                # Get active marker configuration
                active_config = self.get_active_marker_config()
                print(f"Tracking marker {active_config['id']} ({active_config['size']}cm)")

                # Detect marker and get landing target
                x_angle, y_angle, distance = self.detect_marker()

                if x_angle is not None:
                    self.send_landing_target(x_angle, y_angle)
                    self.metrics['found_count'] += 1
                    print(f"Target found - Distance: {distance:.2f}m")
                    print(f"Angles (deg) - X: {math.degrees(x_angle):.1f}, Y: {math.degrees(y_angle):.1f}")
                else:
                    self.metrics['notfound_count'] += 1
                    print("Target not found")

                time.sleep(0.1)  # Control loop rate

        except Exception as e:
            print(f"Error during landing sequence: {str(e)}")
        finally:
            self._print_landing_statistics()

    def _print_landing_statistics(self) -> None:
        """Print the landing performance statistics"""
        total_time = time.time() - self.metrics['start_time']
        total_frames = self.metrics['found_count'] + self.metrics['notfound_count']
        
        print("\nLanding Statistics:")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Total frames processed: {total_frames}")
        print(f"Detection rate: {(self.metrics['found_count']/total_frames)*100:.1f}%")
        print(f"Average FPS: {total_frames/total_time:.1f}")
        print(f"Successful detections: {self.metrics['found_count']}")
        print(f"Failed detections: {self.metrics['notfound_count']}")

def main():
    """Main execution function"""
    try:
        # Initialize landing system
        landing_system = DualMarkerLandingSystem()
        print("Dual marker precision landing system initialized")
        
        # Take off
        landing_system.arm_and_takeoff(landing_system.takeoff_height)
        
        # Execute landing sequence
        landing_system.run_landing_sequence()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'landing_system' in locals():
            landing_system.cap.stop()
            landing_system.vehicle.close()
            print("Systems shutdown complete")

if __name__ == '__main__':
    main()

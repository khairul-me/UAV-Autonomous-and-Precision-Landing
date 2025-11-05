# =================================================================
# Precision Landing System Using Single ArUco Marker
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program implements a precision landing system for drones using ArUco marker detection.
It uses computer vision to detect a single ArUco marker and guides the drone to land
precisely on the target location.

Key Features:
- Real-time ArUco marker detection
- Camera calibration integration
- Precision landing control
- Visual feedback processing
- Automated landing sequence
"""

import time
import math
import argparse
from typing import Tuple, Optional

import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.video import WebcamVideoStream
import imutils

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
from pymavlink import mavutil

class PrecisionLandingSystem:
    def __init__(self):
        """Initialize the precision landing system with all necessary parameters"""
        # ArUco configuration
        self.id_to_find = 72
        self.marker_size = 19  # cm
        self.takeoff_height = 8  # meters
        self.velocity = 0.5     # m/s

        # Camera configuration
        self.horizontal_res = 640
        self.vertical_res = 480
        self.horizontal_fov = 62.2 * (math.pi / 180)  # Pi cam V2
        self.vertical_fov = 48.8 * (math.pi / 180)    # Pi cam V2

        # Initialize ArUco detection
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters_create()

        # Initialize camera
        self.cap = WebcamVideoStream(src=0, 
                                   width=self.horizontal_res, 
                                   height=self.vertical_res).start()

        # Load camera calibration
        self._load_camera_calibration()

        # Performance metrics
        self.found_count = 0
        self.notfound_count = 0
        self.first_run = True
        self.start_time = 0

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
        parser = argparse.ArgumentParser(description='Precision Landing Control')
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

    def arm_and_takeoff(self, target_height: float) -> None:
        """
        Arm the drone and take off to specified height
        
        Args:
            target_height (float): Target altitude in meters
        """
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
            print(f"Altitude: {altitude}")
            if altitude >= target_height * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def detect_marker(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect ArUco marker in camera feed and calculate position
        
        Returns:
            tuple: (x_angle, y_angle, distance) or (None, None, None) if not detected
        """
        frame = self.cap.read()
        frame = cv2.resize(frame, (self.horizontal_res, self.vertical_res))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None and self.id_to_find in ids:
            # Calculate marker pose
            ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, 
                                                self.camera_matrix, 
                                                self.camera_distortion)
            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]
            
            # Calculate center point
            marker_corners = corners[0][0]
            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])
            
            # Calculate angles
            x_angle = (center_x - self.horizontal_res/2) * (self.horizontal_fov/self.horizontal_res)
            y_angle = (center_y - self.vertical_res/2) * (self.vertical_fov/self.vertical_res)
            
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
            0,  # Target y-axis size
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def run_landing_sequence(self) -> None:
        """Execute the precision landing sequence"""
        print("Starting precision landing sequence...")
        
        if self.first_run:
            self.start_time = time.time()
            self.first_run = False
        
        while self.vehicle.armed:
            # Ensure we're in LAND mode
            if self.vehicle.mode != 'LAND':
                self.vehicle.mode = VehicleMode("LAND")
                while self.vehicle.mode != 'LAND':
                    print("Waiting for LAND mode...")
                    time.sleep(1)
            
            # Detect marker and get landing target
            x_angle, y_angle, distance = self.detect_marker()
            
            if x_angle is not None:
                self.send_landing_target(x_angle, y_angle)
                self.found_count += 1
                print(f"Target found - Distance: {distance:.2f}m")
            else:
                self.notfound_count += 1
                print("Target not found")
            
            time.sleep(0.1)  # Control loop rate

        # Landing complete - print statistics
        total_time = time.time() - self.start_time
        print(f"\nLanding Statistics:")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Detection rate: {(self.found_count/(self.found_count + self.notfound_count))*100:.1f}%")
        print(f"Average detections per second: {self.found_count/total_time:.1f}")

def main():
    """Main execution function"""
    try:
        # Initialize landing system
        landing_system = PrecisionLandingSystem()
        
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

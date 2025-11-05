# =================================================================
# Autonomous Taco Delivery Drone System
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program implements an autonomous drone delivery system specifically designed
for taco delivery. It combines precision landing capabilities with servo control
for payload delivery.

Key Features:
- Autonomous navigation to delivery coordinates
- Precision landing using ArUco markers
- Servo-controlled payload release mechanism
- Return-to-home functionality
- Real-time delivery status monitoring
"""

import time
import math
import argparse
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.video import WebcamVideoStream

from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

@dataclass
class DeliveryLocation:
    """Data class for delivery location information"""
    latitude: float
    longitude: float
    altitude: float

class TacoDeliverySystem:
    def __init__(self):
        """Initialize the taco delivery system"""
        # Delivery configuration
        self.delivery_config = {
            'takeoff_height': 8,  # meters
            'cruise_velocity': 0.5,  # m/s
            'delivery_wait_time': 10,  # seconds
            'servo_channel': 14,
            'servo_dropoff_pwm': 1900,
            'servo_hold_pwm': 1100
        }

        # Marker configurations for different altitudes
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

        # Camera configuration
        self.camera_config = {
            'resolution': (640, 480),
            'horizontal_fov': 62.2 * (math.pi / 180),  # Pi cam V2
            'vertical_fov': 48.8 * (math.pi / 180)     # Pi cam V2
        }

        # Initialize systems
        self._setup_vision_system()
        self.vehicle = self._connect_vehicle()
        self._configure_landing_parameters()
        
        # Store home location
        self.home_location = None

    def _setup_vision_system(self):
        """Initialize the vision system components"""
        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters_create()

        # Camera setup
        self.cap = WebcamVideoStream(
            src=0,
            width=self.camera_config['resolution'][0],
            height=self.camera_config['resolution'][1]
        ).start()

        # Load camera calibration
        try:
            calib_path = "/home/pi/video2calibration/calibrationFiles/"
            self.camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
            self.camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')
        except Exception as e:
            raise Exception(f"Failed to load camera calibration: {str(e)}")

    def _connect_vehicle(self):
        """Establish connection with the drone"""
        parser = argparse.ArgumentParser(description='Taco Delivery Drone')
        parser.add_argument('--connect', 
                          default='127.0.0.1:14550',
                          help='Vehicle connection target string')
        args = parser.parse_args()
        
        return connect(args.connect, wait_ready=True)

    def _configure_landing_parameters(self):
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
        print("Starting takeoff sequence...")
        
        while not self.vehicle.is_armable:
            print("Waiting for vehicle to become armable...")
            time.sleep(1)
        
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode != 'GUIDED':
            print("Waiting for GUIDED mode...")
            time.sleep(1)
        
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)
            
        print("Taking off...")
        self.vehicle.simple_takeoff(target_height)
        
        while True:
            altitude = self.vehicle.location.global_relative_frame.alt
            print(f"Altitude: {altitude:.1f}m")
            if altitude >= target_height * 0.95:
                print("Reached target altitude")
                break
            time.sleep(1)

    def save_home_location(self):
        """Save current location as home position"""
        self.home_location = LocationGlobalRelative(
            self.vehicle.location.global_relative_frame.lat,
            self.vehicle.location.global_relative_frame.lon,
            self.delivery_config['takeoff_height']
        )
        print("Home location saved")

    def goto_location(self, location: LocationGlobalRelative) -> None:
        """
        Navigate to specified location
        
        Args:
            location (LocationGlobalRelative): Target location
        """
        self.vehicle.simple_goto(location)
        
        while True:
            current_location = self.vehicle.location.global_relative_frame
            distance = self._get_distance_meters(location, current_location)
            print(f"Distance to target: {distance:.1f}m")
            
            if distance < 1.0:
                print("Reached target location")
                break
            time.sleep(1)

    def _get_distance_meters(self, location1: LocationGlobalRelative, 
                           location2: LocationGlobalRelative) -> float:
        """
        Calculate distance between two global locations
        
        Returns:
            float: Distance in meters
        """
        dlat = location2.lat - location1.lat
        dlong = location2.lon - location1.lon
        return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

    def control_servo(self, pwm_value: int) -> None:
        """
        Control the payload release servo
        
        Args:
            pwm_value (int): PWM value to set
        """
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            self.delivery_config['servo_channel'],
            pwm_value,
            0, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)

    def execute_delivery(self, delivery_location: DeliveryLocation) -> None:
        """
        Execute complete delivery sequence
        
        Args:
            delivery_location (DeliveryLocation): Delivery target location
        """
        try:
            # Save home location
            self.save_home_location()
            
            # Prepare payload
            print("Securing payload...")
            self.control_servo(self.delivery_config['servo_hold_pwm'])
            time.sleep(1)
            
            # Take off
            self.arm_and_takeoff(self.delivery_config['takeoff_height'])

            # Navigate to delivery location
            print("Navigating to delivery location...")
            target_location = LocationGlobalRelative(
                delivery_location.latitude,
                delivery_location.longitude,
                self.delivery_config['takeoff_height']
            )
            self.goto_location(target_location)

            # Execute precision landing
            print("Starting precision landing sequence...")
            self.execute_precision_landing()

            # Deliver payload
            print("Delivering payload...")
            self.control_servo(self.delivery_config['servo_dropoff_pwm'])
            print(f"Waiting {self.delivery_config['delivery_wait_time']} seconds...")
            time.sleep(self.delivery_config['delivery_wait_time'])

            # Return home
            print("Returning to home location...")
            self.arm_and_takeoff(self.delivery_config['takeoff_height'])
            self.goto_location(self.home_location)

            # Final landing
            print("Executing final landing...")
            self.execute_precision_landing()

            print("Delivery mission completed successfully!")

        except Exception as e:
            print(f"Error during delivery: {str(e)}")
            # Emergency return to home if possible
            if self.home_location:
                print("Attempting emergency return to home...")
                self.goto_location(self.home_location)
                self.vehicle.mode = VehicleMode("LAND")

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

    def detect_marker(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect appropriate ArUco marker based on current altitude
        
        Returns:
            tuple: (x_angle, y_angle, distance) or (None, None, None) if not detected
        """
        frame = self.cap.read()
        frame = cv2.resize(frame, self.camera_config['resolution'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None:
            return None, None, None
            
        # Determine active marker based on altitude
        altitude = self.vehicle.location.global_relative_frame.alt
        active_config = (
            self.marker_configs['high_altitude']
            if altitude > self.marker_configs['low_altitude']['height_threshold']
            else self.marker_configs['low_altitude']
        )
        
        # Find the correct marker
        marker_found = False
        for idx, detected_id in enumerate(ids):
            if detected_id == active_config['id']:
                marker_found = True
                marker_corners = [corners[idx]]
                
                # Estimate pose
                ret = aruco.estimatePoseSingleMarkers(
                    marker_corners,
                    active_config['size'],
                    self.camera_matrix,
                    self.camera_distortion
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

    def execute_precision_landing(self) -> None:
        """Execute precision landing sequence using ArUco markers"""
        print("Starting precision landing sequence...")
        
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode != 'LAND':
            print("Waiting for LAND mode...")
            time.sleep(1)

        while self.vehicle.armed:
            x_angle, y_angle, distance = self.detect_marker()
            
            if x_angle is not None:
                self.send_landing_target(x_angle, y_angle)
                print(f"Target found - Distance: {distance:.2f}m")
                print(f"Angles (deg) - X: {math.degrees(x_angle):.1f}, Y: {math.degrees(y_angle):.1f}")
            else:
                print("Target not found")
            
            time.sleep(0.1)

def main():
    """Main execution function"""
    # Example delivery coordinates
    delivery_location = DeliveryLocation(
        latitude=46.710140,    # Replace with actual delivery coordinates
        longitude=-92.095865,  # Replace with actual delivery coordinates
        altitude=8.0           # meters
    )

    try:
        # Initialize delivery system
        delivery_system = TacoDeliverySystem()
        print("Taco delivery system initialized")
        
        # Execute delivery mission
        delivery_system.execute_delivery(delivery_location)
        
    except Exception as e:
        print(f"Mission failed: {str(e)}")
    finally:
        if 'delivery_system' in locals():
            delivery_system.cap.stop()
            delivery_system.vehicle.close()
            print("Systems shutdown complete")

if __name__ == '__main__':
    main()

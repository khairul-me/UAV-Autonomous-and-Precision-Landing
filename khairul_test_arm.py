# =================================================================
# Drone Arming Test Program
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program demonstrates basic drone arming and control functionality using DroneKit.
It establishes a connection with a drone, arms it, and performs a simple land operation.
The program serves as a basic test for drone initialization and control systems.

Key Features:
- Drone connection and initialization
- Arming sequence verification
- Mode switching (GUIDED -> LAND)
- Basic safety checks and wait states
"""

import time
import os
import platform
import sys
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil

class DroneController:
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        """
        Initialize the drone controller with connection parameters
        Args:
            connection_string (str): Connection string for the drone (default: UDP localhost)
        """
        self.vehicle = connect(connection_string, wait_ready=True)
        print("Connected to drone successfully")

    def arm_drone(self):
        """
        Arms the drone after performing necessary safety checks
        Returns:
            None
        """
        # Wait for drone to become armable
        while not self.vehicle.is_armable:
            print("Waiting for vehicle to become armable...")
            time.sleep(1)
        print("Vehicle is now armable")

        # Set vehicle mode to GUIDED
        self.vehicle.mode = VehicleMode("GUIDED")
        while self.vehicle.mode != 'GUIDED':
            print("Waiting for drone to enter GUIDED flight mode...")
            time.sleep(1)
        print("Vehicle now in GUIDED MODE")

        # Arm the vehicle
        self.vehicle.armed = True
        while not self.vehicle.armed:
            print("Waiting for vehicle to become armed...")
            time.sleep(1)
        print("Vehicle armed successfully - props are spinning!")

    def send_ned_velocity(self, velocity_x, velocity_y, velocity_z):
        """
        Send velocity command to drone in North-East-Down frame
        Args:
            velocity_x (float): Velocity in North direction (m/s)
            velocity_y (float): Velocity in East direction (m/s)
            velocity_z (float): Velocity in Down direction (m/s)
        """
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,  # time_boot_ms
            0, 0,  # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
            0b0000111111000111,  # type_mask
            0, 0, 0,  # x, y, z positions
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
            0, 0, 0,  # x, y, z acceleration
            0, 0)  # yaw, yaw_rate
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

def main():
    """
    Main execution function
    """
    try:
        # Initialize drone controller
        drone = DroneController()

        # Arm the drone
        drone.arm_drone()
        time.sleep(2)

        # Switch to LAND mode
        drone.vehicle.mode = VehicleMode("LAND")
        while drone.vehicle.mode != 'LAND':
            time.sleep(1)
            print("Waiting for drone to land...")
        print("Landing sequence completed")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'drone' in locals():
            drone.vehicle.close()
            print("Vehicle connection closed")

if __name__ == '__main__':
    main()

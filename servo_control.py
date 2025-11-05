# =================================================================
# Drone Servo Control Program
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program provides functionality for controlling drone servos using DroneKit.
It allows precise control of servo positions through PWM signals, which is essential
for manipulating drone payloads, cameras, or other servo-controlled mechanisms.

Key Features:
- Servo PWM control
- Configurable servo channels
- High and low position presets
- Safety delays between movements
"""

import time
import os
import platform
import sys
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil

class ServoController:
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        """
        Initialize the servo controller with connection parameters
        Args:
            connection_string (str): Connection string for the drone (default: UDP localhost)
        """
        # Connection parameters
        self.vehicle = connect(connection_string, wait_ready=True)
        print("Connected to vehicle successfully")

        # Servo configuration
        self.servo_channel = 14  # Default servo channel
        self.pwm_high = 1900    # Maximum PWM value
        self.pwm_low = 1100     # Minimum PWM value

    def control_servo(self, servo_number, pwm_value):
        """
        Control a specific servo by sending PWM commands
        Args:
            servo_number (int): The servo channel number
            pwm_value (int): PWM value to set (typically between 1000-2000)
        """
        # Input validation
        if not (1000 <= pwm_value <= 2000):
            print("Warning: PWM value outside normal range (1000-2000)")

        # Create command message
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # command
            0,       # confirmation
            servo_number,  # param 1 - servo number
            pwm_value,    # param 2 - PWM value
            0, 0, 0, 0, 0  # remaining params (not used)
        )
        
        # Send command to vehicle
        self.vehicle.send_mavlink(msg)
        print(f"Servo {servo_number} set to PWM value: {pwm_value}")

    def test_servo_range(self):
        """
        Test servo by moving it through its full range
        """
        print("Testing servo range...")
        # Move to high position
        self.control_servo(self.servo_channel, self.pwm_high)
        time.sleep(1)  # Wait for movement to complete
        
        # Move to low position
        self.control_servo(self.servo_channel, self.pwm_low)
        time.sleep(1)  # Wait for movement to complete
        
        print("Servo range test completed")

def main():
    """
    Main execution function
    """
    try:
        # Initialize servo controller
        controller = ServoController()
        
        # Perform servo test
        controller.test_servo_range()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'controller' in locals():
            controller.vehicle.close()
            print("Vehicle connection closed")

if __name__ == '__main__':
    main()

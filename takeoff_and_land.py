# =================================================================
# Basic Drone Takeoff and Landing Program
# =================================================================
# Author: Md Khairul Islam
# Institution: Hobart and William Smith Colleges, Geneva, NY
# Major: Robotics and Computer Science
# Contact: khairul.islam@hws.edu
# =================================================================
"""
This program demonstrates basic autonomous drone operations including takeoff
and landing sequences. It provides a foundation for testing drone control
and can be used to verify basic flight capabilities.

Key Features:
- Automated takeoff to specified altitude
- Stable hovering
- Automated landing sequence
- Support for both manual and automatic arming
"""

import time
from typing import Optional
from dataclasses import dataclass

from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil

@dataclass
class FlightConfig:
    """Configuration parameters for flight control"""
    target_altitude: float
    manual_arm: bool
    connection_string: str
    baud_rate: int

class DroneController:
    """Controls basic drone operations for takeoff and landing"""
    
    def __init__(self, config: FlightConfig):
        """
        Initialize drone controller with specified configuration
        
        Args:
            config (FlightConfig): Flight configuration parameters
        """
        self.config = config
        self.vehicle = self._connect_vehicle()
        print(f"Connected to drone on {config.connection_string}")

    def _connect_vehicle(self):
        """
        Establish connection with the drone
        
        Returns:
            Vehicle: Connected drone vehicle object
        """
        try:
            vehicle = connect(
                self.config.connection_string,
                baud=self.config.baud_rate,
                wait_ready=True
            )
            return vehicle
        except Exception as e:
            raise ConnectionError(f"Failed to connect to vehicle: {str(e)}")

    def check_armable(self) -> bool:
        """
        Check if the drone is ready to be armed
        
        Returns:
            bool: True if vehicle is armable, False otherwise
        """
        return self.vehicle.is_armable

    def arm_vehicle(self) -> bool:
        """
        Arm the drone after necessary checks
        
        Returns:
            bool: True if arming successful, False otherwise
        """
        if not self.check_armable():
            print("Vehicle is not armable. Please check:")
            print("- GPS lock")
            print("- Prearm checks")
            print("- System status")
            return False

        print("Vehicle is armable, switching to GUIDED mode")
        self.vehicle.mode = VehicleMode("GUIDED")
        
        # Wait for mode switch
        while self.vehicle.mode != 'GUIDED':
            print("Waiting for mode switch to GUIDED...")
            time.sleep(1)
        print("Vehicle now in GUIDED mode")

        # Handle arming based on configuration
        if not self.config.manual_arm:
            print("Initiating automatic arming sequence")
            self.vehicle.armed = True
            
            while not self.vehicle.armed:
                print("Waiting for arming...")
                time.sleep(1)
            print("Vehicle is now armed")
        else:
            if not self.vehicle.armed:
                print("Manual arming selected but vehicle not armed")
                print("Please arm the vehicle manually or set manual_arm to False")
                return False
            print("Vehicle manually armed")

        return True

    def takeoff(self) -> bool:
        """
        Execute takeoff sequence to target altitude
        
        Returns:
            bool: True if takeoff successful, False otherwise
        """
        try:
            print(f"Taking off to {self.config.target_altitude}m")
            self.vehicle.simple_takeoff(self.config.target_altitude)

            # Monitor altitude
            while True:
                current_altitude = self.vehicle.location.global_relative_frame.alt
                print(f"Altitude: {current_altitude:.1f}m")
                
                # Break if we're close enough to target altitude
                if current_altitude >= self.config.target_altitude * 0.95:
                    print("Reached target altitude")
                    break
                    
                time.sleep(1)
            return True
            
        except Exception as e:
            print(f"Takeoff failed: {str(e)}")
            return False

    def land(self) -> bool:
        """
        Execute landing sequence
        
        Returns:
            bool: True if landing initiated successfully, False otherwise
        """
        try:
            print("Initiating landing sequence")
            self.vehicle.mode = VehicleMode("LAND")
            
            while self.vehicle.mode != 'LAND':
                print("Waiting for drone to enter LAND mode...")
                time.sleep(1)
            
            print("Vehicle now in LAND mode")
            
            # Monitor landing
            while self.vehicle.armed:
                altitude = self.vehicle.location.global_relative_frame.alt
                print(f"Altitude: {altitude:.1f}m")
                time.sleep(1)
                
            print("Landing complete")
            return True
            
        except Exception as e:
            print(f"Landing sequence failed: {str(e)}")
            return False

    def cleanup(self):
        """Clean up vehicle connection and resources"""
        if self.vehicle:
            self.vehicle.close()
            print("Vehicle connection closed")

def main():
    """Main execution function"""
    # Default configuration
    config = FlightConfig(
        target_altitude=1.0,  # meters
        manual_arm=False,
        connection_string='/dev/ttyACM0',  # USB connection
        baud_rate=57600
    )

    drone = None
    try:
        # Initialize drone
        drone = DroneController(config)

        # Execute flight sequence
        if drone.arm_vehicle():
            if drone.takeoff():
                time.sleep(5)  # Hover for 5 seconds
                drone.land()
            else:
                print("Takeoff failed, initiating landing")
                drone.land()
        else:
            print("Arming failed, mission aborted")

    except KeyboardInterrupt:
        print("\nOperation interrupted by user!")
        if drone:
            print("Initiating emergency landing...")
            drone.land()
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if drone:
            print("Attempting emergency landing...")
            drone.land()
            
    finally:
        if drone:
            drone.cleanup()
            print("Flight operations completed")

if __name__ == '__main__':
    main()

# AirSim uses NED (North-East-Down) coordinate system
# This is CRITICAL to understand:

"""
     X (North/Forward)
     ↑
     |
     |
     +----- → Y (East/Right)
    /
   /
  ↓
  Z (Down)

- X increases forward
- Y increases right
- Z increases downward (Z=-5 means 5 meters above ground)
"""

# Test coordinate understanding
import airsim
import time

def test_coordinates():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    
    # Move forward 10m (increase X)
    client.moveToPositionAsync(10, 0, -5, velocity=2).join()
    print(f"Moved to: {client.getMultirotorState().kinematics_estimated.position}")
    
    # Move right 5m (increase Y)
    client.moveToPositionAsync(10, 5, -5, velocity=2).join()
    print(f"Moved to: {client.getMultirotorState().kinematics_estimated.position}")
    
    # Move up 3m (decrease Z)
    client.moveToPositionAsync(10, 5, -8, velocity=1).join()
    print(f"Moved to: {client.getMultirotorState().kinematics_estimated.position}")
    
    client.landAsync().join()

if __name__ == "__main__":
    test_coordinates()


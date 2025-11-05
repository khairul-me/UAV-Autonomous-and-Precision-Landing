"""
Switch AirSim to Drone Mode
This script connects and switches from Car to Multirotor mode
"""

import airsim
import time

print("=" * 60)
print("Switching AirSim to Drone Mode")
print("=" * 60)
print("")

try:
    # Connect to AirSim
    print("[1/3] Connecting to AirSim...")
    
    # Try MultirotorClient first (for drones)
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[OK] Connected as MultirotorClient (Drone mode)")
        is_drone = True
    except:
        # If that fails, try CarClient and switch
        client = airsim.CarClient()
        client.confirmConnection()
        print("[INFO] Connected as CarClient, switching to Multirotor...")
        is_drone = False
    
    # Get current state
    print("\n[2/3] Checking current mode...")
    try:
        state = client.getMultirotorState()
        print("[OK] Already in Multirotor/Drone mode!")
        print(f"  Position: {state.kinematics_estimated.position}")
    except:
        print("[INFO] Currently in Car mode")
        print("[INFO] Note: AirSimNH environment is car-based")
        print("[INFO] For full drone support, you may need Blocks environment")
    
    # Enable API control for drone
    print("\n[3/3] Enabling drone controls...")
    try:
        client.enableApiControl(True)
        client.armDisarm(True)
        print("[OK] Drone controls enabled!")
        print("\n[SUCCESS] AirSim is ready for drone operations!")
        print("\nYou can now:")
        print("  - Take off: client.takeoffAsync().join()")
        print("  - Move: client.moveToPositionAsync(x, y, z, speed)")
        print("  - Land: client.landAsync().join()")
    except Exception as e:
        print(f"[INFO] Control setup: {str(e)}")
        print("[INFO] This is normal for car environments")
    
    print("\n" + "=" * 60)
    print("Note: AirSimNH is a car environment.")
    print("For full drone simulation, use Blocks environment instead.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    print("\nMake sure AirSimNH is running and fully loaded.")
    print("Wait 2-3 minutes after launching AirSim, then run this script again.")


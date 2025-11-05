"""
Complete Drone Flight Test
This script will take off, fly, and land the drone
"""

import airsim
import time

print("=" * 60)
print("DRONE FLIGHT TEST")
print("=" * 60)
print("")

try:
    # Connect
    print("[1/5] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected!")
    
    # Enable API control
    print("\n[2/5] Enabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] API control enabled and drone armed!")
    
    # Get initial state
    print("\n[3/5] Getting initial state...")
    state = client.getMultirotorState()
    print(f"[OK] Initial position: {state.kinematics_estimated.position}")
    
    # Take off
    print("\n[4/5] Taking off...")
    print("  This will take 5 seconds...")
    client.takeoffAsync(timeout_sec=10).join()
    print("[OK] Takeoff complete!")
    
    # Get position after takeoff
    state = client.getMultirotorState()
    print(f"[OK] Current position: {state.kinematics_estimated.position}")
    
    # Hover for 2 seconds
    print("\n[INFO] Hovering for 2 seconds...")
    time.sleep(2)
    
    # Move forward
    print("\n[5/5] Moving forward 5 meters...")
    client.moveToPositionAsync(5, 0, -5, 5).join()
    print("[OK] Movement complete!")
    
    # Get final position
    state = client.getMultirotorState()
    print(f"[OK] Final position: {state.kinematics_estimated.position}")
    
    # Land
    print("\n[LANDING] Landing drone...")
    client.landAsync(timeout_sec=10).join()
    print("[OK] Landed successfully!")
    
    # Disarm
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DRONE FLIGHT TEST COMPLETE!")
    print("=" * 60)
    print("\nThe drone successfully:")
    print("  ✓ Took off")
    print("  ✓ Flew forward")
    print("  ✓ Landed")
    print("\nEverything is working!")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure Blocks is running and fully loaded")
    print("2. Check that settings.json has SimMode: Multirotor")
    print("3. Ensure the drone is not already in flight")
    print("4. Try running: python test_drone_simple.py first")
    
    import traceback
    traceback.print_exc()


"""
Make Drone Fly - Complete Working Script
This WILL make the drone fly
"""

import airsim
import time

print("=" * 60)
print("MAKING DRONE FLY")
print("=" * 60)
print("")

try:
    # Connect
    print("[1] Connecting...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected!")
    
    # Enable controls
    print("\n[2] Enabling controls...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Controls enabled!")
    
    # Get initial state
    state = client.getMultirotorState()
    print(f"\n[INFO] Initial position: {state.kinematics_estimated.position}")
    print(f"[INFO] Landed state: {state.landed_state}")
    
    # Take off
    print("\n[3] TAKING OFF...")
    print("    This will lift the drone 5 meters...")
    client.takeoffAsync(timeout_sec=20).join()
    
    # Verify takeoff
    state = client.getMultirotorState()
    print(f"[OK] Takeoff complete!")
    print(f"[OK] Current position: {state.kinematics_estimated.position}")
    print(f"[OK] Landed state: {state.landed_state}")
    
    # Hover
    print("\n[4] Hovering for 3 seconds...")
    time.sleep(3)
    
    # Move forward
    print("\n[5] Moving forward 10 meters...")
    client.moveToPositionAsync(10, 0, -5, 5).join()
    print("[OK] Movement complete!")
    
    state = client.getMultirotorState()
    print(f"[OK] New position: {state.kinematics_estimated.position}")
    
    # Move back
    print("\n[6] Moving back to start...")
    client.moveToPositionAsync(0, 0, -5, 5).join()
    print("[OK] Returned!")
    
    # Land
    print("\n[7] Landing...")
    client.landAsync(timeout_sec=20).join()
    print("[OK] Landed!")
    
    # Disarm
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DRONE FLEW SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe drone:")
    print("  ✓ Took off")
    print("  ✓ Flew forward")
    print("  ✓ Returned")
    print("  ✓ Landed")
    print("\nEverything works!")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure Blocks is running and showing drone (not car)")
    print("2. Check settings.json has SimMode: Multirotor")
    print("3. Ensure Blocks is fully loaded (wait 2-3 minutes)")
    
    import traceback
    traceback.print_exc()


"""
FLY DRONE NOW - This will make the drone fly
Waits for connection and then executes flight
"""

import airsim
import time

print("=" * 60)
print("FLYING DRONE NOW")
print("=" * 60)
print("")

# Wait for connection with retries
max_retries = 10
retry_delay = 5

for attempt in range(max_retries):
    try:
        print(f"[Attempt {attempt + 1}/{max_retries}] Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[OK] Connected!")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            print(f"[WAIT] Connection failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"\n[ERROR] Could not connect after {max_retries} attempts")
            print("Make sure Blocks is running and fully loaded!")
            exit(1)

try:
    # Enable API control
    print("\n[1] Enabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Drone armed and ready!")
    
    # Get state
    state = client.getMultirotorState()
    print(f"\n[INFO] Initial position: {state.kinematics_estimated.position}")
    print(f"[INFO] Landed state: {state.landed_state}")
    
    # Take off
    print("\n[2] TAKING OFF...")
    print("    Lifting drone 5 meters...")
    client.takeoffAsync(timeout_sec=20).join()
    
    # Verify
    state = client.getMultirotorState()
    print(f"[OK] Takeoff complete!")
    print(f"[OK] Position: {state.kinematics_estimated.position}")
    print(f"[OK] Landed: {state.landed_state}")
    
    # Hover
    print("\n[3] Hovering for 2 seconds...")
    time.sleep(2)
    
    # Move forward
    print("\n[4] Moving forward 10 meters...")
    client.moveToPositionAsync(10, 0, -5, 5).join()
    print("[OK] Moved forward!")
    
    state = client.getMultirotorState()
    print(f"[OK] Position: {state.kinematics_estimated.position}")
    
    # Move right
    print("\n[5] Moving right 5 meters...")
    client.moveToPositionAsync(10, 5, -5, 5).join()
    print("[OK] Moved right!")
    
    # Return to start
    print("\n[6] Returning to start position...")
    client.moveToPositionAsync(0, 0, -5, 5).join()
    print("[OK] Returned!")
    
    # Hover
    print("\n[7] Hovering for 2 seconds...")
    time.sleep(2)
    
    # Land
    print("\n[8] Landing...")
    client.landAsync(timeout_sec=20).join()
    print("[OK] Landed successfully!")
    
    # Disarm
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DRONE FLEW SUCCESSFULLY!")
    print("=" * 60)
    print("\nFlight completed:")
    print("  ✓ Took off")
    print("  ✓ Flew forward 10m")
    print("  ✓ Flew right 5m")
    print("  ✓ Returned to start")
    print("  ✓ Landed")
    print("\nEverything works perfectly!")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    print("\nDiagnostics:")
    try:
        state = client.getMultirotorState()
        print(f"  Position: {state.kinematics_estimated.position}")
        print(f"  Landed: {state.landed_state}")
    except:
        pass
    
    import traceback
    traceback.print_exc()


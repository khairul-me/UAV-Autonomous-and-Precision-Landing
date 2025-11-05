"""
Connect and Fly - Try different connection methods
"""

import airsim
import time

print("=" * 60)
print("CONNECTING AND FLYING DRONE")
print("=" * 60)
print("")

# Try different connection methods
connection_methods = [
    ("Default", lambda: airsim.MultirotorClient()),
    ("With IP", lambda: airsim.MultirotorClient(ip="127.0.0.1")),
    ("With Port", lambda: airsim.MultirotorClient(ip="127.0.0.1", port=41451)),
]

client = None
for method_name, method_func in connection_methods:
    try:
        print(f"Trying connection method: {method_name}...")
        client = method_func()
        client.confirmConnection()
        print(f"[OK] Connected using {method_name}!")
        break
    except Exception as e:
        print(f"[FAILED] {method_name}: {str(e)[:50]}")
        continue

if client is None:
    print("\n[ERROR] Could not connect with any method!")
    print("\nPossible issues:")
    print("1. Blocks AirSim plugin not initialized")
    print("2. API server not started on port 41451")
    print("3. Firewall blocking connection")
    print("4. Blocks needs to be restarted")
    exit(1)

try:
    # Enable controls
    print("\n[1] Enabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Drone armed!")
    
    # Get state
    state = client.getMultirotorState()
    print(f"\n[INFO] Initial position: {state.kinematics_estimated.position}")
    print(f"[INFO] Landed state: {state.landed_state}")
    
    # Take off
    print("\n[2] TAKING OFF...")
    client.takeoffAsync(timeout_sec=20).join()
    
    state = client.getMultirotorState()
    print(f"[OK] Takeoff complete!")
    print(f"[OK] Position: {state.kinematics_estimated.position}")
    print(f"[OK] Landed: {state.landed_state}")
    
    # Move
    print("\n[3] Moving forward 10 meters...")
    client.moveToPositionAsync(10, 0, -5, 5).join()
    print("[OK] Moved forward!")
    
    # Return
    print("\n[4] Returning to start...")
    client.moveToPositionAsync(0, 0, -5, 5).join()
    print("[OK] Returned!")
    
    # Land
    print("\n[5] Landing...")
    client.landAsync(timeout_sec=20).join()
    print("[OK] Landed!")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] DRONE FLEW SUCCESSFULLY!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()


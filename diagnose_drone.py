"""
Diagnostic script to check why drone won't fly
"""

import airsim
import json
import os

print("=" * 60)
print("DRONE DIAGNOSTICS")
print("=" * 60)
print("")

try:
    # Check connection
    print("[1] Testing connection...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to AirSim")
    
    # Check settings
    print("\n[2] Checking settings...")
    settings_path = os.path.join(os.path.expanduser('~'), 'Documents', 'AirSim', 'settings.json')
    print(f"Settings path: {settings_path}")
    
    if os.path.exists(settings_path):
        print("[OK] Settings file exists")
        with open(settings_path) as f:
            settings = json.load(f)
        print(f"SimMode: {settings.get('SimMode', 'NOT SET')}")
        if settings.get('SimMode') != 'Multirotor':
            print("[ERROR] SimMode is NOT Multirotor!")
            print("[FIX] Updating settings...")
            settings['SimMode'] = 'Multirotor'
            if 'Vehicles' not in settings:
                settings['Vehicles'] = {
                    "Drone1": {
                        "VehicleType": "SimpleFlight",
                        "X": 0, "Y": 0, "Z": 0,
                        "Yaw": 0
                    }
                }
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print("[OK] Settings updated! Restart Blocks.")
    else:
        print("[ERROR] Settings file not found!")
    
    # Check API control
    print("\n[3] Checking API control...")
    try:
        client.enableApiControl(True)
        print("[OK] API control enabled")
    except Exception as e:
        print(f"[ERROR] Cannot enable API control: {e}")
    
    # Check arming
    print("\n[4] Checking arming...")
    try:
        client.armDisarm(True)
        print("[OK] Drone armed")
    except Exception as e:
        print(f"[ERROR] Cannot arm: {e}")
    
    # Check state
    print("\n[5] Checking drone state...")
    try:
        state = client.getMultirotorState()
        print(f"[OK] Position: {state.kinematics_estimated.position}")
        print(f"[OK] Orientation: {state.kinematics_estimated.orientation}")
        print(f"[OK] Landed state: {state.landed_state}")
    except Exception as e:
        print(f"[ERROR] Cannot get state: {e}")
    
    # Try takeoff
    print("\n[6] Testing takeoff...")
    try:
        print("Attempting takeoff...")
        client.takeoffAsync(timeout_sec=10).join()
        print("[SUCCESS] Takeoff worked!")
        state = client.getMultirotorState()
        print(f"New position: {state.kinematics_estimated.position}")
        print("[OK] Drone is flying!")
    except Exception as e:
        print(f"[ERROR] Takeoff failed: {e}")
        print("\nPossible issues:")
        print("  1. Drone might already be in air")
        print("  2. Settings might not be Multirotor mode")
        print("  3. Blocks might need restart")
    
    print("\n" + "=" * 60)
    print("Diagnostics complete!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()


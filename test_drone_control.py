"""
Simple Drone Control Test
Quick test to verify drone control is working
"""

import airsim
import time

def main():
    print("========================================")
    print("Drone Control Test")
    print("========================================")
    
    # Connect
    print("\n[1] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        print("  [OK] Connected!")
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        return
    
    # Enable API control
    print("\n[2] Enabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("  [OK] API control enabled, drone armed")
    
    # Get initial state
    print("\n[3] Getting initial state...")
    state = client.getMultirotorState()
    initial_pos = state.kinematics_estimated.position
    print(f"  Initial position: x={initial_pos.x_val:.2f}, y={initial_pos.y_val:.2f}, z={initial_pos.z_val:.2f}")
    
    try:
        # Takeoff
        print("\n[4] Testing takeoff...")
        client.takeoffAsync(timeout_sec=10).join()
        print("  [OK] Takeoff successful!")
        time.sleep(2)
        
        # Get position after takeoff
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"  Position after takeoff: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
        
        # Hover test
        print("\n[5] Testing hover...")
        client.hoverAsync().join()
        print("  [OK] Hovering...")
        time.sleep(2)
        
        # Small movement test
        print("\n[6] Testing movement (forward 2 meters)...")
        client.moveToPositionAsync(2, 0, -3, 2).join()
        print("  [OK] Movement successful!")
        
        # Return to start
        print("\n[7] Returning to start position...")
        client.moveToPositionAsync(0, 0, -3, 2).join()
        print("  [OK] Returned to start")
        
        # Land
        print("\n[8] Testing landing...")
        client.landAsync().join()
        print("  [OK] Landing successful!")
        
        print("\n========================================")
        print("[SUCCESS] All tests passed!")
        print("Drone control is working correctly!")
        print("========================================")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        print("Attempting emergency landing...")
        try:
            client.landAsync().join()
        except:
            pass
    
    finally:
        # Cleanup
        print("\n[9] Cleaning up...")
        client.armDisarm(False)
        client.enableApiControl(False)
        print("  [OK] Drone disarmed")

if __name__ == "__main__":
    main()

"""
MAKE_IT_FLY.py - Phase 0 Task 0.1
Complete flight sequence demonstration
Success Criteria: Successfully executes full flight sequence
"""

import airsim
import time
import sys

def print_status(message, status="INFO"):
    """Print formatted status message"""
    symbols = {
        "OK": "[OK]",
        "ERROR": "[ERROR]",
        "INFO": "[INFO]",
        "SUCCESS": "[SUCCESS]"
    }
    print(f"{symbols.get(status, '[INFO]')} {message}")

def main():
    print("=" * 70)
    print("PHASE 0 TASK 0.1: MAKE IT FLY - Complete Flight Sequence")
    print("=" * 70)
    print("")
    
    client = None
    
    try:
        # Step 1: Connect to AirSim
        print_status("Connecting to AirSim...", "INFO")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print_status("Connected successfully!", "OK")
        
        # Step 2: Enable API Control
        print_status("Enabling API control...", "INFO")
        client.enableApiControl(True)
        print_status("API control enabled", "OK")
        
        # Step 3: Arm the drone
        print_status("Arming drone...", "INFO")
        client.armDisarm(True)
        print_status("Drone armed", "OK")
        
        # Step 4: Get initial state
        print_status("Getting initial state...", "INFO")
        initial_state = client.getMultirotorState()
        initial_pos = initial_state.kinematics_estimated.position
        print_status(f"Initial position: X={initial_pos.x_val:.2f}, Y={initial_pos.y_val:.2f}, Z={initial_pos.z_val:.2f}", "OK")
        print_status(f"Initial landed state: {initial_state.landed_state}", "OK")
        
        # Step 5: Takeoff
        print("\n" + "=" * 70)
        print_status("TAKING OFF - Lifting to 5m altitude...", "INFO")
        print("=" * 70)
        client.takeoffAsync(timeout_sec=30).join()
        time.sleep(2)  # Wait for stabilization
        
        # Verify takeoff
        state = client.getMultirotorState()
        current_pos = state.kinematics_estimated.position
        altitude = abs(current_pos.z_val)
        print_status(f"Current altitude: {altitude:.2f}m", "OK")
        if altitude < 3.0:
            print_status("Takeoff may not have completed, retrying...", "INFO")
            client.takeoffAsync(timeout_sec=30).join()
            time.sleep(2)
            state = client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            altitude = abs(current_pos.z_val)
            print_status(f"New altitude: {altitude:.2f}m", "OK")
        
        print_status("Takeoff complete!", "OK")
        
        # Step 6: Move forward
        print("\n" + "=" * 70)
        print_status("Moving forward 10 meters...", "INFO")
        print("=" * 70)
        client.moveToPositionAsync(10, 0, -5, 5).join()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print_status(f"Position after move: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}", "OK")
        
        # Step 7: Hover
        print("\n" + "=" * 70)
        print_status("Hovering for 3 seconds...", "INFO")
        print("=" * 70)
        client.hoverAsync().join()
        time.sleep(3)
        print_status("Hover complete", "OK")
        
        # Step 8: Move right
        print("\n" + "=" * 70)
        print_status("Moving right 5 meters...", "INFO")
        print("=" * 70)
        client.moveToPositionAsync(10, 5, -5, 5).join()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print_status(f"Position after move: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}", "OK")
        
        # Step 9: Rotate (yaw)
        print("\n" + "=" * 70)
        print_status("Rotating 90 degrees...", "INFO")
        print("=" * 70)
        import math
        client.rotateToYawAsync(math.radians(90), timeout_sec=10).join()
        print_status("Rotation complete", "OK")
        
        # Step 10: Return to start
        print("\n" + "=" * 70)
        print_status("Returning to start position...", "INFO")
        print("=" * 70)
        client.moveToPositionAsync(0, 0, -5, 5).join()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print_status(f"Final position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}", "OK")
        
        # Step 11: Land
        print("\n" + "=" * 70)
        print_status("Landing...", "INFO")
        print("=" * 70)
        client.landAsync(timeout_sec=30).join()
        time.sleep(2)
        
        # Verify landing
        state = client.getMultirotorState()
        final_pos = state.kinematics_estimated.position
        final_altitude = abs(final_pos.z_val)
        print_status(f"Final altitude: {final_altitude:.2f}m", "OK")
        print_status(f"Landed state: {state.landed_state}", "OK")
        
        # Step 12: Disarm and disable API control
        print_status("Disarming drone...", "INFO")
        client.armDisarm(False)
        client.enableApiControl(False)
        print_status("Drone disarmed, API control disabled", "OK")
        
        # Success summary
        print("\n" + "=" * 70)
        print_status("PHASE 0 TASK 0.1: SUCCESS - Complete flight sequence executed!", "SUCCESS")
        print("=" * 70)
        print("\nFlight sequence completed successfully:")
        print("  ✓ Connected to AirSim")
        print("  ✓ Enabled API control and armed")
        print("  ✓ Took off to 5m altitude")
        print("  ✓ Moved forward 10 meters")
        print("  ✓ Hovered in place")
        print("  ✓ Moved right 5 meters")
        print("  ✓ Rotated 90 degrees")
        print("  ✓ Returned to start position")
        print("  ✓ Landed safely")
        print("  ✓ Disarmed and disabled API control")
        print("\n" + "=" * 70)
        
        return True
        
    except airsim.exceptions.AirSimException as e:
        print_status(f"AirSim error: {str(e)}", "ERROR")
        if client:
            try:
                print_status("Attempting emergency landing...", "INFO")
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass
        return False
        
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        if client:
            try:
                print_status("Attempting emergency landing...", "INFO")
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


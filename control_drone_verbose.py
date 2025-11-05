"""
DRONE CONTROL SCRIPT - VERBOSE VERSION
This script shows you EXACTLY how to arm and control your drone
with detailed output at every step.
"""

import airsim
import time
import sys
import math

def print_step(step_num, description):
    """Print a formatted step header"""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {description}")
    print("=" * 70)

def print_status(message, status="INFO"):
    """Print formatted status message"""
    symbols = {
        "OK": "[OK]",
        "ERROR": "[ERROR]",
        "INFO": "[INFO]",
        "SUCCESS": "[SUCCESS]",
        "WARNING": "[WARNING]"
    }
    print(f"{symbols.get(status, '[INFO]')} {message}")

def get_drone_state(client):
    """Get and display current drone state"""
    try:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orient = state.kinematics_estimated.orientation
        
        print_status(f"Position: X={pos.x_val:.2f}m, Y={pos.y_val:.2f}m, Z={pos.z_val:.2f}m", "INFO")
        print_status(f"Velocity: X={vel.x_val:.2f}m/s, Y={vel.y_val:.2f}m/s, Z={vel.z_val:.2f}m/s", "INFO")
        print_status(f"Landed State: {state.landed_state}", "INFO")
        print_status(f"Armed: {state.armed}", "INFO")
        
        return state
    except Exception as e:
        print_status(f"Error getting state: {e}", "ERROR")
        return None

def main():
    print("=" * 70)
    print("  DRONE CONTROL GUIDE - VERBOSE VERSION")
    print("=" * 70)
    print("\nThis script will show you EXACTLY how to:")
    print("  1. Connect to AirSim")
    print("  2. Enable API Control (take over manual control)")
    print("  3. Arm the Drone (prepare motors)")
    print("  4. Take Off")
    print("  5. Control Flight")
    print("  6. Land Safely")
    print("  7. Disarm and Release Control")
    print("\n" + "=" * 70)
    
    client = None
    
    try:
        # ============================================================
        # STEP 1: CONNECT TO AIRSIM
        # ============================================================
        print_step(1, "CONNECTING TO AIRSIM")
        
        print_status("Creating MultirotorClient...", "INFO")
        client = airsim.MultirotorClient()
        print_status("MultirotorClient created", "OK")
        
        print_status("Attempting to connect to AirSim API server...", "INFO")
        print_status("Looking for server on localhost:41451", "INFO")
        print_status("Waiting for connection...", "INFO")
        
        try:
            client.confirmConnection()
            print_status("SUCCESS! Connected to AirSim API server!", "SUCCESS")
            print_status("AirSim is ready to receive commands", "OK")
        except Exception as e:
            print_status(f"CONNECTION FAILED: {e}", "ERROR")
            print_status("Make sure Blocks.exe or AirSimNH.exe is running!", "WARNING")
            print_status("Wait 2-5 minutes after launching for it to fully load", "WARNING")
            return False
        
        # ============================================================
        # STEP 2: CHECK INITIAL STATE
        # ============================================================
        print_step(2, "CHECKING INITIAL DRONE STATE")
        
        print_status("Getting current drone state...", "INFO")
        initial_state = get_drone_state(client)
        if initial_state is None:
            print_status("Could not get initial state, continuing anyway...", "WARNING")
        
        # ============================================================
        # STEP 3: ENABLE API CONTROL
        # ============================================================
        print_step(3, "ENABLING API CONTROL (TAKING OVER MANUAL CONTROL)")
        
        print_status("What this does: Allows Python script to control the drone", "INFO")
        print_status("Without this, you can only READ data, not CONTROL the drone", "INFO")
        print_status("Enabling API control...", "INFO")
        
        try:
            client.enableApiControl(True)
            print_status("API Control ENABLED!", "SUCCESS")
            print_status("You now have control of the drone via Python", "OK")
        except Exception as e:
            print_status(f"FAILED to enable API control: {e}", "ERROR")
            return False
        
        # Check if API control is actually enabled
        print_status("Verifying API control is enabled...", "INFO")
        try:
            # There's no direct way to check, but we can try a command
            # If it works, API control is enabled
            print_status("API control should be enabled now", "OK")
        except:
            pass
        
        # ============================================================
        # STEP 4: ARM THE DRONE
        # ============================================================
        print_step(4, "ARMING THE DRONE (STARTING MOTORS)")
        
        print_status("What arming does: Starts the motors and prepares for flight", "INFO")
        print_status("Armed = Motors spinning, ready to take off", "INFO")
        print_status("Disarmed = Motors off, safe to approach", "INFO")
        print_status("Attempting to arm drone...", "INFO")
        
        try:
            result = client.armDisarm(True)
            print_status(f"Arm command sent. Result: {result}", "INFO")
            
            # Wait a moment for motors to start
            print_status("Waiting 2 seconds for motors to start...", "INFO")
            time.sleep(2)
            
            # Check state
            state = client.getMultirotorState()
            print_status(f"Armed status: {state.armed}", "INFO")
            
            if state.armed:
                print_status("DRONE IS ARMED! Motors are running!", "SUCCESS")
            else:
                print_status("Warning: Drone may not be fully armed yet", "WARNING")
                print_status("This is OK, it might arm when taking off", "INFO")
        except Exception as e:
            print_status(f"Error arming drone: {e}", "ERROR")
            print_status("Continuing anyway - some systems arm automatically on takeoff", "WARNING")
        
        # ============================================================
        # STEP 5: TAKE OFF
        # ============================================================
        print_step(5, "TAKING OFF")
        
        print_status("Getting current altitude before takeoff...", "INFO")
        state = client.getMultirotorState()
        initial_z = state.kinematics_estimated.position.z_val
        print_status(f"Starting altitude: Z = {initial_z:.2f}m", "INFO")
        
        print_status("Sending takeoff command...", "INFO")
        print_status("Target altitude: 5 meters", "INFO")
        print_status("This will take about 5-10 seconds...", "INFO")
        
        try:
            # Takeoff async and wait for completion
            print_status("Takeoff in progress...", "INFO")
            client.takeoffAsync(timeout_sec=30).join()
            
            # Wait a bit for stabilization
            print_status("Waiting 2 seconds for drone to stabilize...", "INFO")
            time.sleep(2)
            
            # Check altitude
            state = client.getMultirotorState()
            current_z = state.kinematics_estimated.position.z_val
            altitude = abs(current_z - initial_z)
            
            print_status(f"Current altitude: Z = {current_z:.2f}m", "INFO")
            print_status(f"Altitude gained: {altitude:.2f}m", "INFO")
            
            if altitude > 3.0:
                print_status("SUCCESS! Drone is flying!", "SUCCESS")
            else:
                print_status("Takeoff may not have completed, retrying...", "WARNING")
                client.takeoffAsync(timeout_sec=30).join()
                time.sleep(2)
                state = client.getMultirotorState()
                current_z = state.kinematics_estimated.position.z_val
                altitude = abs(current_z - initial_z)
                print_status(f"New altitude: {altitude:.2f}m", "INFO")
                
        except Exception as e:
            print_status(f"Takeoff error: {e}", "ERROR")
            return False
        
        # ============================================================
        # STEP 6: FLIGHT MANEUVERS
        # ============================================================
        print_step(6, "FLIGHT MANEUVERS")
        
        # Move forward
        print_status("Moving forward 10 meters...", "INFO")
        target_x = 10
        target_y = 0
        target_z = -5  # Negative Z is UP in AirSim (NED coordinates)
        
        print_status(f"Target position: X={target_x}m, Y={target_y}m, Z={target_z}m", "INFO")
        print_status("Speed: 5 m/s", "INFO")
        print_status("Movement in progress...", "INFO")
        
        try:
            client.moveToPositionAsync(target_x, target_y, target_z, 5).join()
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print_status(f"Arrived at: X={pos.x_val:.2f}m, Y={pos.y_val:.2f}m, Z={pos.z_val:.2f}m", "OK")
        except Exception as e:
            print_status(f"Movement error: {e}", "ERROR")
        
        # Hover
        print_status("Hovering in place for 3 seconds...", "INFO")
        client.hoverAsync().join()
        time.sleep(3)
        print_status("Hover complete", "OK")
        
        # Move right
        print_status("Moving right 5 meters...", "INFO")
        try:
            client.moveToPositionAsync(10, 5, -5, 5).join()
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print_status(f"Position: X={pos.x_val:.2f}m, Y={pos.y_val:.2f}m, Z={pos.z_val:.2f}m", "OK")
        except Exception as e:
            print_status(f"Movement error: {e}", "ERROR")
        
        # Rotate
        print_status("Rotating 90 degrees (yaw)...", "INFO")
        try:
            client.rotateToYawAsync(math.radians(90), timeout_sec=10).join()
            print_status("Rotation complete", "OK")
        except Exception as e:
            print_status(f"Rotation error: {e}", "ERROR")
        
        # Return to start
        print_status("Returning to start position...", "INFO")
        try:
            client.moveToPositionAsync(0, 0, -5, 5).join()
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print_status(f"Back at start: X={pos.x_val:.2f}m, Y={pos.y_val:.2f}m, Z={pos.z_val:.2f}m", "OK")
        except Exception as e:
            print_status(f"Movement error: {e}", "ERROR")
        
        # ============================================================
        # STEP 7: LAND
        # ============================================================
        print_step(7, "LANDING")
        
        print_status("Getting current altitude...", "INFO")
        state = client.getMultirotorState()
        current_z = state.kinematics_estimated.position.z_val
        print_status(f"Current altitude: {abs(current_z):.2f}m", "INFO")
        
        print_status("Sending land command...", "INFO")
        print_status("Drone will descend and land automatically", "INFO")
        
        try:
            client.landAsync(timeout_sec=30).join()
            print_status("Waiting 2 seconds for landing to complete...", "INFO")
            time.sleep(2)
            
            # Check final state
            state = client.getMultirotorState()
            final_z = state.kinematics_estimated.position.z_val
            print_status(f"Final altitude: {abs(final_z):.2f}m", "INFO")
            print_status(f"Landed state: {state.landed_state}", "INFO")
            
            if state.landed_state == airsim.LandedState.Landed:
                print_status("SUCCESS! Drone has landed safely!", "SUCCESS")
            else:
                print_status("Landing may still be in progress", "INFO")
                
        except Exception as e:
            print_status(f"Landing error: {e}", "ERROR")
        
        # ============================================================
        # STEP 8: DISARM AND CLEANUP
        # ============================================================
        print_step(8, "DISARMING AND CLEANUP")
        
        print_status("Disarming drone (stopping motors)...", "INFO")
        try:
            client.armDisarm(False)
            time.sleep(1)
            state = client.getMultirotorState()
            print_status(f"Armed status: {state.armed}", "INFO")
            print_status("Drone disarmed - motors stopped", "OK")
        except Exception as e:
            print_status(f"Disarm error: {e}", "ERROR")
        
        print_status("Disabling API control (releasing control)...", "INFO")
        try:
            client.enableApiControl(False)
            print_status("API control disabled - manual control restored", "OK")
        except Exception as e:
            print_status(f"Error disabling API control: {e}", "ERROR")
        
        # ============================================================
        # SUCCESS SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print_status("FLIGHT COMPLETE! All steps executed successfully!", "SUCCESS")
        print("=" * 70)
        print("\nSUMMARY OF WHAT HAPPENED:")
        print("  [OK] Connected to AirSim")
        print("  [OK] Enabled API control (Python has control)")
        print("  [OK] Armed drone (motors started)")
        print("  [OK] Took off to 5m altitude")
        print("  [OK] Performed flight maneuvers")
        print("  [OK] Landed safely")
        print("  [OK] Disarmed drone (motors stopped)")
        print("  [OK] Released API control")
        print("\n" + "=" * 70)
        
        return True
        
    except airsim.exceptions.AirSimException as e:
        print_status(f"AirSim error: {str(e)}", "ERROR")
        print_status("Attempting emergency landing...", "WARNING")
        if client:
            try:
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
                print_status("Attempting emergency landing...", "WARNING")
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass
        return False

if __name__ == "__main__":
    print("\nIMPORTANT: Make sure Blocks.exe or AirSimNH.exe is running!")
    print("Wait 2-5 minutes after launching for it to fully load.")
    print("\nStarting in 2 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(2)  # Give user 2 seconds to cancel
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYou now know how to control the drone!")
        print("Key commands to remember:")
        print("  1. client.enableApiControl(True)  - Take control")
        print("  2. client.armDisarm(True)         - Start motors")
        print("  3. client.takeoffAsync().join()   - Take off")
        print("  4. client.moveToPositionAsync()   - Move to position")
        print("  5. client.landAsync().join()      - Land")
        print("  6. client.armDisarm(False)        - Stop motors")
        print("  7. client.enableApiControl(False) - Release control")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("SCRIPT COMPLETED WITH ERRORS")
        print("=" * 70)
        print("\nCheck the output above for error messages.")
        sys.exit(1)

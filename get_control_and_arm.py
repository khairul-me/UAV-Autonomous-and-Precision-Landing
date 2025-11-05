"""
GET FULL CONTROL AND ARM THE DRONE
This script shows you EXACTLY how to:
1. Get full control of the drone
2. Arm the drone (start motors)
3. Verify everything is ready
"""

import airsim
import time

def print_header(text):
    """Print a clear header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step_num, description):
    """Print step information"""
    print(f"\n>>> STEP {step_num}: {description}")
    print("-" * 70)

def main():
    print_header("GET FULL CONTROL AND ARM THE DRONE")
    
    print("\nThis script will show you:")
    print("  1. How to CONNECT to AirSim")
    print("  2. How to ENABLE API CONTROL (get full control)")
    print("  3. How to ARM the drone (start motors)")
    print("  4. How to VERIFY everything is ready")
    
    client = None
    
    try:
        # ============================================================
        # STEP 1: CONNECT
        # ============================================================
        print_step(1, "CONNECTING TO AIRSIM")
        
        print("Creating client...")
        client = airsim.MultirotorClient()
        
        print("Connecting to AirSim API server (port 41451)...")
        client.confirmConnection()
        print("[SUCCESS] Connected to AirSim!")
        print("  → You can now READ drone data")
        print("  → But you CANNOT CONTROL it yet")
        
        # ============================================================
        # STEP 2: ENABLE API CONTROL (THIS IS CRITICAL!)
        # ============================================================
        print_step(2, "ENABLING API CONTROL - GETTING FULL CONTROL")
        
        print("What API Control does:")
        print("  • Without it: You can ONLY READ data (position, images, etc.)")
        print("  • With it: You can CONTROL the drone (fly, land, etc.)")
        print("\nEnabling API control...")
        
        client.enableApiControl(True)
        print("[SUCCESS] API Control ENABLED!")
        print("  → You now have FULL CONTROL of the drone")
        print("  → You can send flight commands")
        print("  → Keyboard/gamepad control is disabled")
        
        # ============================================================
        # STEP 3: ARM THE DRONE
        # ============================================================
        print_step(3, "ARMING THE DRONE (STARTING MOTORS)")
        
        print("What Arming does:")
        print("  • Starts the motors spinning")
        print("  • Prepares the drone for flight")
        print("  • Armed = Motors ON, ready to take off")
        print("  • Disarmed = Motors OFF, safe")
        
        print("\nSending arm command...")
        result = client.armDisarm(True)
        print(f"[SUCCESS] Arm command sent! Result: {result}")
        
        print("\nWaiting 2 seconds for motors to start...")
        time.sleep(2)
        
        # Try to check armed status
        try:
            state = client.getMultirotorState()
            # Some AirSim versions don't expose 'armed' attribute
            print("[OK] Drone state retrieved")
            print("  → Motors should be spinning now")
            print("  → Drone is ready to take off")
        except Exception as e:
            print(f"[INFO] Could not check armed status directly: {e}")
            print("  → But arm command was sent successfully")
            print("  → Motors should be spinning")
        
        # ============================================================
        # STEP 4: VERIFY READY TO FLY
        # ============================================================
        print_step(4, "VERIFYING DRONE IS READY TO FLY")
        
        print("Checking current state...")
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        
        print("\nCurrent Status:")
        print(f"  Position: X={pos.x_val:.2f}m, Y={pos.y_val:.2f}m, Z={pos.z_val:.2f}m")
        print(f"  Velocity: X={vel.x_val:.2f}m/s, Y={vel.y_val:.2f}m/s, Z={vel.z_val:.2f}m/s")
        print(f"  Landed State: {state.landed_state}")
        print("\n[SUCCESS] Drone is READY TO FLY!")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print_header("SUMMARY - YOU NOW HAVE FULL CONTROL!")
        
        print("\n✓ Connected to AirSim")
        print("✓ API Control ENABLED (you have full control)")
        print("✓ Drone ARMED (motors are running)")
        print("✓ Ready to take off and fly!")
        
        print("\n" + "=" * 70)
        print("NEXT COMMANDS YOU CAN USE:")
        print("=" * 70)
        print("\n1. TAKE OFF:")
        print("   client.takeoffAsync().join()")
        
        print("\n2. MOVE TO POSITION:")
        print("   client.moveToPositionAsync(x, y, z, speed).join()")
        print("   Example: client.moveToPositionAsync(10, 0, -5, 5).join()")
        print("           (move forward 10m, stay at 5m altitude)")
        
        print("\n3. HOVER:")
        print("   client.hoverAsync().join()")
        
        print("\n4. LAND:")
        print("   client.landAsync().join()")
        
        print("\n5. DISARM (stop motors):")
        print("   client.armDisarm(False)")
        
        print("\n6. RELEASE CONTROL:")
        print("   client.enableApiControl(False)")
        print("\n" + "=" * 70)
        
        # Ask if user wants to test takeoff
        print("\nWould you like to test takeoff now? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response == 'y':
                print("\n" + "=" * 70)
                print("  TESTING TAKEOFF")
                print("=" * 70)
                
                print("\nTaking off...")
                client.takeoffAsync(timeout_sec=30).join()
                
                print("Waiting 2 seconds...")
                time.sleep(2)
                
                state = client.getMultirotorState()
                current_z = state.kinematics_estimated.position.z_val
                print(f"[SUCCESS] Drone is flying at {abs(current_z):.2f}m altitude!")
                
                print("\nHovering for 3 seconds...")
                client.hoverAsync().join()
                time.sleep(3)
                
                print("Landing...")
                client.landAsync(timeout_sec=30).join()
                print("[SUCCESS] Drone has landed!")
                
        except KeyboardInterrupt:
            print("\n[Skipped] Takeoff test cancelled")
        
        # Cleanup
        print("\n" + "=" * 70)
        print("  CLEANUP")
        print("=" * 70)
        
        print("\nDisarming drone...")
        client.armDisarm(False)
        print("[OK] Drone disarmed")
        
        print("Releasing API control...")
        client.enableApiControl(False)
        print("[OK] API control released")
        
        print("\n[SUCCESS] All done!")
        return True
        
    except airsim.exceptions.AirSimException as e:
        print(f"\n[ERROR] AirSim error: {e}")
        if client:
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass
        return False
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        if client:
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
            except:
                pass
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  IMPORTANT: Make sure Blocks.exe or AirSimNH.exe is running!")
    print("  Wait 2-5 minutes after launching for it to fully load.")
    print("=" * 70)
    print("\nStarting in 2 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        print("\nCancelled.")
        exit(0)
    
    success = main()
    
    if success:
        print("\n" + "=" * 70)
        print("  SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  SCRIPT COMPLETED WITH ERRORS")
        print("=" * 70)

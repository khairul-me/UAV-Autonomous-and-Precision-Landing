"""
MAKE_IT_FLY.py - Phase 0 Task 0.1
Execute full flight sequence: takeoff, move, land
Success Criteria: Python scripts can connect and control drone
"""

import airsim
import time
import sys

def execute_full_flight_sequence():
    """Execute complete flight sequence: takeoff, waypoint navigation, land"""
    
    print("=" * 60)
    print("MAKE_IT_FLY.py - Full Flight Sequence Test")
    print("=" * 60)
    print()
    
    try:
        # Connect to AirSim
        print("[1/6] Connecting to AirSim...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[OK] Connected to AirSim!")
        
        # Enable API control
        print("[2/6] Enabling API control...")
        client.enableApiControl(True)
        client.armDisarm(True)
        print("[OK] API control enabled and armed!")
        
        # Takeoff
        print("[3/6] Taking off...")
        client.takeoffAsync().join()
        print("[OK] Takeoff complete!")
        
        # Wait for stabilization
        time.sleep(2)
        
        # Move to waypoint 1
        print("[4/6] Moving to waypoint 1 (5, 0, -5)...")
        client.moveToPositionAsync(5, 0, -5, 5).join()
        print("[OK] Reached waypoint 1!")
        
        time.sleep(1)
        
        # Move to waypoint 2
        print("[5/6] Moving to waypoint 2 (5, 5, -5)...")
        client.moveToPositionAsync(5, 5, -5, 5).join()
        print("[OK] Reached waypoint 2!")
        
        time.sleep(1)
        
        # Return to start
        print("[6/6] Returning to start position...")
        client.moveToPositionAsync(0, 0, -5, 5).join()
        print("[OK] Returned to start!")
        
        time.sleep(1)
        
        # Land
        print("[FINAL] Landing...")
        client.landAsync().join()
        print("[OK] Landed safely!")
        
        # Disarm
        client.armDisarm(False)
        client.enableApiControl(False)
        
        print()
        print("=" * 60)
        print("[SUCCESS] FULL FLIGHT SEQUENCE COMPLETED!")
        print("=" * 60)
        print()
        print("Phase 0 Task 0.1 Success Criteria: MET")
        print("✅ Python scripts connect to AirSim")
        print("✅ Drone can be controlled via API")
        print("✅ Full flight sequence (takeoff, move, land) works")
        
        return True
        
    except Exception as e:
        print()
        print("[ERROR] Flight sequence failed!")
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("1. Ensure Blocks.exe or AirSimNH.exe is running")
        print("2. Wait 30-60 seconds after launching AirSim before running this script")
        print("3. Check that settings.json is configured correctly")
        print("4. Verify drone is visible in the environment")
        return False

if __name__ == "__main__":
    success = execute_full_flight_sequence()
    sys.exit(0 if success else 1)


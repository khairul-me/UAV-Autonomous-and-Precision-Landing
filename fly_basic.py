"""
Basic Drone Flight Script
Simple takeoff, move, and land sequence
"""

import airsim
import time

def main():
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to AirSim!")
    
    # Enable API control and arm
    print("\nEnabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] API control enabled, drone armed")
    
    try:
        # Takeoff
        print("\n=== Taking off ===")
        client.takeoffAsync().join()
        print("[OK] Takeoff complete")
        time.sleep(2)  # Wait for stabilization
        
        # Move forward
        print("\n=== Moving forward 10 meters ===")
        client.moveToPositionAsync(10, 0, -5, 5).join()
        print("[OK] Moved forward")
        
        # Hover
        print("\n=== Hovering for 3 seconds ===")
        client.hoverAsync().join()
        time.sleep(3)
        
        # Move right
        print("\n=== Moving right 5 meters ===")
        client.moveToPositionAsync(10, 5, -5, 5).join()
        print("[OK] Moved right")
        
        # Return to start
        print("\n=== Returning to start position ===")
        client.moveToPositionAsync(0, 0, -5, 5).join()
        print("[OK] Returned to start")
        
        # Land
        print("\n=== Landing ===")
        client.landAsync().join()
        print("[OK] Landed safely")
        
    except Exception as e:
        print(f"\n[ERROR] Flight error: {e}")
        print("Attempting emergency landing...")
        client.landAsync().join()
    
    finally:
        # Disarm and disable API control
        print("\nDisarming drone...")
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[OK] Drone disarmed, API control disabled")
        print("\nFlight complete!")

if __name__ == "__main__":
    main()

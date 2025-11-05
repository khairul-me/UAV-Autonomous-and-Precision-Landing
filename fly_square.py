"""
Square Flight Pattern
Flies the drone in a square pattern
"""

import airsim
import time

def fly_square(client, side_length=10, altitude=-5, speed=3):
    """
    Fly a square pattern
    
    Args:
        client: AirSim MultirotorClient
        side_length: Length of each side in meters
        altitude: Flight altitude (negative = up)
        speed: Flight speed in m/s
    """
    print(f"\n=== Starting square flight pattern ({side_length}m x {side_length}m) ===")
    
    # Square corners
    waypoints = [
        (side_length, 0, altitude),      # Forward
        (side_length, side_length, altitude),  # Right
        (0, side_length, altitude),      # Back
        (0, 0, altitude)                 # Return to start
    ]
    
    for idx, (x, y, z) in enumerate(waypoints):
        print(f"Waypoint {idx+1}/4: Moving to ({x}, {y}, {z})")
        client.moveToPositionAsync(x, y, z, speed).join()
        print(f"  [OK] Reached waypoint {idx+1}")
        time.sleep(0.5)  # Brief pause at each corner
    
    print("\n[OK] Square pattern complete!")

def main():
    print("========================================")
    print("Square Flight Pattern")
    print("========================================")
    
    # Connect to AirSim
    print("\nConnecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected!")
    
    # Enable API control and arm
    print("\nEnabling API control...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Ready for flight")
    
    try:
        # Takeoff
        print("\n=== Taking off ===")
        client.takeoffAsync().join()
        print("[OK] Airborne!")
        time.sleep(2)
        
        # Fly square pattern
        fly_square(client, side_length=10, altitude=-5, speed=3)
        
        # Land
        print("\n=== Landing ===")
        client.landAsync().join()
        print("[OK] Landed safely")
        
    except Exception as e:
        print(f"\n[ERROR] Flight error: {e}")
        print("Emergency landing...")
        client.landAsync().join()
    
    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        print("\nFlight complete!")

if __name__ == "__main__":
    main()

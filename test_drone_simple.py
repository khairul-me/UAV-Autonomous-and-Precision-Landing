"""
Simple Drone Test - Run this after Blocks is fully loaded
"""

import airsim
import time

print("=" * 60)
print("Drone Connection Test")
print("=" * 60)
print("")

try:
    print("Connecting to Blocks (drone mode)...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to drone!")
    print("")
    
    print("Enabling drone controls...")
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] Drone armed and ready!")
    print("")
    
    print("=" * 60)
    print("[SUCCESS] DRONE IS READY!")
    print("=" * 60)
    print("")
    print("You can now control the drone:")
    print("")
    print("  # Take off")
    print("  client.takeoffAsync().join()")
    print("")
    print("  # Move to position (x, y, z, speed)")
    print("  client.moveToPositionAsync(5, 0, -5, 5).join()")
    print("")
    print("  # Land")
    print("  client.landAsync().join()")
    print("")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure Blocks.exe is fully loaded (wait 2-3 minutes)")
    print("2. You should see the Blocks 3D environment on screen")
    print("3. Try running this script again in a minute")


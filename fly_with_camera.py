"""
Drone Flight with Camera Capture
Flies to waypoints and captures images at each location
"""

import airsim
import time
import os

def capture_images(client, location_name, output_dir="captured_images"):
    """
    Capture scene, depth, and segmentation images
    
    Args:
        client: AirSim MultirotorClient
        location_name: Name for this capture location
        output_dir: Directory to save images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Capturing images at {location_name}...")
    
    # Request multiple image types
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene),           # RGB
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar),     # Depth
        airsim.ImageRequest("0", airsim.ImageType.Segmentation)     # Segmentation
    ])
    
    # Save images
    image_types = ["scene", "depth", "segmentation"]
    saved_files = []
    
    for idx, response in enumerate(responses):
        if response.pixels_as_float:
            # Convert float to uint8
            import numpy as np
            img_array = np.array(response.image_data_float, dtype=np.float32)
            img_array = (img_array * 255).astype(np.uint8)
            img_array = img_array.reshape(response.height, response.width, -1)
        else:
            img_array = response.image_data_uint8
        
        filename = f"{output_dir}/{location_name}_{image_types[idx]}.png"
        airsim.write_file(filename, response.image_data_uint8)
        saved_files.append(filename)
        print(f"    Saved: {filename}")
    
    return saved_files

def main():
    print("========================================")
    print("Drone Flight with Camera Capture")
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
    
    # Mission waypoints
    waypoints = [
        (5, 0, -5, "waypoint_1"),
        (5, 5, -5, "waypoint_2"),
        (0, 5, -5, "waypoint_3"),
        (0, 0, -5, "waypoint_4")
    ]
    
    try:
        # Takeoff
        print("\n=== Taking off ===")
        client.takeoffAsync().join()
        print("[OK] Airborne!")
        time.sleep(2)
        
        # Capture at takeoff position
        capture_images(client, "takeoff_position")
        
        # Visit each waypoint and capture images
        print("\n=== Starting mission ===")
        for idx, (x, y, z, name) in enumerate(waypoints):
            print(f"\n[{idx+1}/{len(waypoints)}] Flying to {name} ({x}, {y}, {z})")
            client.moveToPositionAsync(x, y, z, 3).join()
            
            # Hover and stabilize
            client.hoverAsync().join()
            time.sleep(1)
            
            # Capture images
            capture_images(client, name)
        
        # Return to start
        print("\n=== Returning to start ===")
        client.moveToPositionAsync(0, 0, -5, 3).join()
        capture_images(client, "final_position")
        
        # Land
        print("\n=== Landing ===")
        client.landAsync().join()
        print("[OK] Landed safely")
        
        print(f"\n[SUCCESS] Mission complete! Images saved to 'captured_images/' directory")
        
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

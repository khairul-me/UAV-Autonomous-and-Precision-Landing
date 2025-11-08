# capture_images.py
import airsim
import numpy as np
import cv2

client = airsim.MultirotorClient()
client.confirmConnection()

# AirSim supports multiple image types:
image_types = {
    'Scene': airsim.ImageType.Scene,          # RGB image
    'DepthPlanner': airsim.ImageType.DepthPlanner,  # Depth (grayscale)
    'DepthPerspective': airsim.ImageType.DepthPerspective,
    'DepthVis': airsim.ImageType.DepthVis,    # Depth visualization
    'Segmentation': airsim.ImageType.Segmentation,  # Semantic segmentation
}

def get_images():
    """Get all sensor data at once"""
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True),  # floating point
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
    ]
    
    responses = client.simGetImages(requests)
    
    # Process RGB image
    rgb_response = responses[0]
    rgb_1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    rgb_image = rgb_1d.reshape(rgb_response.height, rgb_response.width, 3)
    
    # Process Depth image (THIS IS WHAT YOU'LL ATTACK!)
    depth_response = responses[1]
    depth_image = airsim.list_to_2d_float_array(
        depth_response.image_data_float, 
        depth_response.width, 
        depth_response.height
    )
    
    # Process Segmentation
    seg_response = responses[2]
    seg_1d = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
    seg_image = seg_1d.reshape(seg_response.height, seg_response.width, 3)
    
    return rgb_image, depth_image, seg_image

# Test image capture
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

rgb, depth, seg = get_images()

print(f"RGB shape: {rgb.shape}")      # Should be (height, width, 3)
print(f"Depth shape: {depth.shape}")  # Should be (height, width)
print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")

# Save images
cv2.imwrite('rgb_image.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite('depth_image.png', (depth / depth.max() * 255).astype(np.uint8))
cv2.imwrite('seg_image.png', cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))

client.landAsync().join()


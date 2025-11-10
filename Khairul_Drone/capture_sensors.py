import os

import airsim
import cv2
import numpy as np


def save_rgb_image(client: airsim.MultirotorClient, output_dir: str) -> None:
    responses = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)]
    )
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    rgb = rgb.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite(os.path.join(output_dir, "rgb.png"), rgb)
    print("[OK] RGB image saved")


def save_depth_image(client: airsim.MultirotorClient, output_dir: str) -> None:
    responses = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True)]
    )
    depth = np.array(responses[0].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[0].height, responses[0].width)
    max_depth = np.max(depth) if depth.size else 1.0
    depth_normalized = (depth / max_depth * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "depth.png"), depth_normalized)
    print("[OK] Depth image saved")


def save_segmentation_image(client: airsim.MultirotorClient, output_dir: str) -> None:
    responses = client.simGetImages(
        [airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)]
    )
    seg = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    seg = seg.reshape(responses[0].height, responses[0].width, 3)
    cv2.imwrite(os.path.join(output_dir, "segmentation.png"), seg)
    print("[OK] Segmentation image saved")


def main() -> None:
    output_dir = "sensor_data"
    os.makedirs(output_dir, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    print("Capturing sensor data...")
    save_rgb_image(client, output_dir)
    save_depth_image(client, output_dir)
    save_segmentation_image(client, output_dir)

    imu = client.getImuData()
    print(f"[OK] IMU - Orientation: {imu.orientation}")
    print(f"  Linear Acceleration: {imu.linear_acceleration}")
    print(f"  Angular Velocity: {imu.angular_velocity}")

    gps = client.getGpsData()
    print(
        "[OK] GPS - Location:"
        f" ({gps.gnss.geo_point.latitude}, {gps.gnss.geo_point.longitude})"
    )

    baro = client.getBarometerData()
    print(f"[OK] Barometer - Altitude: {baro.altitude} m")

    print("\n[OK] All sensor data captured")
    print("Check the 'sensor_data' folder for images")


if __name__ == "__main__":
    main()


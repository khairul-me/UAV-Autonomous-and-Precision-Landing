#!/usr/bin/env python3
"""
Verify enhanced AirSim configuration.

Checks:
  - All camera streams (RGB, DepthPlanar, Segmentation)
  - Core sensors (IMU, GPS, Barometer, Magnetometer, LiDAR, Distance)
  - Update rates over a fixed interval
  - Sample image capture for manual review
"""

import time
from pathlib import Path

import airsim
import cv2
import numpy as np


class ConfigVerifier:
    """Run validation routines against the enhanced AirSim sensor setup."""

    def __init__(self):
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        print("[OK] Connected\n")

    def verify_cameras(self):
        """Capture from every configured camera and report the basic stats."""
        print("=" * 60)
        print("CAMERA VERIFICATION")
        print("=" * 60)

        cameras = ["front_center", "front_left", "front_right", "bottom"]
        image_types = {
            airsim.ImageType.Scene: "Scene (RGB)",
            airsim.ImageType.DepthPlanar: "DepthPlanar",
            airsim.ImageType.Segmentation: "Segmentation",
        }

        results = {}

        for camera in cameras:
            print(f"\nCamera: {camera}")
            print("-" * 40)
            camera_results = {}

            for img_type, type_name in image_types.items():
                try:
                    response = self.client.simGetImages(
                        [airsim.ImageRequest(camera, img_type, pixels_as_float=False, compress=False)]
                    )[0]

                    if response.height > 0 and response.width > 0:
                        print(f"  [OK] {type_name}: {response.width}x{response.height}")

                        if img_type == airsim.ImageType.Scene:
                            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                            img_rgb = img.reshape(response.height, response.width, 3)
                            camera_results[type_name] = {
                                "success": True,
                                "shape": img_rgb.shape,
                                "dtype": str(img_rgb.dtype),
                                "range": (int(img_rgb.min()), int(img_rgb.max())),
                            }
                        elif img_type == airsim.ImageType.DepthPlanar:
                            img = np.array(response.image_data_float, dtype=np.float32)
                            img_depth = img.reshape(response.height, response.width)
                            camera_results[type_name] = {
                                "success": True,
                                "shape": img_depth.shape,
                                "dtype": str(img_depth.dtype),
                                "range": (float(np.nanmin(img_depth)), float(np.nanmax(img_depth))),
                            }
                        else:
                            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                            img_seg = img.reshape(response.height, response.width, 3)
                            camera_results[type_name] = {
                                "success": True,
                                "shape": img_seg.shape,
                                "dtype": str(img_seg.dtype),
                                "unique_values": int(np.unique(img_seg).size),
                            }
                    else:
                        print(f"  [FAIL] {type_name}: no data returned")
                        camera_results[type_name] = {"success": False, "error": "no data"}
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"  [FAIL] {type_name}: {exc}")
                    camera_results[type_name] = {"success": False, "error": str(exc)}

            results[camera] = camera_results

        return results

    def verify_sensors(self):
        """Read core vehicle sensors and report values."""
        print("\n" + "=" * 60)
        print("SENSOR VERIFICATION")
        print("=" * 60)

        results = {}

        # IMU
        print("\nSensor: IMU")
        print("-" * 40)
        try:
            imu_data = self.client.getImuData()
            print(f"  [OK] Orientation: ({imu_data.orientation.x_val:.3f}, "
                  f"{imu_data.orientation.y_val:.3f}, {imu_data.orientation.z_val:.3f})")
            print(f"  [OK] Angular velocity: ({imu_data.angular_velocity.x_val:.3f}, "
                  f"{imu_data.angular_velocity.y_val:.3f}, {imu_data.angular_velocity.z_val:.3f})")
            print(f"  [OK] Linear acceleration: ({imu_data.linear_acceleration.x_val:.3f}, "
                  f"{imu_data.linear_acceleration.y_val:.3f}, {imu_data.linear_acceleration.z_val:.3f})")
            results["IMU"] = {"success": True}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] IMU error: {exc}")
            results["IMU"] = {"success": False, "error": str(exc)}

        # GPS
        print("\nSensor: GPS")
        print("-" * 40)
        try:
            gps_data = self.client.getGpsData()
            geo = gps_data.gnss.geo_point
            print(f"  [OK] Latitude: {geo.latitude:.6f}")
            print(f"  [OK] Longitude: {geo.longitude:.6f}")
            print(f"  [OK] Altitude: {geo.altitude:.2f} m")
            results["GPS"] = {"success": True}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] GPS error: {exc}")
            results["GPS"] = {"success": False, "error": str(exc)}

        # Barometer
        print("\nSensor: Barometer")
        print("-" * 40)
        try:
            baro_data = self.client.getBarometerData()
            print(f"  [OK] Altitude: {baro_data.altitude:.2f} m")
            print(f"  [OK] Pressure: {baro_data.pressure:.2f} Pa")
            results["Barometer"] = {"success": True}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] Barometer error: {exc}")
            results["Barometer"] = {"success": False, "error": str(exc)}

        # Magnetometer
        print("\nSensor: Magnetometer")
        print("-" * 40)
        try:
            mag_data = self.client.getMagnetometerData()
            field = mag_data.magnetic_field_body
            print(f"  [OK] Magnetic field: ({field.x_val:.3f}, {field.y_val:.3f}, {field.z_val:.3f})")
            results["Magnetometer"] = {"success": True}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] Magnetometer error: {exc}")
            results["Magnetometer"] = {"success": False, "error": str(exc)}

        # LiDAR
        print("\nSensor: LiDAR")
        print("-" * 40)
        try:
            lidar_data = self.client.getLidarData()
            if lidar_data.point_cloud:
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                print(f"  [OK] Points: {points.shape[0]}")
                print(
                    "  [OK] Range: "
                    f"X({points[:, 0].min():.2f} to {points[:, 0].max():.2f}), "
                    f"Y({points[:, 1].min():.2f} to {points[:, 1].max():.2f}), "
                    f"Z({points[:, 2].min():.2f} to {points[:, 2].max():.2f})"
                )
                results["LiDAR"] = {"success": True, "num_points": int(points.shape[0])}
            else:
                print("  [WARN] No LiDAR points captured")
                results["LiDAR"] = {"success": True, "num_points": 0}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] LiDAR error: {exc}")
            results["LiDAR"] = {"success": False, "error": str(exc)}

        # Distance sensor
        print("\nSensor: Distance")
        print("-" * 40)
        try:
            dist_data = self.client.getDistanceSensorData()
            print(f"  [OK] Distance: {dist_data.distance:.2f} m "
                  f"(range {dist_data.min_distance:.2f} - {dist_data.max_distance:.2f})")
            results["Distance"] = {"success": True}
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [FAIL] Distance sensor error: {exc}")
            results["Distance"] = {"success": False, "error": str(exc)}

        return results

    def test_update_rate(self, duration=5.0):
        """Poll sensors for a fixed window and report call frequency."""
        print("\n" + "=" * 60)
        print("UPDATE RATE TEST")
        print("=" * 60)
        print(f"Testing for {duration} seconds...\n")

        start_time = time.time()
        counts = {"camera": 0, "imu": 0, "gps": 0, "lidar": 0}

        while time.time() - start_time < duration:
            try:
                self.client.simGetImages(
                    [airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, False, False)]
                )
                counts["camera"] += 1
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                self.client.getImuData()
                counts["imu"] += 1
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                self.client.getGpsData()
                counts["gps"] += 1
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                self.client.getLidarData()
                counts["lidar"] += 1
            except Exception:  # pylint: disable=broad-except
                pass

        elapsed = time.time() - start_time
        rates = {sensor: count / elapsed for sensor, count in counts.items()}

        print("Update rates (Hz)")
        print("-" * 40)
        for sensor, rate in rates.items():
            print(f"  {sensor.upper()}: {rate:.1f}")

        return rates

    def save_sample_images(self, output_dir="outputs/config_verification"):
        """Dump RGB and depth frames from all cameras for manual inspection."""
        print("\n" + "=" * 60)
        print("CAPTURING SAMPLE IMAGES")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cameras = ["front_center", "front_left", "front_right", "bottom"]

        for camera in cameras:
            print(f"\nCapturing camera: {camera}")

            try:
                rgb_resp = self.client.simGetImages(
                    [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
                )[0]
                img = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).reshape(
                    rgb_resp.height, rgb_resp.width, 3
                )
                rgb_path = output_path / f"{camera}_rgb.png"
                cv2.imwrite(str(rgb_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"  [OK] Saved RGB -> {rgb_path}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  [FAIL] RGB capture failed: {exc}")

            try:
                depth_resp = self.client.simGetImages(
                    [airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, False, False)]
                )[0]
                depth = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(
                    depth_resp.height, depth_resp.width
                )
                depth_vis = np.clip(depth, 0, 100)
                depth_vis = (depth_vis / 100.0 * 255).astype(np.uint8)
                depth_path = output_path / f"{camera}_depth.png"
                cv2.imwrite(str(depth_path), depth_vis)
                print(f"  [OK] Saved Depth -> {depth_path}")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  [FAIL] Depth capture failed: {exc}")

        print(f"\n[OK] Images saved under {output_path.resolve()}")

    @staticmethod
    def generate_report(camera_results, sensor_results, update_rates):
        """Aggregate verification results and print a simple summary."""
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        print("\nCameras:")
        for camera, results in camera_results.items():
            for img_type, outcome in results.items():
                total_tests += 1
                if outcome.get("success"):
                    passed_tests += 1
                    status = "[OK]"
                else:
                    status = "[FAIL]"
                print(f"  {status} {camera} - {img_type}")

        print("\nSensors:")
        for sensor, outcome in sensor_results.items():
            total_tests += 1
            if outcome.get("success"):
                passed_tests += 1
                status = "[OK]"
            else:
                status = "[FAIL]"
            print(f"  {status} {sensor}")

        print("\nUpdate rates (Hz):")
        for sensor, rate in update_rates.items():
            print(f"  {sensor.upper()}: {rate:.1f}")

        success_rate = (passed_tests / total_tests) * 100 if total_tests else 0.0
        print("\n" + "=" * 60)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        print("=" * 60)

        if success_rate == 100.0:
            print("\n[OK] Configuration verification PASSED")
            print("  - All cameras operational")
            print("  - All sensors functional")
            print("  - System ready for training")
        elif success_rate >= 80.0:
            print("\n[WARN] Configuration verification mostly passed; check warnings above.")
        else:
            print("\n[FAIL] Configuration verification failed; review errors above.")

        return success_rate >= 80.0


def main():
    """Entrypoint for running the configuration verification."""
    print("=" * 60)
    print("AIRSIM ENHANCED CONFIGURATION VERIFIER")
    print("=" * 60)
    print("\nThis routine will test:")
    print("  - All camera configurations")
    print("  - All core sensor readings")
    print("  - Sensor update rates")
    print("  - Sample image capture")
    print("\nEnsure AirSim is running with settings_enhanced.json before continuing.\n")

    input("Press Enter to start verification...")

    verifier = ConfigVerifier()
    camera_results = verifier.verify_cameras()
    sensor_results = verifier.verify_sensors()
    update_rates = verifier.test_update_rate()
    verifier.save_sample_images()

    success = verifier.generate_report(camera_results, sensor_results, update_rates)

    if success:
        print("\n[OK] Configuration verification complete. Proceed to enhanced observation setup.")
    else:
        print("\n[FAIL] Configuration requires attention before proceeding.")

    return success


if __name__ == "__main__":
    main()


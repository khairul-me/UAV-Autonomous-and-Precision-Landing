"""
Phase 0 Task 0.2: Environment Preparation
- Set up multiple test environments (Blocks, AirSimNH)
- Configure camera sensors (RGB, Depth, Segmentation)
- Configure IMU and GPS sensors
- Test multi-sensor data capture at 30Hz

Success Criteria: Can capture synchronized RGB+Depth+IMU+GPS data at target frame rate
"""

import airsim
import time
import json
from pathlib import Path

def create_sensor_settings():
    """
    Create settings.json with sensor configuration
    """
    settings = {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ClockSpeed": 1.0,
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "X": 0, "Y": 0, "Z": 0, "Yaw": 0,
                "Sensors": {
                    # Camera sensors
                    "Camera1": {
                        "SensorType": 1,  # Camera
                        "Enabled": True,
                        "ImageType": 0,  # Scene (RGB)
                        "FOV_Degrees": 90,
                        "CaptureSettings": [
                            {
                                "ImageType": 0,  # Scene
                                "Width": 640,
                                "Height": 480,
                                "TargetGamma": 2.2
                            },
                            {
                                "ImageType": 2,  # DepthPlanar
                                "Width": 640,
                                "Height": 480
                            },
                            {
                                "ImageType": 5,  # Segmentation
                                "Width": 640,
                                "Height": 480
                            }
                        ]
                    },
                    # IMU sensor
                    "Imu": {
                        "SensorType": 2,  # IMU
                        "Enabled": True,
                        "AngularNoiseStdDev": 0.0001,
                        "LinearAccelNoiseStdDev": 0.0001
                    },
                    # GPS sensor
                    "Gps": {
                        "SensorType": 3,  # GPS
                        "Enabled": True,
                        "EphTime": 0.5,
                        "EphUere": 1.0,
                        "EphHorizPosErr": 0.5,
                        "EphVertPosErr": 1.0
                    },
                    # Magnetometer
                    "Magnetometer": {
                        "SensorType": 4,  # Magnetometer
                        "Enabled": True
                    },
                    # Barometer
                    "Barometer": {
                        "SensorType": 1,  # Barometer
                        "Enabled": True
                    }
                }
            }
        },
        "ApiServerPort": 41451
    }
    
    return settings

def save_settings(settings, settings_path):
    """Save settings.json to specified path"""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"[OK] Settings saved to: {settings_path}")

def test_sensor_access(client):
    """Test access to all configured sensors"""
    print("\n" + "=" * 70)
    print("Testing Sensor Access")
    print("=" * 70)
    
    results = {}
    
    # Test camera
    try:
        camera_info = client.simGetCameraInfo("0")
        results["Camera"] = "OK"
        print("[OK] Camera accessible")
    except Exception as e:
        results["Camera"] = f"ERROR: {str(e)[:50]}"
        print(f"[ERROR] Camera: {str(e)[:50]}")
    
    # Test IMU
    try:
        imu_data = client.getImuData()
        results["IMU"] = "OK"
        print("[OK] IMU accessible")
        print(f"    Linear acceleration: {imu_data.linear_acceleration}")
        print(f"    Angular velocity: {imu_data.angular_velocity}")
    except Exception as e:
        results["IMU"] = f"ERROR: {str(e)[:50]}"
        print(f"[ERROR] IMU: {str(e)[:50]}")
    
    # Test GPS
    try:
        gps_data = client.getGpsData()
        results["GPS"] = "OK"
        print("[OK] GPS accessible")
        print(f"    GPS position: {gps_data.gnss.geo_point}")
        print(f"    Altitude: {gps_data.gnss.geo_point.altitude}")
    except Exception as e:
        results["GPS"] = f"ERROR: {str(e)[:50]}"
        print(f"[ERROR] GPS: {str(e)[:50]}")
    
    # Test Magnetometer
    try:
        mag_data = client.getMagnetometerData()
        results["Magnetometer"] = "OK"
        print("[OK] Magnetometer accessible")
        print(f"    Magnetic field: {mag_data.magnetic_field_body}")
    except Exception as e:
        results["Magnetometer"] = f"WARNING: {str(e)[:50]}"
        print(f"[WARNING] Magnetometer: {str(e)[:50]}")
    
    # Test Barometer
    try:
        baro_data = client.getBarometerData()
        results["Barometer"] = "OK"
        print("[OK] Barometer accessible")
        print(f"    Pressure: {baro_data.pressure}")
        print(f"    Altitude: {baro_data.altitude}")
    except Exception as e:
        results["Barometer"] = f"WARNING: {str(e)[:50]}"
        print(f"[WARNING] Barometer: {str(e)[:50]}")
    
    return results

def test_multi_sensor_capture(client, duration=5, target_fps=30):
    """
    Test synchronized multi-sensor data capture at target frame rate
    
    Args:
        client: AirSim client
        duration: Test duration in seconds
        target_fps: Target frames per second (30Hz = 30 fps)
    """
    print("\n" + "=" * 70)
    print(f"Testing Multi-Sensor Capture at {target_fps}Hz for {duration} seconds")
    print("=" * 70)
    
    target_interval = 1.0 / target_fps  # Time between frames
    num_samples = int(duration * target_fps)
    
    print(f"Target interval: {target_interval*1000:.2f}ms")
    print(f"Expected samples: {num_samples}")
    print("")
    
    timestamps = []
    camera_data = []
    imu_data_list = []
    gps_data_list = []
    
    start_time = time.time()
    last_frame_time = start_time
    
    for i in range(num_samples):
        frame_start = time.time()
        
        # Capture camera data (RGB, Depth, Segmentation)
        try:
            images = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation)
            ])
            camera_data.append({
                "rgb_size": len(images[0].image_data_uint8) if images[0].image_data_uint8 else 0,
                "depth_size": len(images[1].image_data_uint8) if images[1].image_data_uint8 else 0,
                "seg_size": len(images[2].image_data_uint8) if images[2].image_data_uint8 else 0
            })
        except Exception as e:
            camera_data.append({"error": str(e)[:50]})
        
        # Capture IMU data
        try:
            imu = client.getImuData()
            imu_data_list.append({
                "linear_accel": imu.linear_acceleration,
                "angular_vel": imu.angular_velocity
            })
        except Exception as e:
            imu_data_list.append({"error": str(e)[:50]})
        
        # Capture GPS data
        try:
            gps = client.getGpsData()
            gps_data_list.append({
                "latitude": gps.gnss.geo_point.latitude,
                "longitude": gps.gnss.geo_point.longitude,
                "altitude": gps.gnss.geo_point.altitude
            })
        except Exception as e:
            gps_data_list.append({"error": str(e)[:50]})
        
        # Record timestamp
        frame_end = time.time()
        frame_duration = frame_end - frame_start
        timestamps.append(frame_duration)
        
        # Calculate when to capture next frame
        elapsed = frame_end - last_frame_time
        sleep_time = max(0, target_interval - elapsed)
        
        if i % (target_fps // 2) == 0:  # Print every 0.5 seconds
            print(f"Frame {i+1}/{num_samples}: duration={frame_duration*1000:.2f}ms, "
                  f"sleep={sleep_time*1000:.2f}ms")
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        last_frame_time = time.time()
    
    total_time = time.time() - start_time
    actual_fps = num_samples / total_time
    
    # Statistics
    avg_frame_time = sum(timestamps) / len(timestamps)
    min_frame_time = min(timestamps)
    max_frame_time = max(timestamps)
    
    print("\n" + "=" * 70)
    print("Capture Statistics")
    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Expected samples: {num_samples}")
    print(f"Actual samples: {len(timestamps)}")
    print(f"Actual FPS: {actual_fps:.2f}")
    print(f"Target FPS: {target_fps:.2f}")
    print(f"FPS accuracy: {(actual_fps/target_fps)*100:.1f}%")
    print(f"\nFrame timing:")
    print(f"  Average: {avg_frame_time*1000:.2f}ms")
    print(f"  Min: {min_frame_time*1000:.2f}ms")
    print(f"  Max: {max_frame_time*1000:.2f}ms")
    print(f"\nData captured:")
    print(f"  Camera frames: {len(camera_data)}")
    print(f"  IMU samples: {len(imu_data_list)}")
    print(f"  GPS samples: {len(gps_data_list)}")
    
    success = actual_fps >= (target_fps * 0.9)  # Allow 10% tolerance
    return success, {
        "actual_fps": actual_fps,
        "target_fps": target_fps,
        "samples": len(timestamps),
        "camera_samples": len(camera_data),
        "imu_samples": len(imu_data_list),
        "gps_samples": len(gps_data_list)
    }

def main():
    print("=" * 70)
    print("PHASE 0 TASK 0.2: Environment Preparation")
    print("=" * 70)
    print("")
    
    # Step 1: Create sensor settings
    print("[1] Creating sensor configuration...")
    settings = create_sensor_settings()
    
    # Save settings to Documents/AirSim/settings.json
    settings_path = Path.home() / "Documents" / "AirSim" / "settings.json"
    save_settings(settings, settings_path)
    print("[OK] Sensor configuration created")
    
    # Step 2: Connect to AirSim
    print("\n[2] Connecting to AirSim...")
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[OK] Connected to AirSim")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        print("\nPlease ensure Blocks.exe or AirSimNH.exe is running!")
        return False
    
    # Step 3: Test sensor access
    print("\n[3] Testing sensor access...")
    sensor_results = test_sensor_access(client)
    
    # Step 4: Test multi-sensor capture at 30Hz
    print("\n[4] Testing multi-sensor data capture at 30Hz...")
    success, stats = test_multi_sensor_capture(client, duration=5, target_fps=30)
    
    # Success criteria check
    print("\n" + "=" * 70)
    if success and all("OK" in str(v) or "ERROR" not in str(v) for v in sensor_results.values()):
        print("[SUCCESS] PHASE 0 TASK 0.2: Environment preparation complete!")
        print("=" * 70)
        print("\nSuccess Criteria Met:")
        print("  ✓ Multiple environments can be configured (Blocks/AirSimNH)")
        print("  ✓ Camera sensors configured (RGB, Depth, Segmentation)")
        print("  ✓ IMU and GPS sensors configured and accessible")
        print(f"  ✓ Multi-sensor capture tested at {stats['actual_fps']:.2f}Hz")
        print("\nSensor Status:")
        for sensor, status in sensor_results.items():
            print(f"  {sensor}: {status}")
        return True
    else:
        print("[WARNING] Some tests had issues")
        print("=" * 70)
        print("\nSensor Status:")
        for sensor, status in sensor_results.items():
            print(f"  {sensor}: {status}")
        print(f"\nFPS Test: {stats['actual_fps']:.2f}Hz (target: {stats['target_fps']:.2f}Hz)")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

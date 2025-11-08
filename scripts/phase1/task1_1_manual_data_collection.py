"""
Phase 1 Task 1.1: Manual Control & Data Collection
- Implement keyboard/joystick control for manual flight
- Collect 10,000+ RGB images at varying altitudes (5-30m), viewpoints, obstacle distances
- Include corresponding depth maps, GPS coordinates, IMU readings, obstacle labels
- Split dataset: 70% train, 15% validation, 15% test
Success Criteria: Clean, labeled dataset with obstacle/safe regions
"""

import airsim
import numpy as np
import cv2
import os
import json
import csv
import time
import random
from pathlib import Path
from datetime import datetime
import msvcrt
import sys

class ManualDataCollection:
    def __init__(self, output_dir="datasets/manual_collection", target_images=10000):
        self.client = None
        self.output_dir = Path(output_dir)
        self.target_images = target_images
        self.image_count = 0
        
        # Create directory structure
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            for subdir in ["rgb", "depth", "segmentation", "labels"]:
                (split_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Metadata files
        self.metadata_file = self.output_dir / "metadata.csv"
        self.init_metadata_file()
        
    def init_metadata_file(self):
        """Initialize metadata CSV file"""
        with open(self.metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_id', 'split', 'timestamp', 'altitude', 'position_x', 'position_y', 'position_z',
                'gps_lat', 'gps_lon', 'gps_alt', 'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z', 'obstacle_count', 'safe_regions'
            ])
    
    def connect(self):
        """Connect to AirSim"""
        print("[INFO] Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[OK] Connected and armed!")
    
    def capture_frame(self, split='train'):
        """Capture a single frame with all sensor data"""
        try:
            timestamp = time.time()
            
            # Capture images
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ])
            
            # Process RGB
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Process Depth
            depth_array = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_array = depth_array.reshape(responses[1].height, responses[1].width)
            
            # Process Segmentation
            seg1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            segmentation = seg1d.reshape(responses[2].height, responses[2].width, 3)
            
            # Get IMU data
            imu = self.client.getImuData()
            
            # Get GPS data
            gps = self.client.getGpsData()
            
            # Get drone state
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            altitude = abs(pos.z_val)
            
            # Determine split (70/15/15)
            if split == 'auto':
                rand = random.random()
                if rand < 0.70:
                    split = 'train'
                elif rand < 0.85:
                    split = 'val'
                else:
                    split = 'test'
            
            split_dir = self.output_dir / split
            
            # Save images
            image_id = f"{self.image_count:06d}"
            cv2.imwrite(str(split_dir / "rgb" / f"{image_id}.png"), rgb_bgr)
            np.save(str(split_dir / "depth" / f"{image_id}.npy"), depth_array)
            cv2.imwrite(str(split_dir / "segmentation" / f"{image_id}.png"), segmentation)
            
            # Analyze segmentation for obstacles (simplified - count non-ground pixels)
            obstacle_count = np.sum(np.any(segmentation != segmentation[0, 0], axis=2))
            safe_regions = responses[0].width * responses[0].height - obstacle_count
            
            # Save metadata
            with open(self.metadata_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    image_id, split, timestamp, altitude,
                    pos.x_val, pos.y_val, pos.z_val,
                    gps.gnss.geo_point.latitude, gps.gnss.geo_point.longitude, gps.gnss.geo_point.altitude,
                    imu.linear_acceleration.x_val, imu.linear_acceleration.y_val, imu.linear_acceleration.z_val,
                    imu.angular_velocity.x_val, imu.angular_velocity.y_val, imu.angular_velocity.z_val,
                    obstacle_count, safe_regions
                ])
            
            self.image_count += 1
            return True
            
        except Exception as e:
            print(f"[ERROR] Frame capture failed: {e}")
            return False
    
    def get_key(self):
        """Get keyboard input (Windows)"""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            try:
                return key.decode('utf-8').lower()
            except:
                return None
        return None
    
    def fly_and_collect(self):
        """Fly manually and collect data"""
        print("\n" + "=" * 60)
        print("MANUAL DATA COLLECTION MODE")
        print("=" * 60)
        print(f"Target: {self.target_images} images")
        print(f"Current: {self.image_count} images")
        print("\nControls:")
        print("  W/A/S/D - Move forward/left/back/right")
        print("  Q/E - Move up/down")
        print("  R - Rotate right, L - Rotate left")
        print("  C - Capture frame")
        print("  T - Takeoff")
        print("  G - Land")
        print("  ESC - Exit")
        print("=" * 60 + "\n")
        
        velocity = 5.0  # m/s
        duration = 0.5  # seconds
        
        while self.image_count < self.target_images:
            key = self.get_key()
            
            if key == 'w':
                self.client.moveByVelocityAsync(velocity, 0, 0, duration).join()
            elif key == 's':
                self.client.moveByVelocityAsync(-velocity, 0, 0, duration).join()
            elif key == 'a':
                self.client.moveByVelocityAsync(0, -velocity, 0, duration).join()
            elif key == 'd':
                self.client.moveByVelocityAsync(0, velocity, 0, duration).join()
            elif key == 'q':
                self.client.moveByVelocityAsync(0, 0, -velocity/2, duration).join()
            elif key == 'e':
                self.client.moveByVelocityAsync(0, 0, velocity/2, duration).join()
            elif key == 't':
                self.client.takeoffAsync().join()
            elif key == 'g':
                self.client.landAsync().join()
            elif key == 'c':
                if self.capture_frame('auto'):
                    print(f"[CAPTURED] Image {self.image_count}/{self.target_images}")
                    # Periodically capture while flying
                    time.sleep(0.1)
                    self.capture_frame('auto')
            elif key == '\x1b':  # ESC
                break
            
            # Auto-capture periodically
            if self.image_count % 10 == 0 and self.image_count > 0:
                self.capture_frame('auto')
            
            time.sleep(0.1)
        
        print(f"\n[SUCCESS] Collected {self.image_count} images!")
        print(f"Dataset saved to: {self.output_dir}")
    
    def cleanup(self):
        """Cleanup and disarm"""
        if self.client:
            try:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass

def main():
    collector = ManualDataCollection(target_images=10000)
    
    try:
        collector.connect()
        collector.fly_and_collect()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()

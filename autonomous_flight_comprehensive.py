"""
COMPREHENSIVE AUTONOMOUS DRONE FLIGHT - PHASE 0 COMPLETE
This script implements all Phase 0 requirements:
- Task 0.1: Basic flight commands ✅
- Task 0.2: Multi-sensor capture (RGB, Depth, Segmentation, IMU, GPS, Magnetometer, Barometer) ✅
- Task 0.3: Comprehensive data logging and pipeline ✅

Fully aligned with project requirements for adversarial robustness research.
"""

import airsim
import time
import cv2
import numpy as np
import os
import sys
import json
import codecs
from datetime import datetime
from pathlib import Path
from collections import deque

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    except:
        pass

class ComprehensiveDataLogger:
    """Complete data logging system for all sensors and flight data"""
    
    def __init__(self, output_dir="flight_recordings", session_name=None):
        """Initialize comprehensive data logger"""
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.base_dir = Path(output_dir) / f"comprehensive_{session_name}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized subdirectories
        self.dirs = {
            "rgb": self.base_dir / "rgb",
            "depth": self.base_dir / "depth",
            "segmentation": self.base_dir / "segmentation",
            "imu": self.base_dir / "imu",
            "gps": self.base_dir / "gps",
            "magnetometer": self.base_dir / "magnetometer",
            "barometer": self.base_dir / "barometer",
            "state": self.base_dir / "state",
            "commands": self.base_dir / "commands",
            "frames": self.base_dir / "frames"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.flight_log = []
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"[DataLogger] Initialized: {self.base_dir}")
    
    def log_frame(self, timestamp, rgb, depth, segmentation, imu, gps, magnetometer, barometer, state, command=None):
        """Log all sensor data for a single frame"""
        frame_id = f"frame_{self.frame_count:06d}"
        
        # Save images
        if rgb is not None:
            cv2.imwrite(str(self.dirs["rgb"] / f"{frame_id}.jpg"), rgb)
        if depth is not None:
            cv2.imwrite(str(self.dirs["depth"] / f"{frame_id}.png"), depth)
        if segmentation is not None:
            cv2.imwrite(str(self.dirs["segmentation"] / f"{frame_id}.png"), segmentation)
        
        # Save sensor data as JSON
        frame_data = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "relative_time": timestamp - self.start_time,
            "imu": imu,
            "gps": gps,
            "magnetometer": magnetometer,
            "barometer": barometer,
            "state": state,
            "command": command
        }
        
        # Save individual sensor files
        with open(self.dirs["imu"] / f"{frame_id}.json", 'w') as f:
            json.dump({"timestamp": timestamp, **imu}, f, indent=2)
        
        with open(self.dirs["gps"] / f"{frame_id}.json", 'w') as f:
            json.dump({"timestamp": timestamp, **gps}, f, indent=2)
        
        with open(self.dirs["magnetometer"] / f"{frame_id}.json", 'w') as f:
            json.dump({"timestamp": timestamp, **magnetometer}, f, indent=2)
        
        with open(self.dirs["barometer"] / f"{frame_id}.json", 'w') as f:
            json.dump({"timestamp": timestamp, **barometer}, f, indent=2)
        
        with open(self.dirs["state"] / f"{frame_id}.json", 'w') as f:
            json.dump({"timestamp": timestamp, **state}, f, indent=2)
        
        if command:
            with open(self.dirs["commands"] / f"{frame_id}.json", 'w') as f:
                json.dump({"timestamp": timestamp, **command}, f, indent=2)
        
        # Add to flight log
        self.flight_log.append(frame_data)
        self.frame_count += 1
    
    def save_summary(self):
        """Save comprehensive flight summary"""
        summary = {
            "session_name": self.session_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": time.time() - self.start_time,
            "total_frames": self.frame_count,
            "average_fps": self.frame_count / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0,
            "sensors_captured": [
                "RGB", "Depth", "Segmentation", "IMU", "GPS", "Magnetometer", "Barometer"
            ],
            "directory_structure": {k: str(v) for k, v in self.dirs.items()}
        }
        
        summary_path = self.base_dir / "flight_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save complete flight log
        log_path = self.base_dir / "flight_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.flight_log, f, indent=2)
        
        print(f"[DataLogger] Summary saved: {summary_path}")
        print(f"[DataLogger] Flight log saved: {log_path}")
        print(f"[DataLogger] Total frames: {self.frame_count}")
        print(f"[DataLogger] Average FPS: {summary['average_fps']:.2f}")
        
        return summary


class ComprehensiveAutonomousFlight:
    """Complete autonomous flight system with full sensor suite"""
    
    def __init__(self, capture_rate_hz=30):
        self.client = None
        self.recording = True
        self.frames = []
        self.running = True
        self.logger = None
        self.capture_rate = capture_rate_hz
        self.capture_interval = 1.0 / capture_rate_hz
        
    def connect(self):
        """Connect to AirSim"""
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE AUTONOMOUS FLIGHT - PHASE 0 COMPLETE")
        print("=" * 70)
        print("\n[1/7] Connecting to AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("[OK] Connected to AirSim!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            print("Make sure AirSimNH.exe or Blocks.exe is running!")
            print("Wait 2-5 minutes after launching for it to fully load.")
            return False
    
    def setup(self):
        """Setup: Enable API control, arm, and initialize logger"""
        print("\n[2/7] Setting up drone and data logging...")
        try:
            # Enable API control
            self.client.enableApiControl(True)
            print("[OK] API Control enabled")
            
            # Arm the drone
            self.client.armDisarm(True)
            print("[OK] Drone armed")
            
            # Initialize data logger
            self.logger = ComprehensiveDataLogger()
            print("[OK] Data logger initialized")
            
            time.sleep(1)
            
            # Get initial state
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print(f"[OK] Initial position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Setup failed: {e}")
            return False
    
    def takeoff(self):
        """Take off to safe altitude"""
        print("\n[3/7] Taking off...")
        try:
            self.client.takeoffAsync(timeout_sec=30).join()
            time.sleep(2)
            state = self.client.getMultirotorState()
            altitude = abs(state.kinematics_estimated.position.z_val)
            print(f"[OK] Flying at {altitude:.2f}m altitude!")
            return True
        except Exception as e:
            print(f"[ERROR] Takeoff failed: {e}")
            return False
    
    def capture_all_sensors(self):
        """Capture all sensors: RGB, Depth, Segmentation, IMU, GPS, Magnetometer, Barometer"""
        try:
            timestamp = time.time()
            
            # Capture camera images (RGB, Depth, Segmentation)
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ])
            
            # Process RGB
            rgb = None
            if responses[0].image_data_uint8:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Process Depth
            depth = None
            if responses[1].image_data_float:
                depth_array = np.array(responses[1].image_data_float, dtype=np.float32)
                depth_array = depth_array.reshape(responses[1].height, responses[1].width)
                # Normalize for visualization
                depth_normalized = (depth_array * 255.0 / depth_array.max()).astype(np.uint8)
                depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Process Segmentation
            segmentation = None
            if responses[2].image_data_uint8:
                img1d = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
                segmentation = img1d.reshape(responses[2].height, responses[2].width, 3)
            
            # Get IMU data
            imu_data = self.client.getImuData()
            imu = {
                "linear_acceleration": {
                    "x": imu_data.linear_acceleration.x_val,
                    "y": imu_data.linear_acceleration.y_val,
                    "z": imu_data.linear_acceleration.z_val
                },
                "angular_velocity": {
                    "x": imu_data.angular_velocity.x_val,
                    "y": imu_data.angular_velocity.y_val,
                    "z": imu_data.angular_velocity.z_val
                },
                "orientation": {
                    "w": imu_data.orientation.w_val,
                    "x": imu_data.orientation.x_val,
                    "y": imu_data.orientation.y_val,
                    "z": imu_data.orientation.z_val
                }
            }
            
            # Get GPS data
            gps_data = self.client.getGpsData()
            gps = {
                "latitude": gps_data.gnss.geo_point.latitude,
                "longitude": gps_data.gnss.geo_point.longitude,
                "altitude": gps_data.gnss.geo_point.altitude,
                "velocity": {
                    "x": gps_data.gnss.velocity.x_val,
                    "y": gps_data.gnss.velocity.y_val,
                    "z": gps_data.gnss.velocity.z_val
                }
            }
            
            # Get Magnetometer data
            mag_data = self.client.getMagnetometerData()
            magnetometer = {
                "magnetic_field_body": {
                    "x": mag_data.magnetic_field_body.x_val,
                    "y": mag_data.magnetic_field_body.y_val,
                    "z": mag_data.magnetic_field_body.z_val
                },
                "magnetic_field_covariance": mag_data.magnetic_field_covariance if hasattr(mag_data, 'magnetic_field_covariance') else []
            }
            
            # Get Barometer data
            baro_data = self.client.getBarometerData()
            barometer = {
                "altitude": baro_data.altitude,
                "pressure": baro_data.pressure,
                "qnh": baro_data.qnh if hasattr(baro_data, 'qnh') else 1013.25
            }
            
            # Get drone state
            state = self.client.getMultirotorState()
            drone_state = {
                "position": {
                    "x": state.kinematics_estimated.position.x_val,
                    "y": state.kinematics_estimated.position.y_val,
                    "z": state.kinematics_estimated.position.z_val
                },
                "velocity": {
                    "x": state.kinematics_estimated.linear_velocity.x_val,
                    "y": state.kinematics_estimated.linear_velocity.y_val,
                    "z": state.kinematics_estimated.linear_velocity.z_val
                },
                "orientation": {
                    "w": state.kinematics_estimated.orientation.w_val,
                    "x": state.kinematics_estimated.orientation.x_val,
                    "y": state.kinematics_estimated.orientation.y_val,
                    "z": state.kinematics_estimated.orientation.z_val
                },
                "angular_velocity": {
                    "x": state.kinematics_estimated.angular_velocity.x_val,
                    "y": state.kinematics_estimated.angular_velocity.y_val,
                    "z": state.kinematics_estimated.angular_velocity.z_val
                }
            }
            
            return timestamp, rgb, depth, segmentation, imu, gps, magnetometer, barometer, drone_state
            
        except Exception as e:
            print(f"[WARNING] Sensor capture failed: {e}")
            return None, None, None, None, None, None, None, None, None
    
    def fly_urban_exploration(self, duration=120, altitude=-15):
        """Fly around exploring urban environment with comprehensive sensor capture"""
        print(f"\n[4/7] Starting comprehensive urban exploration ({duration}s)...")
        print(f"   Altitude: {abs(altitude)}m (safe for urban environment)")
        print(f"   Capture rate: {self.capture_rate}Hz")
        print(f"   Press Ctrl+C at any time to exit and land safely\n")
        
        speed = 3.0  # m/s
        start_time = time.time()
        waypoint_interval = 8  # Change waypoint every 8 seconds
        waypoint_count = 0
        last_capture_time = start_time
        
        while self.running and (time.time() - start_time < duration):
            try:
                # Generate waypoint - spiral exploration pattern
                state = self.client.getMultirotorState()
                current_pos = state.kinematics_estimated.position
                
                angle = waypoint_count * 0.5
                radius = min(50, waypoint_count * 2)
                
                offset_x = radius * np.cos(angle)
                offset_y = radius * np.sin(angle)
                
                target_x = current_pos.x_val + offset_x
                target_y = current_pos.y_val + offset_y
                
                waypoint_count += 1
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                print(f"   Waypoint {waypoint_count}: ({target_x:.1f}, {target_y:.1f}, {abs(altitude):.1f}m) "
                      f"[{elapsed:.0f}s/{duration}s, {remaining:.0f}s remaining]")
                
                # Move to waypoint
                move_future = self.client.moveToPositionAsync(target_x, target_y, altitude, speed)
                
                # Capture frames during movement at specified rate
                capture_start = time.time()
                while (time.time() - capture_start < waypoint_interval and 
                       self.running and 
                       time.time() - start_time < duration):
                    
                    # Capture at specified rate
                    current_time = time.time()
                    if (current_time - last_capture_time) >= self.capture_interval:
                        if self.recording:
                            timestamp, rgb, depth, seg, imu, gps, mag, baro, drone_state = self.capture_all_sensors()
                            
                            if rgb is not None:
                                # Log all sensor data
                                command = {
                                    "type": "move_to_position",
                                    "target": {"x": target_x, "y": target_y, "z": altitude},
                                    "speed": speed
                                }
                                
                                self.logger.log_frame(
                                    timestamp, rgb, depth, seg, imu, gps, mag, baro, drone_state, command
                                )
                                
                                # Save frame for video
                                overlay = rgb.copy()
                                pos = drone_state["position"]
                                vel = drone_state["velocity"]
                                cv2.putText(overlay, f"X: {pos['x']:.2f}m", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(overlay, f"Y: {pos['y']:.2f}m", (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(overlay, f"Alt: {abs(pos['z']):.2f}m", (10, 90), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.putText(overlay, f"Speed: {np.sqrt(vel['x']**2 + vel['y']**2):.2f}m/s", (10, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                self.frames.append(overlay)
                            
                            last_capture_time = current_time
                    
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
                # Wait for movement to complete
                move_future.join()
                
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] User requested exit...")
                self.running = False
                break
            except Exception as e:
                print(f"[WARNING] Waypoint error: {e}")
                continue
        
        if self.running:
            print("[OK] Exploration complete!")
        else:
            print("[OK] Exploration stopped by user")
    
    def return_to_start(self):
        """Return to starting position"""
        try:
            state = self.client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            
            print(f"\n   Returning to start (0, 0, {abs(current_pos.z_val):.1f}m)...")
            self.client.moveToPositionAsync(0, 0, current_pos.z_val, 3.0).join()
            print("[OK] Returned to start")
        except Exception as e:
            print(f"[WARNING] Return to start failed: {e}")
    
    def land(self):
        """Land the drone safely"""
        print("\n[5/7] Landing...")
        try:
            self.client.landAsync(timeout_sec=30).join()
            time.sleep(1)
            print("[OK] Landed safely!")
            return True
        except Exception as e:
            print(f"[ERROR] Landing failed: {e}")
            return False
    
    def emergency_land(self):
        """Emergency landing"""
        print("\n[!!! EMERGENCY LANDING !!!]")
        try:
            self.client.landAsync(timeout_sec=10).join()
            print("[OK] Emergency landing complete")
        except:
            pass
    
    def cleanup(self):
        """Cleanup: Disarm and release control"""
        print("\n[6/7] Cleaning up...")
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("[OK] Disarmed and released control")
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")
    
    def save_recording(self):
        """Save captured frames as video and finalize data logging"""
        print("\n[7/7] Saving recording and finalizing data...")
        
        # Save video
        if self.frames:
            height, width = self.frames[0].shape[:2]
            video_path = self.logger.base_dir / "flight_recording.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, self.capture_rate, (width, height))
            
            print(f"   Saving {len(self.frames)} frames to {video_path}...")
            for frame in self.frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"[OK] Video saved: {video_path}")
        
        # Save comprehensive data summary
        if self.logger:
            summary = self.logger.save_summary()
            print(f"\n[OK] All data saved to: {self.logger.base_dir}")
        
        print("[OK] Recording complete!")
    
    def run_comprehensive_flight(self, duration=120, capture_rate_hz=30):
        """Run complete comprehensive flight pattern"""
        self.capture_rate = capture_rate_hz
        self.capture_interval = 1.0 / capture_rate_hz
        
        if not self.connect():
            return False
        
        if not self.setup():
            return False
        
        if not self.takeoff():
            return False
        
        try:
            self.fly_urban_exploration(duration=duration, altitude=-15)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Emergency landing...")
            self.emergency_land()
        finally:
            if self.running:
                self.return_to_start()
            self.land()
            self.cleanup()
        
        if self.recording:
            self.save_recording()
        
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE FLIGHT COMPLETE!")
        print("=" * 70)
        return True


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE AUTONOMOUS DRONE FLIGHT - PHASE 0 COMPLETE")
    print("=" * 70)
    print("\nENVIRONMENT: AirSimNH (Neighborhood - Urban)")
    print("\nFEATURES:")
    print("  [OK] Multi-sensor capture (RGB, Depth, Segmentation)")
    print("  [OK] IMU, GPS, Magnetometer, Barometer")
    print("  [OK] Comprehensive data logging")
    print("  [OK] Organized directory structure")
    print("  [OK] Synchronized timestamps")
    print("\nIMPORTANT: Make sure AirSimNH.exe is running!")
    print("Wait 2-5 minutes after launching for it to fully load.")
    print("\nEXIT: Press Ctrl+C at any time to exit and land safely")
    print("\nStarting in 3 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return
    
    flight = ComprehensiveAutonomousFlight()
    
    # Get parameters
    try:
        duration_input = input("\nEnter flight duration in seconds (default=120, max=600): ").strip()
        if duration_input:
            duration = min(600, max(30, int(duration_input)))
        else:
            duration = 120
        
        rate_input = input(f"Enter capture rate in Hz (default=30, max=60): ").strip()
        if rate_input:
            capture_rate = min(60, max(1, int(rate_input)))
        else:
            capture_rate = 30
    except:
        duration = 120
        capture_rate = 30
    
    print(f"\nStarting {duration}s flight with {capture_rate}Hz capture rate...")
    print("Press Ctrl+C at any time to exit and land safely\n")
    
    flight.run_comprehensive_flight(duration=duration, capture_rate_hz=capture_rate)
    
    print("\nPress Enter to exit...")
    try:
        input()
    except:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Program interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

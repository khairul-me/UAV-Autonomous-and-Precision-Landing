"""
AUTONOMOUS DRONE FLIGHT IN REALISTIC ENVIRONMENT
This script flies the drone autonomously in AirSimNH (Neighborhood) environment
with realistic urban exploration patterns and recording.
"""

import airsim
import time
import cv2
import numpy as np
import os
import sys
from datetime import datetime

class RealisticAutonomousFlight:
    def __init__(self):
        self.client = None
        self.recording = True
        self.frames = []
        self.flight_data = []
        self.running = True
        
    def connect(self):
        """Connect to AirSim"""
        print("\n" + "=" * 70)
        print("  AUTONOMOUS FLIGHT - REALISTIC ENVIRONMENT")
        print("=" * 70)
        print("\n[1/6] Connecting to AirSim...")
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
        """Setup: Enable API control and arm"""
        print("\n[2/6] Setting up drone...")
        try:
            # Enable API control
            self.client.enableApiControl(True)
            print("[OK] API Control enabled")
            
            # Arm the drone
            self.client.armDisarm(True)
            print("[OK] Drone armed")
            
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
        print("\n[3/6] Taking off...")
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
    
    def capture_frame(self):
        """Capture current camera frame"""
        try:
            # Get RGB image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and responses[0].image_data_uint8:
                # Convert to numpy array
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Get state
                state = self.client.getMultirotorState()
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                
                # Add overlay with flight info
                overlay = img_rgb.copy()
                cv2.putText(overlay, f"X: {pos.x_val:.2f}m", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Y: {pos.y_val:.2f}m", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Alt: {abs(pos.z_val):.2f}m", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Speed: {np.sqrt(vel.x_val**2 + vel.y_val**2):.2f}m/s", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                return overlay, pos
            return None, None
        except Exception as e:
            print(f"[WARNING] Frame capture failed: {e}")
            return None, None
    
    def fly_urban_exploration(self, duration=120, altitude=-15):
        """Fly around exploring urban environment realistically"""
        print(f"\n[4/6] Starting urban exploration pattern ({duration}s)...")
        print(f"   Altitude: {abs(altitude)}m (safe for urban environment)")
        print(f"   Press Ctrl+C at any time to exit and land safely\n")
        
        speed = 3.0  # m/s - reasonable speed for exploration
        start_time = time.time()
        waypoint_interval = 8  # Change waypoint every 8 seconds
        
        waypoint_count = 0
        
        while self.running and (time.time() - start_time < duration):
            try:
                # Generate waypoint - explore in a spiral pattern
                state = self.client.getMultirotorState()
                current_pos = state.kinematics_estimated.position
                
                # Spiral exploration pattern
                angle = waypoint_count * 0.5  # Increment angle
                radius = min(50, waypoint_count * 2)  # Gradually increase radius
                
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
                self.client.moveToPositionAsync(target_x, target_y, altitude, speed).join()
                
                # Capture frames during flight
                capture_start = time.time()
                while (time.time() - capture_start < waypoint_interval and 
                       self.running and 
                       time.time() - start_time < duration):
                    if self.recording:
                        frame, pos = self.capture_frame()
                        if frame is not None:
                            self.frames.append(frame)
                            self.flight_data.append({
                                'time': time.time(),
                                'position': (pos.x_val, pos.y_val, pos.z_val),
                                'waypoint': waypoint_count
                            })
                    time.sleep(0.3)  # Capture every 0.3 seconds
                
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
        print("\n[5/6] Landing...")
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
        print("\n[6/6] Cleaning up...")
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("[OK] Disarmed and released control")
        except Exception as e:
            print(f"[WARNING] Cleanup error: {e}")
    
    def save_recording(self):
        """Save captured frames as video"""
        if not self.frames:
            print("[WARNING] No frames captured!")
            return
        
        print("\n[Saving recording...]")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"flight_recordings/realistic_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame dimensions
        if self.frames:
            height, width = self.frames[0].shape[:2]
            
            # Save as video using OpenCV
            video_path = f"{output_dir}/flight_recording.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 3.33, (width, height))  # ~3.33 fps (0.3s intervals)
            
            print(f"   Saving {len(self.frames)} frames to {video_path}...")
            
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"[OK] Video saved: {video_path}")
            
            # Save flight data
            import json
            data_path = f"{output_dir}/flight_data.json"
            with open(data_path, 'w') as f:
                json.dump(self.flight_data, f, indent=2)
            print(f"[OK] Flight data saved: {data_path}")
            
            # Save summary
            summary_path = f"{output_dir}/summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Realistic Environment Flight Summary\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Frames: {len(self.frames)}\n")
                f.write(f"Flight Duration: {self.flight_data[-1]['time'] - self.flight_data[0]['time']:.2f}s\n")
                f.write(f"Waypoints Visited: {len(set(d['waypoint'] for d in self.flight_data))}\n")
            print(f"[OK] Summary saved: {summary_path}")
    
    def run_realistic_flight(self, duration=120):
        """Run complete realistic flight pattern"""
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
        print("  FLIGHT COMPLETE!")
        print("=" * 70)
        return True


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  AUTONOMOUS DRONE FLIGHT - REALISTIC ENVIRONMENT")
    print("=" * 70)
    print("\nENVIRONMENT: AirSimNH (Neighborhood - Urban)")
    print("\nIMPORTANT: Make sure AirSimNH.exe is running!")
    print("Wait 2-5 minutes after launching for it to fully load.")
    print("\nThis will:")
    print("  1. Connect to AirSim")
    print("  2. Take off to safe altitude")
    print("  3. Explore urban environment autonomously")
    print("  4. Record all frames with flight data")
    print("  5. Return to start and land safely")
    print("\nEXIT: Press Ctrl+C at any time to exit and land safely")
    print("\nStarting in 3 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return
    
    flight = RealisticAutonomousFlight()
    
    # Get duration
    try:
        duration_input = input("\nEnter flight duration in seconds (default=120, max=600): ").strip()
        if duration_input:
            duration = min(600, max(30, int(duration_input)))
        else:
            duration = 120
    except:
        duration = 120
    
    print(f"\nStarting {duration}s flight...")
    print("Press Ctrl+C at any time to exit and land safely\n")
    
    flight.run_realistic_flight(duration=duration)
    
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

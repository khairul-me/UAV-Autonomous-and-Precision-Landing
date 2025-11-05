"""
AUTONOMOUS DRONE FLIGHT WITH RECORDING
This script flies the drone autonomously in a predefined pattern
and records the screen/frames.
"""

import airsim
import time
import cv2
import numpy as np
import os
from datetime import datetime

class AutonomousFlight:
    def __init__(self):
        self.client = None
        self.recording = True
        self.frames = []
        self.flight_data = []
        
    def connect(self):
        """Connect to AirSim"""
        print("\n" + "=" * 70)
        print("  AUTONOMOUS DRONE FLIGHT")
        print("=" * 70)
        print("\n[1/6] Connecting to AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("[OK] Connected to AirSim!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            print("Make sure Blocks.exe is running and wait 2-5 minutes after launch.")
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
        """Take off to 5m altitude"""
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
                
                # Add overlay
                overlay = img_rgb.copy()
                cv2.putText(overlay, f"X: {pos.x_val:.2f}m", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Y: {pos.y_val:.2f}m", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Z: {abs(pos.z_val):.2f}m", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                return overlay, pos
            return None, None
        except Exception as e:
            print(f"[WARNING] Frame capture failed: {e}")
            return None, None
    
    def fly_square_pattern(self, side_length=10, altitude=-5):
        """Fly in a square pattern"""
        print("\n[4/6] Starting square flight pattern...")
        print(f"   Side length: {side_length}m")
        print(f"   Altitude: {abs(altitude)}m")
        
        speed = 3.0  # m/s
        
        # Get current position
        state = self.client.getMultirotorState()
        start_pos = state.kinematics_estimated.position
        
        # Waypoints for square
        waypoints = [
            (start_pos.x_val + side_length, start_pos.y_val, altitude),
            (start_pos.x_val + side_length, start_pos.y_val + side_length, altitude),
            (start_pos.x_val, start_pos.y_val + side_length, altitude),
            (start_pos.x_val, start_pos.y_val, altitude),
        ]
        
        for i, (x, y, z) in enumerate(waypoints):
            print(f"\n   Flying to waypoint {i+1}/4: ({x:.1f}, {y:.1f}, {abs(z):.1f}m)")
            
            # Move to waypoint
            self.client.moveToPositionAsync(x, y, z, speed).join()
            
            # Hover for a moment
            time.sleep(1)
            
            # Capture frame
            if self.recording:
                frame, pos = self.capture_frame()
                if frame is not None:
                    self.frames.append(frame)
                    self.flight_data.append({
                        'time': time.time(),
                        'position': (pos.x_val, pos.y_val, pos.z_val)
                    })
        
        # Return to start
        print("\n   Returning to start position...")
        self.client.moveToPositionAsync(start_pos.x_val, start_pos.y_val, altitude, speed).join()
        print("[OK] Square pattern complete!")
    
    def fly_exploration_pattern(self, duration=60):
        """Fly around exploring the environment"""
        print(f"\n[4/6] Starting exploration pattern ({duration}s)...")
        
        speed = 2.0  # m/s
        start_time = time.time()
        waypoint_time = 0
        waypoint_interval = 5  # Change waypoint every 5 seconds
        
        while time.time() - start_time < duration:
            # Generate random waypoint within bounds
            state = self.client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            
            # Random offset (-10 to +10 meters)
            offset_x = np.random.uniform(-10, 10)
            offset_y = np.random.uniform(-10, 10)
            altitude = -5  # Maintain 5m altitude
            
            target_x = current_pos.x_val + offset_x
            target_y = current_pos.y_val + offset_y
            
            print(f"   Exploring: ({target_x:.1f}, {target_y:.1f}, {abs(altitude):.1f}m)")
            
            # Move to waypoint
            self.client.moveToPositionAsync(target_x, target_y, altitude, speed).join()
            
            # Capture frames during flight
            elapsed = time.time() - start_time
            while time.time() - start_time < elapsed + waypoint_interval and time.time() - start_time < duration:
                if self.recording:
                    frame, pos = self.capture_frame()
                    if frame is not None:
                        self.frames.append(frame)
                        self.flight_data.append({
                            'time': time.time(),
                            'position': (pos.x_val, pos.y_val, pos.z_val)
                        })
                time.sleep(0.5)  # Capture every 0.5 seconds
        
        print("[OK] Exploration complete!")
    
    def land(self):
        """Land the drone"""
        print("\n[5/6] Landing...")
        try:
            self.client.landAsync(timeout_sec=30).join()
            time.sleep(1)
            print("[OK] Landed safely!")
            return True
        except Exception as e:
            print(f"[ERROR] Landing failed: {e}")
            return False
    
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
        output_dir = f"flight_recordings/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame dimensions
        if self.frames:
            height, width = self.frames[0].shape[:2]
            
            # Save as video using OpenCV
            video_path = f"{output_dir}/flight_recording.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 2.0, (width, height))
            
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
            
            # Save individual frames
            frames_dir = f"{output_dir}/frames"
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(self.frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{frames_dir}/frame_{i:04d}.jpg", frame_bgr)
            print(f"[OK] Individual frames saved: {frames_dir}/")
    
    def run_square_flight(self):
        """Run complete square flight pattern"""
        if not self.connect():
            return False
        
        if not self.setup():
            return False
        
        if not self.takeoff():
            return False
        
        try:
            self.fly_square_pattern(side_length=10, altitude=-5)
        finally:
            self.land()
            self.cleanup()
        
        if self.recording:
            self.save_recording()
        
        print("\n" + "=" * 70)
        print("  FLIGHT COMPLETE!")
        print("=" * 70)
        return True
    
    def run_exploration_flight(self, duration=60):
        """Run exploration flight pattern"""
        if not self.connect():
            return False
        
        if not self.setup():
            return False
        
        if not self.takeoff():
            return False
        
        try:
            self.fly_exploration_pattern(duration=duration)
        finally:
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
    print("  AUTONOMOUS DRONE FLIGHT WITH RECORDING")
    print("=" * 70)
    print("\nIMPORTANT: Make sure Blocks.exe is running!")
    print("Wait 2-5 minutes after launching for it to fully load.")
    print("\nThis will:")
    print("  1. Connect to AirSim")
    print("  2. Take off")
    print("  3. Fly autonomously in a pattern")
    print("  4. Record all frames")
    print("  5. Save video and data")
    print("\nStarting in 3 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    flight = AutonomousFlight()
    
    # Choose flight pattern
    print("\nChoose flight pattern:")
    print("  1. Square pattern (quick, ~30 seconds)")
    print("  2. Exploration pattern (60 seconds)")
    
    try:
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    except:
        choice = "1"
    
    if choice == "2":
        try:
            duration = int(input("Enter duration in seconds (default=60): ").strip() or "60")
        except:
            duration = 60
        flight.run_exploration_flight(duration=duration)
    else:
        flight.run_square_flight()


if __name__ == "__main__":
    main()

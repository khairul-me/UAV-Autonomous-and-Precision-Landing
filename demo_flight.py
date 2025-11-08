"""
Interactive Demo Flight

Runs for 1-2 minutes with live visualization and recording

"""

import airsim
import numpy as np
import cv2
import time
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

class DemoFlight:
    """Interactive demo flight with visualization"""
    
    def __init__(self, duration=120, record_video=True):
        self.duration = duration
        self.record_video = record_video
        
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("[OK] Connected to AirSim")
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("[OK] Drone armed and ready")
        
        self.output_dir = f"demo_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OK] Output directory: {self.output_dir}")
        
        self.video_writer = None
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.output_dir, 'demo_flight.mp4')
            self.video_writer = cv2.VideoWriter(
                video_path,
                fourcc, 20.0, (1280, 720)
            )
            print(f"[OK] Video recording initialized: {video_path}")
        
        self.telemetry = {
            'time': [],
            'position': [],
            'velocity': [],
            'depth_min': [],
            'yaw': []
        }
        
        self.position_history = deque(maxlen=100)
    
    def take_off(self):
        """Take off to initial altitude"""
        print("\n[TAKEOFF]")
        print("  Taking off to 5m altitude...")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-5, velocity=1).join()
        print("  [OK] Airborne at 5m")
        time.sleep(1)
    
    def fly_pattern(self):
        """Fly a demonstration pattern"""
        print("\n[FLIGHT PATTERN]")
        print(f"  Flying for {self.duration} seconds...")
        print("  Pattern: Navigation with obstacle avoidance")
        
        start_time = time.time()
        step = 0
        goal = airsim.Vector3r(50, 50, -5)
        
        while (time.time() - start_time) < self.duration:
            elapsed = time.time() - start_time
            
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            
            depth_img = self._get_depth_image()
            min_depth = np.min(depth_img) * 100 if depth_img.size > 0 else 100.0
            
            dx = goal.x_val - pos.x_val
            dy = goal.y_val - pos.y_val
            distance_to_goal = np.sqrt(dx**2 + dy**2)
            
            if min_depth < 5.0:
                vx = 0.5
                vy = 1.0 * np.sign(dy) if dy != 0 else 1.0
                vz = 0.0
            elif distance_to_goal > 2.0:
                vx = 2.0 * dx / (distance_to_goal + 0.1)
                vy = 2.0 * dy / (distance_to_goal + 0.1)
                vz = 0.0
            else:
                angle = elapsed * 0.5
                vx = 1.5 * np.cos(angle)
                vy = 1.5 * np.sin(angle)
                vz = 0.0
            
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration=0.2,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
            ).join()
            
            orientation = state.kinematics_estimated.orientation
            yaw = airsim.to_eularian_angles(orientation)[2]
            
            self.telemetry['time'].append(elapsed)
            self.telemetry['position'].append((pos.x_val, pos.y_val, pos.z_val))
            self.telemetry['velocity'].append((vel.x_val, vel.y_val, vel.z_val))
            self.telemetry['depth_min'].append(min_depth)
            self.telemetry['yaw'].append(yaw)
            
            self.position_history.append((pos.x_val, pos.y_val))
            
            frame = self._create_visualization_frame(
                pos, vel, min_depth, distance_to_goal, elapsed
            )
            
            if self.video_writer is not None:
                self.video_writer.write(frame)
            
            try:
                cv2.imshow('Demo Flight', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Demo interrupted by user (Q key)")
                    break
            except:
                pass  # Window might not be available
            
            if step % 100 == 0:
                print(f"  [{elapsed:.1f}s] Pos: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f}) | "
                      f"Speed: {np.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2):.2f} m/s | "
                      f"Nearest obstacle: {min_depth:.2f}m")
            
            step += 1
            time.sleep(0.05)
        
        print(f"  [OK] Flight complete! Flew for {elapsed:.1f} seconds")
    
    def land(self):
        """Land the drone"""
        print("\n[LANDING]")
        print("  Landing...")
        self.client.landAsync().join()
        print("  [OK] Landed safely")
        time.sleep(1)
    
    def _get_depth_image(self):
        """Get depth image from camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, 
                                   pixels_as_float=True, compress=False)
            ])
            
            response = responses[0]
            depth_img = airsim.list_to_2d_float_array(
                response.image_data_float, response.width, response.height
            )
            
            return depth_img
        except:
            return np.ones((100, 100)) * 10.0  # Fallback
    
    def _get_rgb_image(self):
        """Get RGB image from camera"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            
            return img_rgb
        except:
            return np.zeros((240, 320, 3), dtype=np.uint8)  # Fallback
    
    def _create_visualization_frame(self, pos, vel, min_depth, dist_to_goal, elapsed):
        """Create visualization frame with telemetry overlay"""
        rgb_img = self._get_rgb_image()
        frame = cv2.resize(rgb_img, (1280, 720))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 300), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 40
        line_height = 35
        
        texts = [
            f"Time: {elapsed:.1f}s / {self.duration}s",
            f"Position: ({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f}) m",
            f"Velocity: ({vel.x_val:.2f}, {vel.y_val:.2f}, {vel.z_val:.2f}) m/s",
            f"Speed: {np.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2):.2f} m/s",
            f"Nearest Obstacle: {min_depth:.2f} m",
            f"Distance to Goal: {dist_to_goal:.2f} m",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_offset + i * line_height),
                       font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        status_color = (0, 255, 0) if min_depth > 5.0 else (0, 165, 255)
        status_text = "CLEAR" if min_depth > 5.0 else "OBSTACLE NEAR"
        cv2.putText(frame, status_text, (20, 280),
                   font, 0.8, status_color, 2, cv2.LINE_AA)
        
        if len(self.position_history) > 1:
            try:
                traj_img = self._create_trajectory_plot()
                traj_img_small = cv2.resize(traj_img, (300, 300))
                frame[10:310, 970:1270] = traj_img_small
            except:
                pass
        
        return frame
    
    def _create_trajectory_plot(self):
        """Create mini trajectory plot"""
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        
        if len(self.position_history) > 0:
            positions = np.array(list(self.position_history))
            ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
            ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10)
            ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8)
        
        ax.set_xlim(-10, 60)
        ax.set_ylim(-10, 60)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_title('Trajectory', fontsize=10)
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    def save_telemetry(self):
        """Save telemetry data and plots"""
        print("\n[SAVING DATA]")
        
        np.save(
            os.path.join(self.output_dir, 'telemetry.npy'),
            self.telemetry
        )
        print(f"  [OK] Saved telemetry data")
        
        self._plot_telemetry()
        print(f"  [OK] Saved telemetry plots")
    
    def _plot_telemetry(self):
        """Create telemetry plots"""
        fig = plt.figure(figsize=(15, 10))
        
        time_data = np.array(self.telemetry['time'])
        positions = np.array(self.telemetry['position'])
        
        # 3D trajectory
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Flight Trajectory')
        ax.legend()
        
        # Speed over time
        velocities = np.array(self.telemetry['velocity'])
        speeds = np.linalg.norm(velocities, axis=1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(time_data, speeds, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Obstacle distance
        depth_min = np.array(self.telemetry['depth_min'])
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(time_data, depth_min, 'r-', linewidth=2)
        ax3.axhline(y=5.0, color='orange', linestyle='--', label='Warning threshold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Nearest Obstacle (m)')
        ax3.set_title('Obstacle Proximity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Top view
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax4.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
        ax4.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('Top View')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'telemetry_plots.png'), dpi=300)
        plt.close()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[CLEANUP]")
        
        if self.video_writer is not None:
            self.video_writer.release()
            print("  [OK] Video saved")
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("  [OK] Drone disarmed")
        except:
            pass
        
        print(f"\n[OK] All outputs saved to: {self.output_dir}")

def main():
    print("="*80)
    print("DEMO FLIGHT")
    print("="*80)
    print("Duration: 2 minutes")
    print("Pattern: Navigate towards goal with obstacle avoidance")
    print("Recording: Video + Telemetry")
    print("Press 'Q' during flight to stop early")
    print("="*80 + "\n")
    
    try:
        input("Press ENTER to start demo flight (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n[INFO] Cancelled by user")
        return
    
    demo = DemoFlight(duration=120, record_video=True)
    
    try:
        demo.take_off()
        demo.fly_pattern()
        demo.land()
        demo.save_telemetry()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print(f"Check outputs in: {demo.output_dir}/")
        print("  - demo_flight.mp4: Recorded video")
        print("  - telemetry.npy: Flight data")
        print("  - telemetry_plots.png: Data visualization")
        print("\n[OK] Ready to start training!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Demo interrupted by user")
        demo.land()
    except Exception as e:
        print(f"\n\n[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()

if __name__ == '__main__':
    main()


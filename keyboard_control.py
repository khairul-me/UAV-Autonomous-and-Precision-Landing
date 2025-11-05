"""
KEYBOARD CONTROL FOR DRONE
Press a key to get full control, then use keyboard to fly the drone

Controls:
  [C] - Claim Control (Enable API Control + Arm Drone)
  [T] - Take Off
  [L] - Land
  [H] - Hover
  
  Movement (while flying):
  [W] or [UP]    - Move Forward
  [S] or [DOWN]  - Move Backward
  [A] or [LEFT]  - Move Left
  [D] or [RIGHT] - Move Right
  [Q] or [E]     - Rotate Left/Right (Yaw)
  [R] or [F]     - Move Up/Down
  
  [X] - Emergency Stop (Land immediately)
  [ESC] or [Q] - Quit and Release Control
"""

import airsim
import time
import sys

# Windows-specific keyboard input
try:
    import msvcrt
    WINDOWS = True
except ImportError:
    try:
        import getch
        WINDOWS = False
    except ImportError:
        print("[ERROR] Please install 'py-getch' for non-Windows systems: pip install py-getch")
        sys.exit(1)

class KeyboardController:
    """Keyboard controller for AirSim drone"""
    
    def __init__(self):
        self.client = None
        self.control_claimed = False
        self.flying = False
        self.move_speed = 2.0  # m/s
        self.rotate_speed = 30  # degrees per key press
        self.move_distance = 2.0  # meters per key press
        self.altitude_change = 1.0  # meters per key press
        
    def print_instructions(self):
        """Print control instructions"""
        print("\n" + "=" * 70)
        print("  KEYBOARD DRONE CONTROL")
        print("=" * 70)
        print("\n[C] - Claim Control (Enable API + Arm)")
        print("[T] - Take Off")
        print("[L] - Land")
        print("[H] - Hover")
        print("\nMovement (while flying):")
        print("  [W] / [UP]    - Forward")
        print("  [S] / [DOWN]  - Backward")
        print("  [A] / [LEFT]  - Left")
        print("  [D] / [RIGHT] - Right")
        print("  [R] / [F]     - Up / Down")
        print("  [Q] / [E]     - Rotate Left / Right")
        print("\n[X] - Emergency Stop")
        print("[ESC] or [Q] - Quit")
        print("\n" + "=" * 70)
        print("Press [C] to claim control when ready...")
        print("=" * 70 + "\n")
    
    def get_key(self):
        """Get a single key press (non-blocking on Windows)"""
        if WINDOWS:
            if msvcrt.kbhit():
                try:
                    key = msvcrt.getch()
                    # Handle special keys
                    if key == b'\x00' or key == b'\xe0':  # Function key or arrow key prefix
                        key2 = msvcrt.getch()
                        # Map arrow keys
                        if key2 == b'H':  # Up arrow
                            return 'up'
                        elif key2 == b'P':  # Down arrow
                            return 'down'
                        elif key2 == b'K':  # Left arrow
                            return 'left'
                        elif key2 == b'M':  # Right arrow
                            return 'right'
                    elif key == b'\x1b':  # ESC
                        return 'esc'
                    else:
                        try:
                            decoded = key.decode('utf-8').lower()
                            # Filter out invalid characters and control keys
                            if decoded == '\r' or decoded == '\n':
                                return None  # Ignore Enter/Return key
                            if decoded.isprintable():
                                return decoded
                        except:
                            pass
                except Exception as e:
                    pass
            return None
        else:
            # Non-Windows: blocking for now
            import select
            import termios
            import tty
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return getch.getch().lower()
            return None
    
    def connect(self):
        """Connect to AirSim"""
        print("[1] Connecting to AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("[OK] Connected to AirSim!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            print("Make sure Blocks.exe or AirSimNH.exe is running!")
            return False
    
    def claim_control(self):
        """Enable API control and arm the drone"""
        if self.control_claimed:
            print("[INFO] Control already claimed!")
            return True
        
        print("\n" + "=" * 70)
        print("  CLAIMING CONTROL...")
        print("=" * 70)
        
        try:
            # Enable API control
            print("[1] Enabling API control...")
            self.client.enableApiControl(True)
            print("[OK] API Control enabled!")
            
            # Arm the drone
            print("[2] Arming drone...")
            result = self.client.armDisarm(True)
            print(f"[OK] Drone armed! (Result: {result})")
            
            time.sleep(1)  # Wait for motors to start
            
            # Verify state
            state = self.client.getMultirotorState()
            print(f"[OK] Current position: X={state.kinematics_estimated.position.x_val:.2f}, "
                  f"Y={state.kinematics_estimated.position.y_val:.2f}, "
                  f"Z={state.kinematics_estimated.position.z_val:.2f}")
            
            self.control_claimed = True
            print("\n[SUCCESS] You now have FULL CONTROL of the drone!")
            print("Press [T] to take off, or use movement keys while on ground")
            print("=" * 70 + "\n")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to claim control: {e}")
            return False
    
    def takeoff(self):
        """Take off the drone"""
        if not self.control_claimed:
            print("[ERROR] Claim control first with [C]!")
            return
        
        print("\n[TAKING OFF...]")
        try:
            self.client.takeoffAsync(timeout_sec=30).join()
            time.sleep(1)
            state = self.client.getMultirotorState()
            altitude = abs(state.kinematics_estimated.position.z_val)
            print(f"[OK] Flying at {altitude:.2f}m altitude!")
            self.flying = True
        except Exception as e:
            print(f"[ERROR] Takeoff failed: {e}")
    
    def land(self):
        """Land the drone"""
        if not self.control_claimed:
            print("[ERROR] Claim control first with [C]!")
            return
        
        print("\n[LANDING...]")
        try:
            self.client.landAsync(timeout_sec=30).join()
            time.sleep(1)
            print("[OK] Landed!")
            self.flying = False
        except Exception as e:
            print(f"[ERROR] Landing failed: {e}")
    
    def hover(self):
        """Hover in place"""
        if not self.control_claimed:
            print("[ERROR] Claim control first with [C]!")
            return
        
        try:
            self.client.hoverAsync().join()
            print("[HOVERING...]")
        except Exception as e:
            print(f"[ERROR] Hover failed: {e}")
    
    def move_forward(self):
        """Move forward"""
        self._move_relative(self.move_distance, 0, 0)
    
    def move_backward(self):
        """Move backward"""
        self._move_relative(-self.move_distance, 0, 0)
    
    def move_left(self):
        """Move left"""
        self._move_relative(0, -self.move_distance, 0)
    
    def move_right(self):
        """Move right"""
        self._move_relative(0, self.move_distance, 0)
    
    def move_up(self):
        """Move up"""
        self._move_relative(0, 0, -self.altitude_change)
    
    def move_down(self):
        """Move down"""
        self._move_relative(0, 0, self.altitude_change)
    
    def rotate_left(self):
        """Rotate left (yaw)"""
        self._rotate_relative(-self.rotate_speed)
    
    def rotate_right(self):
        """Rotate right (yaw)"""
        self._rotate_relative(self.rotate_speed)
    
    def _move_relative(self, dx, dy, dz):
        """Move relative to current position"""
        if not self.control_claimed:
            print("[ERROR] Claim control first with [C]!")
            return
        
        try:
            state = self.client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            
            target_x = current_pos.x_val + dx
            target_y = current_pos.y_val + dy
            target_z = current_pos.z_val + dz
            
            # Clamp Z to reasonable altitude if flying
            if self.flying and target_z > -0.5:  # Don't go too low
                target_z = -1.0
            
            self.client.moveToPositionAsync(
                target_x, target_y, target_z, 
                self.move_speed
            )
            
            direction = ""
            if dx > 0: direction = "Forward"
            elif dx < 0: direction = "Backward"
            elif dy > 0: direction = "Right"
            elif dy < 0: direction = "Left"
            elif dz < 0: direction = "Up"
            elif dz > 0: direction = "Down"
            
            print(f"[MOVING {direction}...]")
            
        except Exception as e:
            print(f"[ERROR] Move failed: {e}")
    
    def _rotate_relative(self, degrees):
        """Rotate relative to current orientation"""
        if not self.control_claimed:
            print("[ERROR] Claim control first with [C]!")
            return
        
        try:
            state = self.client.getMultirotorState()
            current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
            target_yaw = current_yaw + airsim.utils.to_radians(degrees)
            
            self.client.rotateToYawAsync(target_yaw, timeout_sec=5)
            direction = "Left" if degrees < 0 else "Right"
            print(f"[ROTATING {direction}...]")
            
        except Exception as e:
            print(f"[ERROR] Rotate failed: {e}")
    
    def emergency_stop(self):
        """Emergency stop - land immediately"""
        print("\n[!!! EMERGENCY STOP !!!]")
        self.land()
    
    def cleanup(self):
        """Cleanup - disarm and release control"""
        if self.client:
            try:
                print("\n[Cleaning up...]")
                if self.flying:
                    self.land()
                if self.control_claimed:
                    self.client.armDisarm(False)
                    self.client.enableApiControl(False)
                    print("[OK] Disarmed and released control")
            except Exception as e:
                print(f"[WARNING] Cleanup error: {e}")
    
    def run(self):
        """Main control loop"""
        # Print instructions
        self.print_instructions()
        
        # Connect
        if not self.connect():
            print("\n[ERROR] Cannot connect to AirSim!")
            print("Please make sure Blocks.exe is running and wait 2-5 minutes after launch.")
            print("\nPress any key to exit...")
            # Don't use input() - it causes EOFError when run non-interactively
            try:
                if WINDOWS:
                    import msvcrt
                    msvcrt.getch()
            except:
                time.sleep(2)
            return False
        
        print("\n" + "="*70)
        print("WAITING FOR INPUT...")
        print("Press [C] to claim control and start flying!")
        print("="*70 + "\n")
        
        # Main loop
        try:
            while True:
                key = self.get_key()
                
                if key:
                    if key == 'c':
                        self.claim_control()
                    
                    elif key == 't' and self.control_claimed:
                        self.takeoff()
                    
                    elif key == 'l' and self.control_claimed:
                        self.land()
                    
                    elif key == 'h' and self.control_claimed:
                        self.hover()
                    
                    elif (key == 'w' or key == 'up') and self.control_claimed:
                        self.move_forward()
                    
                    elif (key == 's' or key == 'down') and self.control_claimed:
                        self.move_backward()
                    
                    elif (key == 'a' or key == 'left') and self.control_claimed:
                        self.move_left()
                    
                    elif (key == 'd' or key == 'right') and self.control_claimed:
                        self.move_right()
                    
                    elif key == 'r' and self.control_claimed:
                        self.move_up()
                    
                    elif key == 'f' and self.control_claimed:
                        self.move_down()
                    
                    elif key == 'q':
                        # Rotate left or quit
                        if self.control_claimed:
                            self.rotate_left()
                        else:
                            break
                    
                    elif key == 'e' and self.control_claimed:
                        self.rotate_right()
                    
                    elif key == 'x' and self.control_claimed:
                        self.emergency_stop()
                    
                    elif key == 'esc':
                        break
                    
                    elif self.control_claimed:
                        # Any other key while not flying
                        pass
                
                # Small delay to prevent CPU spinning
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[Interrupted by user]")
        
        finally:
            self.cleanup()
        
        print("\n[Exiting...]")
        return True

def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  AIRSIM KEYBOARD DRONE CONTROLLER")
    print("=" * 70)
    print("\nIMPORTANT: Make sure Blocks.exe or AirSimNH.exe is running!")
    print("Wait 2-5 minutes after launching for it to fully load.")
    print("\nStarting in 2 seconds... (Press Ctrl+C to cancel)")
    
    try:
        time.sleep(2)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    controller = KeyboardController()
    controller.run()

if __name__ == "__main__":
    main()

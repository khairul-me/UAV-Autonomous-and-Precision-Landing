# 🚁 Quick Start: Flying Your Drone

## ✅ Status: Drone is Visible and Ready!

Your drone is now visible in the Blocks environment. Here's how to arm and fly it.

---

## 🎮 Option 1: Keyboard Control (Easiest)

### Launch Keyboard Control:
```powershell
cd E:\Drone
.\run_keyboard_control.bat
```

### Controls:
- **T** - Takeoff (arms and takes off)
- **L** - Land
- **W/↑** - Move forward
- **S/↓** - Move backward
- **A/←** - Move left
- **D/→** - Move right
- **Q** - Move up
- **Z** - Move down
- **Y** - Yaw left (rotate counter-clockwise)
- **C** - Yaw right (rotate clockwise)
- **G** - Get control (if lost)
- **ESC** - Exit

### Steps:
1. Run `run_keyboard_control.bat`
2. Wait for "Connected to AirSim!"
3. Press **T** to takeoff
4. Use WASD/Arrow keys to move around
5. Press **L** to land when done

---

## 🐍 Option 2: Python Script (Automatic)

### Run the flight sequence:
```powershell
cd E:\Drone
.\venv\Scripts\python.exe MAKE_IT_FLY.py
```

This will automatically:
1. Connect to AirSim
2. Arm the drone
3. Take off
4. Move to a waypoint
5. Hover and rotate
6. Return to start
7. Land and disarm

---

## 📝 Option 3: Custom Python Script

Create your own script:

```python
import airsim
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control
client.enableApiControl(True)

# Arm the drone
client.armDisarm(True)
print("Drone armed!")

# Takeoff
client.takeoffAsync(timeout_sec=10).join()
print("Taking off...")

# Wait a bit
time.sleep(2)

# Move forward 5 meters
client.moveToPositionAsync(5, 0, -5, 5).join()
print("Moving forward...")

# Land
client.landAsync().join()
print("Landing...")

# Disarm
client.armDisarm(False)
print("Drone disarmed!")
```

---

## 🔧 Troubleshooting

### "Connection refused" error:
- Make sure Blocks.exe is running and fully loaded (wait 2-5 minutes after launch)
- Check that API server is ready (you should see "Asset database ready!" in console)

### "API control not enabled":
- The script should enable it automatically with `client.enableApiControl(True)`
- If it fails, the drone might already be under API control

### Drone doesn't respond:
- Make sure you've pressed **T** (takeoff) in keyboard control, or called `takeoffAsync()` in Python
- Check that the drone is armed: `client.armDisarm(True)`

---

## 🎯 Quick Commands Reference

| Action | Keyboard | Python |
|--------|----------|--------|
| Arm/Disarm | Auto on takeoff | `client.armDisarm(True)` |
| Takeoff | **T** | `client.takeoffAsync().join()` |
| Land | **L** | `client.landAsync().join()` |
| Move | WASD/Arrows | `client.moveToPositionAsync(x, y, z, speed)` |
| Hover | Release keys | `client.hoverAsync().join()` |

---

## ✨ Next Steps

Now that you can fly:
1. Try the keyboard control for manual flight
2. Test `MAKE_IT_FLY.py` for automated flight
3. Use `phase0_task03_data_pipeline.py` to collect sensor data while flying
4. Start collecting your dataset for Phase 1!

---

**Enjoy flying! 🚁**

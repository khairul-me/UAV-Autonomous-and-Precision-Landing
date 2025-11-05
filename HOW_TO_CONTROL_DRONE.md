# How to Control Your Drone in AirSim

## Quick Answer: How to Arm and Control the Drone

### The 3 Essential Commands:

1. **`client.enableApiControl(True)`** - This gives Python control of the drone
2. **`client.armDisarm(True)`** - This starts the motors (arms the drone)
3. **`client.takeoffAsync().join()`** - This makes the drone take off

### Complete Sequence:

```python
import airsim

# 1. Connect
client = airsim.MultirotorClient()
client.confirmConnection()

# 2. Take control
client.enableApiControl(True)

# 3. Arm the drone (start motors)
client.armDisarm(True)

# 4. Take off
client.takeoffAsync().join()

# 5. Fly around
client.moveToPositionAsync(10, 0, -5, 5).join()  # Move forward 10m

# 6. Land
client.landAsync().join()

# 7. Disarm (stop motors)
client.armDisarm(False)

# 8. Release control
client.enableApiControl(False)
```

---

## Step-by-Step Explanation

### Step 1: Connect to AirSim
```python
client = airsim.MultirotorClient()
client.confirmConnection()
```
**What this does:** Creates a connection to the AirSim API server. Make sure Blocks.exe is running!

### Step 2: Enable API Control
```python
client.enableApiControl(True)
```
**What this does:** 
- **CRITICAL!** Without this, you can only READ data, not CONTROL the drone
- This tells AirSim "Python script wants to control the drone"
- The drone will ignore keyboard/gamepad input while this is enabled

### Step 3: Arm the Drone
```python
client.armDisarm(True)
```
**What this does:**
- Starts the motors spinning
- Prepares the drone for flight
- **Armed = Motors on, ready to fly**
- **Disarmed = Motors off, safe**

### Step 4: Take Off
```python
client.takeoffAsync().join()
```
**What this does:**
- Drone automatically lifts off and hovers at ~2-3 meters
- `.join()` waits for takeoff to complete
- Takes about 5-10 seconds

### Step 5: Fly (Move Around)
```python
# Move to position (x, y, z, speed)
client.moveToPositionAsync(10, 0, -5, 5).join()
```
**Coordinates:**
- **X**: Forward (+) / Backward (-)
- **Y**: Right (+) / Left (-)
- **Z**: Up (-) / Down (+) - **NOTE: Negative Z is UP in AirSim!**
- **Speed**: Meters per second

### Step 6: Land
```python
client.landAsync().join()
```
**What this does:** Drone descends and lands automatically

### Step 7: Disarm
```python
client.armDisarm(False)
```
**What this does:** Stops the motors (motors turn off)

### Step 8: Release Control
```python
client.enableApiControl(False)
```
**What this does:** Returns control to manual/keyboard

---

## Common Flight Commands

### Basic Movement:
```python
# Hover in place
client.hoverAsync().join()

# Move forward 10 meters at 5 m/s
client.moveToPositionAsync(10, 0, -5, 5).join()

# Move right 5 meters
client.moveToPositionAsync(0, 5, -5, 5).join()

# Rotate (yaw) 90 degrees
import math
client.rotateToYawAsync(math.radians(90)).join()

# Go to specific altitude (-10 means 10 meters up)
client.moveToPositionAsync(0, 0, -10, 5).join()
```

### Velocity Control (Continuous Movement):
```python
# Move forward continuously at 5 m/s
client.moveByVelocityAsync(5, 0, 0, 1).join()  # Forward 1 second

# Move right continuously
client.moveByVelocityAsync(0, 5, 0, 1).join()  # Right 1 second

# Hover
client.moveByVelocityAsync(0, 0, 0, 1).join()  # Stop
```

---

## Using the Verbose Control Script

I've created `control_drone_verbose.py` which shows you EXACTLY what's happening at each step.

### To run it:

1. **Make sure Blocks.exe is running** (wait 2-5 minutes for it to load)

2. **Open PowerShell:**
```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python control_drone_verbose.py
```

3. **Watch the output** - It will show you:
   - Each step as it happens
   - What each command does
   - Current drone state (position, velocity, armed status)
   - Success/failure of each operation

---

## Troubleshooting

### Problem: "Connection refused"
**Solution:** 
- Make sure Blocks.exe is running
- Wait 2-5 minutes after launching Blocks
- Check if port 41451 is listening: `netstat -an | findstr 41451`

### Problem: Drone won't arm
**Solution:**
- Make sure you called `enableApiControl(True)` FIRST
- Some drones auto-arm on takeoff, so try `takeoffAsync()` anyway

### Problem: Drone won't take off
**Solution:**
- Check if it's already armed: `state = client.getMultirotorState(); print(state.armed)`
- Try arming first: `client.armDisarm(True)`
- Wait 2 seconds after arming before taking off

### Problem: Drone moves but doesn't fly
**Solution:**
- Check Z coordinate - remember **negative Z is UP** in AirSim!
- Try: `client.moveToPositionAsync(0, 0, -5, 5).join()` (negative Z = 5 meters up)

---

## Quick Test Script

Create `test_control.py`:

```python
import airsim
import time

print("Connecting...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("[OK] Connected!")

print("Enabling API control...")
client.enableApiControl(True)
print("[OK] API control enabled!")

print("Arming...")
client.armDisarm(True)
time.sleep(2)
print("[OK] Armed!")

print("Taking off...")
client.takeoffAsync().join()
print("[OK] Flying!")

print("Landing...")
client.landAsync().join()
print("[OK] Landed!")

print("Disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("[OK] Done!")
```

Run it: `python test_control.py`

---

## Key Takeaways

1. **Always enable API control first** - `enableApiControl(True)`
2. **Arm before takeoff** - `armDisarm(True)`
3. **Remember: Negative Z is UP** - `moveToPositionAsync(x, y, -5, speed)` means 5 meters up
4. **Use `.join()` to wait** - `takeoffAsync().join()` waits for completion
5. **Always disarm and release control** - Clean up when done

---

**Now run `control_drone_verbose.py` to see it all in action with detailed output!**

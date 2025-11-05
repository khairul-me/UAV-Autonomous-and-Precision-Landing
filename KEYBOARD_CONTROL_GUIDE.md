# Keyboard Control Guide

## Quick Start

Run the keyboard control script:

```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python keyboard_control.py
```

## How It Works

1. **Press [C] to Claim Control** - This automatically:
   - Enables API control (gets full control)
   - Arms the drone (starts motors)
   - Verifies everything is ready

2. **Then use keyboard keys to fly!**

---

## Control Keys

### Setup
- **[C]** - Claim Control (Enable API + Arm Drone)
  - **Do this FIRST!** This gives you full control and starts the motors

### Flight Commands
- **[T]** - Take Off
- **[L]** - Land
- **[H]** - Hover (stop and hold position)

### Movement (While Flying or on Ground)
- **[W]** or **[↑]** - Move Forward
- **[S]** or **[↓]** - Move Backward
- **[A]** or **[←]** - Move Left
- **[D]** or **[→]** - Move Right
- **[R]** - Move Up (Increase altitude)
- **[F]** - Move Down (Decrease altitude)

### Rotation
- **[Q]** - Rotate Left (Yaw)
- **[E]** - Rotate Right (Yaw)

### Safety
- **[X]** - Emergency Stop (Land immediately)
- **[ESC]** or **[Q]** - Quit and Release Control

---

## Step-by-Step Usage

### 1. Start AirSim
- Launch `Blocks.exe` or `AirSimNH.exe`
- Wait 2-5 minutes for it to fully load

### 2. Run Keyboard Control Script
```powershell
python keyboard_control.py
```

### 3. Claim Control
- Press **[C]** to enable API control and arm the drone
- You'll see confirmation messages
- Motors will start spinning

### 4. Take Off
- Press **[T]** to take off
- Drone will lift to default altitude

### 5. Fly!
- Use **[W]**, **[A]**, **[S]**, **[D]** to move
- Use **[R]**, **[F]** to change altitude
- Use **[Q]**, **[E]** to rotate
- Press **[H]** to hover at any time

### 6. Land and Quit
- Press **[L]** to land
- Press **[ESC]** or **[Q]** to quit and release control

---

## Movement Settings

The script has default settings you can modify in the code:

```python
self.move_speed = 2.0        # m/s (how fast drone moves)
self.rotate_speed = 30       # degrees per key press
self.move_distance = 2.0     # meters per key press
self.altitude_change = 1.0   # meters per key press
```

To change these, edit `keyboard_control.py` and modify the values in the `__init__` method.

---

## Important Notes

### ⚠️ You MUST Claim Control First!
- Press **[C]** before any other commands
- This enables API control and arms the drone
- Without this, movement commands won't work

### Movement Commands
- Movement is **relative** to current position
- Each key press moves the drone a set distance
- The drone will smoothly move to the target position

### Safety Features
- **[X]** provides emergency stop (lands immediately)
- Automatic cleanup on exit (disarms and releases control)
- Safe altitude limits to prevent crashes

---

## Troubleshooting

### "Claim control first with [C]!" error
**Solution:** Press **[C]** first to enable API control and arm the drone.

### Drone doesn't respond to movement keys
**Possible causes:**
1. Control not claimed - Press **[C]** first
2. Still landing/taking off - Wait for command to complete
3. AirSim not fully loaded - Wait 2-5 minutes after launching

### Keys not registering
**Solution:** Make sure the terminal window has focus (click on it)

### Arrow keys don't work
**Solution:** The script also accepts **[W]**, **[A]**, **[S]**, **[D]** for movement.

---

## Tips

1. **Start slow**: Get familiar with controls before doing complex maneuvers
2. **Use hover**: Press **[H]** frequently to stabilize and check position
3. **Watch altitude**: Use **[R]** and **[F]** to maintain safe altitude
4. **Practice landing**: Always land with **[L]** before quitting
5. **Emergency stop**: **[X]** is your safety button - use it if needed!

---

## Complete Example Session

```
1. Start Blocks.exe
2. Wait 2-5 minutes for loading
3. Run: python keyboard_control.py
4. Press [C] - Claim control (motors start)
5. Press [T] - Take off
6. Press [W] - Move forward
7. Press [D] - Move right
8. Press [R] - Move up
9. Press [H] - Hover
10. Press [L] - Land
11. Press [ESC] - Quit
```

---

## Summary

**To control with keyboard:**
1. ✅ Run `keyboard_control.py`
2. ✅ Press **[C]** to claim control (enables API + arms)
3. ✅ Press **[T]** to take off
4. ✅ Use **[WASD]** or arrow keys to move
5. ✅ Press **[L]** to land, **[ESC]** to quit

**That's it! You're flying!** 🚁

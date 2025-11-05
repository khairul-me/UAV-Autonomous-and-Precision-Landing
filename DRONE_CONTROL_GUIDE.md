# 🚁 AirSim Drone Control Guide

## 🎉 Great News!

Your AirSim setup is **WORKING**! The API connection is successful and you can now control the drone programmatically.

---

## 📚 Basic Concepts

### AirSim MultirotorClient

The `MultirotorClient` is your interface to control the drone:

```python
import airsim

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()  # Verify connection

# Enable API control (required before flight)
client.enableApiControl(True)

# Arm the drone
client.armDisarm(True)
```

### Coordinate System

- **X-axis**: Forward (+) / Backward (-)
- **Y-axis**: Right (+) / Left (-)
- **Z-axis**: Down (+) / Up (-) 
  - **Note:** Z is negative UP, positive DOWN (NED coordinate system)
  - To go up 5 meters: `z = -5`
  - To go down 5 meters: `z = 5`

---

## 🚀 Basic Flight Operations

### 1. Takeoff

```python
client.takeoffAsync().join()
# OR with timeout
client.takeoffAsync(timeout_sec=10).join()
```

### 2. Move to Position

```python
# Move to position (x, y, z, velocity)
# x=5m forward, y=0, z=-5m up (5m altitude), velocity=5 m/s
client.moveToPositionAsync(5, 0, -5, 5).join()
```

### 3. Move by Velocity

```python
# Move forward at 5 m/s for 3 seconds
client.moveByVelocityAsync(5, 0, 0, 3).join()
```

### 4. Hover

```python
# Hover in place
client.hoverAsync().join()
```

### 5. Land

```python
client.landAsync().join()
```

---

## 📋 Complete Flight Sequence Example

```python
import airsim
import time

# Connect
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control and arm
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)  # Wait for stabilization

# Move forward
print("Moving forward 10 meters...")
client.moveToPositionAsync(10, 0, -5, 5).join()

# Hover
print("Hovering...")
client.hoverAsync().join()
time.sleep(2)

# Move right
print("Moving right 5 meters...")
client.moveToPositionAsync(10, 5, -5, 5).join()

# Land
print("Landing...")
client.landAsync().join()

# Disarm
client.armDisarm(False)
client.enableApiControl(False)
```

---

## 🎮 Advanced Control Functions

### Rotate (Yaw)

```python
# Rotate 45 degrees (in radians)
import math
client.rotateToYawAsync(math.radians(45)).join()

# Rotate by relative angle
client.rotateByYawRateAsync(math.radians(30), 2).join()  # 30 deg/s for 2 seconds
```

### Move Along Path

```python
# Follow a waypoint path
waypoints = [
    airsim.Vector3r(10, 0, -5),
    airsim.Vector3r(10, 10, -5),
    airsim.Vector3r(0, 10, -5),
    airsim.Vector3r(0, 0, -5)
]
client.moveOnPathAsync(waypoints, 5, timeout_sec=60).join()
```

### Camera Control

```python
# Get camera images
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene),
    airsim.ImageRequest("0", airsim.ImageType.DepthPlanar),
    airsim.ImageRequest("0", airsim.ImageType.Segmentation)
])

# Save images
for idx, response in enumerate(responses):
    airsim.write_file(f"image_{idx}.png", response.image_data_uint8)
```

### Get Vehicle State

```python
# Get current state
state = client.getMultirotorState()
print(f"Position: {state.kinematics_estimated.position}")
print(f"Velocity: {state.kinematics_estimated.linear_velocity}")
print(f"Orientation: {state.kinematics_estimated.orientation}")
```

---

## ⚙️ Important Settings

### Timeout and Safety

```python
# Set timeouts for operations
client.takeoffAsync(timeout_sec=10).join()

# Set safety checks
client.setSafety(
    enable_reasons=airsim.SafetyEval.SafetyViolationType.All,
    obs_clearance=1.0,  # 1 meter clearance from obstacles
    obs_avoidance_vel=0.5,  # 0.5 m/s when avoiding
    obs_avoidance_dist=5.0  # Start avoiding at 5 meters
)
```

---

## 🎯 Common Flight Patterns

### 1. Square Pattern

```python
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
time.sleep(2)

altitude = -5  # 5 meters up
speed = 3  # 3 m/s

# Square: 10m x 10m
client.moveToPositionAsync(10, 0, altitude, speed).join()
client.moveToPositionAsync(10, 10, altitude, speed).join()
client.moveToPositionAsync(0, 10, altitude, speed).join()
client.moveToPositionAsync(0, 0, altitude, speed).join()

client.landAsync().join()
```

### 2. Spiral Pattern

```python
import airsim
import math
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
time.sleep(2)

radius = 5
altitude = -5
num_turns = 2
steps = 20

for i in range(steps):
    angle = (2 * math.pi * num_turns * i) / steps
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    client.moveToPositionAsync(x, y, altitude, 2).join()
    time.sleep(0.1)

client.landAsync().join()
```

### 3. Autonomous Mission with Images

```python
import airsim
import time
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
time.sleep(2)

# Mission waypoints
waypoints = [
    (5, 0, -5),
    (5, 5, -5),
    (0, 5, -5),
    (0, 0, -5)
]

for idx, (x, y, z) in enumerate(waypoints):
    print(f"Flying to waypoint {idx+1}: ({x}, {y}, {z})")
    client.moveToPositionAsync(x, y, z, 3).join()
    
    # Hover and capture image
    time.sleep(1)
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene)
    ])
    airsim.write_file(f"waypoint_{idx}.png", responses[0].image_data_uint8)
    print(f"  Captured image: waypoint_{idx}.png")

client.landAsync().join()
print("Mission complete!")
```

---

## 🛡️ Safety Best Practices

1. **Always check connection** before operations
2. **Enable API control** before takeoff
3. **Arm the drone** before flight
4. **Set reasonable timeouts** for all operations
5. **Monitor vehicle state** during flight
6. **Land safely** before disarming
7. **Disable API control** when done

### Emergency Stop

```python
# Emergency: Stop all movement and land immediately
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
```

---

## 📊 Monitoring During Flight

```python
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

# Monitor during flight
for i in range(10):
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    
    print(f"Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
    print(f"Velocity: {vel.x_val:.2f} m/s")
    print("---")
    
    time.sleep(1)
    client.moveByVelocityAsync(1, 0, 0, 1).join()

client.landAsync().join()
```

---

## 🔧 Troubleshooting

### Common Issues:

1. **"API control is not enabled"**
   - Solution: Call `client.enableApiControl(True)` first

2. **"Drone is not armed"**
   - Solution: Call `client.armDisarm(True)` after enabling API control

3. **Operation timeout**
   - Solution: Increase timeout or check for obstacles

4. **Connection lost**
   - Solution: Ensure Blocks.exe is running and port 41451 is listening

---

## 📚 Next Steps

1. **Try the example scripts** in this directory
2. **Experiment with different flight patterns**
3. **Capture images during flight** for your research
4. **Develop custom missions** for data collection
5. **Implement obstacle avoidance** and safety features

---

**Happy Flying! 🚁**

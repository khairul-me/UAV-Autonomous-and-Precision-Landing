# test_airsim_connection.py
import airsim
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

print("✓ Connected to AirSim!")

# Enable API control
client.enableApiControl(True)
client.armDisarm(True)

# Take off
print("Taking off...")
client.takeoffAsync().join()

time.sleep(2)

# Get drone state
state = client.getMultirotorState()
print(f"Position: {state.kinematics_estimated.position}")
print(f"Velocity: {state.kinematics_estimated.linear_velocity}")

# Land
print("Landing...")
client.landAsync().join()

# Cleanup
client.armDisarm(False)
client.enableApiControl(False)


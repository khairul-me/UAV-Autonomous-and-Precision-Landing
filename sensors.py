# sensors.py
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

# 1. IMU Data (Inertial Measurement Unit)
imu_data = client.getImuData()
print(f"IMU Orientation: {imu_data.orientation}")
print(f"Angular Velocity: {imu_data.angular_velocity}")
print(f"Linear Acceleration: {imu_data.linear_acceleration}")

# 2. GPS Data
gps_data = client.getGpsData()
print(f"GPS Location: {gps_data.gnss.geo_point}")
print(f"GPS Velocity: {gps_data.gnss.velocity}")

# 3. Barometer Data
baro_data = client.getBarometerData()
print(f"Barometer Altitude: {baro_data.altitude}")
print(f"Barometer Pressure: {baro_data.pressure}")

# 4. Magnetometer Data
mag_data = client.getMagnetometerData()
print(f"Magnetic Field: {mag_data.magnetic_field_body}")

# 5. Lidar Data (if enabled in settings.json)
try:
    lidar_data = client.getLidarData()
    print(f"Number of Lidar points: {len(lidar_data.point_cloud)}")
except:
    print("Lidar not enabled in settings.json")

# 6. Collision Info (IMPORTANT for training!)
collision_info = client.simGetCollisionInfo()
print(f"Has collided: {collision_info.has_collided}")
print(f"Collision normal: {collision_info.normal}")
print(f"Impact point: {collision_info.impact_point}")

client.landAsync().join()


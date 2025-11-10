import airsim
import time


def main() -> None:
    """Basic AirSim connectivity smoke test."""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[OK] Connected to AirSim")

    client.enableApiControl(True)
    client.armDisarm(True)
    print("[OK] API control enabled")

    print("Taking off...")
    client.takeoffAsync().join()
    print("[OK] Drone is airborne")

    time.sleep(3)

    state = client.getMultirotorState()
    print(f"[OK] Position: {state.kinematics_estimated.position}")
    print(f"[OK] Velocity: {state.kinematics_estimated.linear_velocity}")

    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[OK] Test complete")


if __name__ == "__main__":
    main()


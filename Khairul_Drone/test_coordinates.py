import time

import airsim


def main() -> None:
    """Demonstrate basic movement in the NED frame."""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.moveToZAsync(-5, 3).join()
    print("[OK] Moved to 5m altitude")
    time.sleep(2)

    client.moveByVelocityAsync(5, 0, 0, 3).join()
    print("[OK] Moved forward (North)")
    time.sleep(1)

    client.moveByVelocityAsync(0, 5, 0, 3).join()
    print("[OK] Moved right (East)")
    time.sleep(1)

    client.moveByVelocityAsync(0, 0, -2, 3).join()
    print("[OK] Moved up")
    time.sleep(1)

    client.goHomeAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    print("[OK] Coordinate test complete")
    print("Remember: NED coordinate system")
    print("  X = North (Forward)")
    print("  Y = East (Right)")
    print("  Z = Down (negative = altitude)")


if __name__ == "__main__":
    main()


# test_environment.py
from environments.airsim_env import AirSimDroneEnv
import numpy as np
import time

env = AirSimDroneEnv()

# Test reset
print("Testing environment reset...")
obs = env.reset()
print(f"✓ Observation depth shape: {obs['depth_image'].shape}")
print(f"✓ Self-state shape: {obs['self_state'].shape}")
print(f"✓ Goal position: {env.goal_pos}")

# Test random actions
print("\nTesting random actions...")
for i in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    
    print(f"Step {i+1}: Reward={reward:.3f}, Dist to goal={info['distance_to_goal']:.2f}m, Done={done}")
    
    if done:
        print(f"Episode ended: {info}")
        break
    
    time.sleep(0.1)

env.close()
print("\n✓ Environment test complete!")


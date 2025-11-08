# environments/multi_agent_manager.py
import airsim
import numpy as np
import multiprocessing as mp

try:
    from environments.airsim_env import AirSimDroneEnv
except ImportError:
    # Fallback if AirSimDroneEnv not available
    AirSimDroneEnv = None

class MultiAgentManager:
    """
    Manages multiple drones in AirSim for distributed training
    Based on DPRL paper's multi-agent exploration strategy
    """
    
    def __init__(self, num_agents=3, ip_address="127.0.0.1"):
        self.num_agents = num_agents
        self.ip_address = ip_address
        
        # Vehicle names in AirSim
        self.vehicle_names = [f"Drone{i+1}" for i in range(num_agents)]
        
        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        
        print(f"✓ Connected to AirSim with {num_agents} drones")
        
        # Enable API control for all drones
        for vehicle_name in self.vehicle_names:
            self.client.enableApiControl(True, vehicle_name)
            self.client.armDisarm(True, vehicle_name)
        
        print(f"✓ All drones armed and ready")
    
    def reset_all(self):
        """Reset all drones to initial positions"""
        for vehicle_name in self.vehicle_names:
            self.client.reset(vehicle_name)
            self.client.enableApiControl(True, vehicle_name)
            self.client.armDisarm(True, vehicle_name)
    
    def takeoff_all(self):
        """Make all drones take off"""
        tasks = []
        for vehicle_name in self.vehicle_names:
            task = self.client.takeoffAsync(vehicle_name=vehicle_name)
            tasks.append(task)
        
        # Wait for all takeoffs
        for task in tasks:
            task.join()
        
        print("✓ All drones airborne")
    
    def get_states(self):
        """Get states of all drones"""
        states = {}
        for vehicle_name in self.vehicle_names:
            state = self.client.getMultirotorState(vehicle_name)
            states[vehicle_name] = state
        return states
    
    def move_all(self, velocities, duration=0.1):
        """
        Command all drones simultaneously
        
        Args:
            velocities: Dict {vehicle_name: (vx, vy, vz, yaw_rate)}
            duration: Control duration
        """
        tasks = []
        for vehicle_name, vel in velocities.items():
            vx, vy, vz, yaw_rate = vel
            task = self.client.moveByVelocityAsync(
                vx, vy, vz, duration,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                vehicle_name=vehicle_name
            )
            tasks.append(task)
        
        # Wait for all movements
        for task in tasks:
            task.join()
    
    def close(self):
        """Cleanup"""
        for vehicle_name in self.vehicle_names:
            self.client.armDisarm(False, vehicle_name)
            self.client.enableApiControl(False, vehicle_name)

class MultiAgentEnvironment:
    """
    Wrapper to run multiple drone environments in parallel
    Each drone operates in the same physical environment but with different goals
    """
    
    def __init__(self, num_agents=3, ip_address="127.0.0.1"):
        self.num_agents = num_agents
        
        # Create environment for each agent
        self.envs = []
        for i in range(num_agents):
            # Create env with vehicle name
            if AirSimDroneEnv is not None:
                # Modify AirSimDroneEnv to accept vehicle_name parameter
                env = AirSimDroneEnv(ip_address=ip_address)
                # Note: vehicle_name support would need to be added to AirSimDroneEnv
                # For now, we'll use the default implementation
                self.envs.append(env)
            else:
                raise ImportError("AirSimDroneEnv not available")
        
        print(f"✓ Created {num_agents} parallel environments")
    
    def reset_all(self):
        """Reset all environments"""
        observations = []
        for env in self.envs:
            obs = env.reset()
            observations.append(obs)
        return observations
    
    def step_all(self, actions):
        """
        Execute actions for all agents
        
        Args:
            actions: List of actions, one per agent
        
        Returns:
            List of (obs, reward, done, info) tuples
        """
        results = []
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            results.append(result)
        return results
    
    def close_all(self):
        """Close all environments"""
        for env in self.envs:
            env.close()

# Test multi-agent setup
if __name__ == "__main__":
    import time
    
    print("Testing Multi-Agent Setup...")
    print("="*60)
    
    # Option 1: Direct control of multiple drones
    print("\n1. Testing MultiAgentManager...")
    manager = MultiAgentManager(num_agents=3)
    
    manager.reset_all()
    manager.takeoff_all()
    
    # Move all drones forward simultaneously
    velocities = {
        "Drone1": (1.0, 0.0, 0.0, 0.0),
        "Drone2": (1.0, 0.0, 0.0, 0.0),
        "Drone3": (1.0, 0.0, 0.0, 0.0)
    }
    
    for i in range(10):
        manager.move_all(velocities, duration=0.5)
        states = manager.get_states()
        print(f"Step {i}: Drone1 pos = {states['Drone1'].kinematics_estimated.position}")
        time.sleep(0.1)
    
    manager.close()
    print("✓ MultiAgentManager test complete")
    
    # Option 2: Multiple independent environments
    print("\n2. Testing MultiAgentEnvironment...")
    multi_env = MultiAgentEnvironment(num_agents=3)
    
    observations = multi_env.reset_all()
    print(f"✓ All environments reset")
    
    # Random actions for all agents
    actions = [
        np.random.randn(4) for _ in range(3)
    ]
    
    results = multi_env.step_all(actions)
    print(f"✓ All agents executed actions")
    
    for i, (obs, reward, done, info) in enumerate(results):
        print(f"  Agent {i+1}: reward={reward:.2f}, done={done}")
    
    multi_env.close_all()
    print("✓ MultiAgentEnvironment test complete")


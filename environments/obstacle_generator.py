# environments/obstacle_generator.py
import airsim
import numpy as np
import json

class ObstacleGenerator:
    """
    Generate random obstacles in AirSim environment
    Following DPRL paper: 70 cylindrical obstacles
    """
    
    def __init__(self, client, environment_bounds=(-85, 85, -85, 85)):
        """
        Args:
            client: AirSim MultirotorClient
            environment_bounds: (x_min, x_max, y_min, y_max)
        """
        self.client = client
        self.x_min, self.x_max, self.y_min, self.y_max = environment_bounds
        
        self.obstacles = []
    
    def generate_random_obstacles(self, num_obstacles=70, radius=2.5, height=15.0):
        """
        Generate obstacles in circular pattern (like DPRL paper)
        
        Args:
            num_obstacles: Number of obstacles
            radius: Radius of each obstacle (meters)
            height: Height of obstacles (meters)
        """
        print(f"Generating {num_obstacles} obstacles...")
        
        self.obstacles = []
        
        # Place obstacles in circular pattern (like DPRL paper setup)
        # Obstacles arranged in circle of radius 60m
        circle_radius = 60.0
        
        for i in range(num_obstacles):
            # Angle for this obstacle
            angle = (2 * np.pi * i) / num_obstacles
            
            # Position on circle
            x = circle_radius * np.cos(angle)
            y = circle_radius * np.sin(angle)
            z = 0  # Ground level
            
            # Create obstacle info
            obstacle = {
                'name': f'Obstacle_{i}',
                'position': airsim.Vector3r(x, y, z),
                'radius': radius,
                'height': height
            }
            
            self.obstacles.append(obstacle)
            
            # Note: Actual spawning depends on Unreal Engine setup
            # In practice, obstacles are pre-placed in Unreal or spawned via blueprint
        
        print(f"[OK] Generated {num_obstacles} obstacle positions")
        return self.obstacles
    
    def save_obstacles(self, filename='obstacles.json'):
        """Save obstacle configuration to file"""
        obstacles_data = []
        for obs in self.obstacles:
            obstacles_data.append({
                'name': obs['name'],
                'position': {
                    'x': obs['position'].x_val,
                    'y': obs['position'].y_val,
                    'z': obs['position'].z_val
                },
                'radius': obs['radius'],
                'height': obs['height']
            })
        
        with open(filename, 'w') as f:
            json.dump(obstacles_data, f, indent=2)
        
        print(f"[OK] Saved obstacles to {filename}")
    
    def load_obstacles(self, filename='obstacles.json'):
        """Load obstacle configuration from file"""
        try:
            with open(filename, 'r') as f:
                obstacles_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Generating default obstacles.")
            self.generate_random_obstacles()
            self.save_obstacles(filename)
            return self.obstacles
        
        self.obstacles = []
        for obs_data in obstacles_data:
            obstacle = {
                'name': obs_data['name'],
                'position': airsim.Vector3r(
                    obs_data['position']['x'],
                    obs_data['position']['y'],
                    obs_data['position']['z']
                ),
                'radius': obs_data['radius'],
                'height': obs_data['height']
            }
            self.obstacles.append(obstacle)
        
        print(f"[OK] Loaded {len(self.obstacles)} obstacles from {filename}")
        return self.obstacles
    
    def check_collision(self, position, safety_margin=1.0):
        """
        Check if position is too close to any obstacle
        
        Args:
            position: airsim.Vector3r or (x, y, z) tuple
            safety_margin: Additional safety distance
        
        Returns:
            (is_collision, min_distance)
        """
        if isinstance(position, tuple):
            pos = airsim.Vector3r(*position)
        else:
            pos = position
        
        min_dist = float('inf')
        
        for obstacle in self.obstacles:
            # Distance in XY plane (ignore Z for cylindrical obstacles)
            dx = pos.x_val - obstacle['position'].x_val
            dy = pos.y_val - obstacle['position'].y_val
            dist_xy = np.sqrt(dx**2 + dy**2)
            
            # Subtract obstacle radius
            dist_to_surface = dist_xy - obstacle['radius']
            
            min_dist = min(min_dist, dist_to_surface)
        
        is_collision = min_dist < safety_margin
        
        return is_collision, min_dist
    
    def get_nearest_obstacle_distance(self, position):
        """Get distance to nearest obstacle surface"""
        _, min_dist = self.check_collision(position, safety_margin=0.0)
        return min_dist
    
    def visualize_obstacles(self):
        """Print obstacle positions for debugging"""
        print("\nObstacle Configuration:")
        print("="*60)
        for i, obs in enumerate(self.obstacles[:5]):  # Show first 5
            pos = obs['position']
            print(f"Obstacle {i}: pos=({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f}), "
                  f"radius={obs['radius']:.1f}m, height={obs['height']:.1f}m")
        print(f"... and {len(self.obstacles)-5} more")
        print("="*60)

# Test
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create dummy client (not actually connected)
    class DummyClient:
        pass
    
    client = DummyClient()
    
    # Generate obstacles
    gen = ObstacleGenerator(client)
    obstacles = gen.generate_random_obstacles(num_obstacles=70)
    
    # Save and load
    gen.save_obstacles('test_obstacles.json')
    gen.load_obstacles('test_obstacles.json')
    
    # Visualize
    gen.visualize_obstacles()
    
    # Plot obstacle layout
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for obs in obstacles:
        pos = obs['position']
        circle = plt.Circle((pos.x_val, pos.y_val), obs['radius'], 
                           color='gray', alpha=0.5, edgecolor='black')
        ax.add_patch(circle)
    
    # Plot flight space
    ax.plot([-85, 85, 85, -85, -85], [-85, -85, 85, 85, -85], 'r--', label='Flight bounds')
    
    # Plot target circle
    theta = np.linspace(0, 2*np.pi, 100)
    target_radius = 65
    ax.plot(target_radius*np.cos(theta), target_radius*np.sin(theta), 
           'g--', label='Target circle (65m)')
    
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('AirSim Obstacle Layout (DPRL-style)')
    
    plt.tight_layout()
    plt.savefig('obstacle_layout.png', dpi=150)
    print("\n[OK] Saved obstacle visualization")


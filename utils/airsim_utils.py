# utils/airsim_utils.py
import airsim
import numpy as np
import cv2
import os

class AirSimRecorder:
    """Record data from AirSim for debugging/analysis"""
    
    def __init__(self, save_dir='./recordings'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_data = []
        self.current_episode = []
    
    def record_step(self, observation, action, reward, done, info):
        """Record one step of data"""
        step_data = {
            'depth_image': observation['depth_image'],
            'self_state': observation['self_state'],
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        }
        self.current_episode.append(step_data)
        
        if done:
            self.episode_data.append(self.current_episode)
            self.current_episode = []
    
    def save_episode(self, episode_idx):
        """Save episode data"""
        if episode_idx >= len(self.episode_data):
            print(f"Episode {episode_idx} not found")
            return
        
        episode = self.episode_data[episode_idx]
        
        # Create episode directory
        ep_dir = os.path.join(self.save_dir, f'episode_{episode_idx}')
        os.makedirs(ep_dir, exist_ok=True)
        
        # Save images and data
        for step_idx, step in enumerate(episode):
            # Save depth image
            depth_img = (step['depth_image'] * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(ep_dir, f'depth_{step_idx:04d}.png'),
                depth_img
            )
            
            # Save metadata
            np.save(
                os.path.join(ep_dir, f'metadata_{step_idx:04d}.npy'),
                {
                    'state': step['self_state'],
                    'action': step['action'],
                    'reward': step['reward']
                }
            )
        
        print(f"✓ Saved episode {episode_idx} to {ep_dir}")

def visualize_trajectory(positions, obstacles, goal, save_path='trajectory.png'):
    """
    Visualize drone trajectory
    
    Args:
        positions: List of (x, y, z) positions
        obstacles: List of obstacle dicts from ObstacleGenerator
        goal: (x, y, z) goal position
        save_path: Where to save figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # Top view
    ax1 = fig.add_subplot(131)
    for obs in obstacles:
        pos = obs['position']
        circle = plt.Circle((pos.x_val, pos.y_val), obs['radius'], 
                           color='gray', alpha=0.5)
        ax1.add_patch(circle)
    
    positions_arr = np.array(positions)
    ax1.plot(positions_arr[:, 0], positions_arr[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(positions_arr[0, 0], positions_arr[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    ax1.set_xlim(-90, 90)
    ax1.set_ylim(-90, 90)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Top View')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    
    # Side view
    ax2 = fig.add_subplot(132)
    ax2.plot(positions_arr[:, 0], -positions_arr[:, 2], 'b-', linewidth=2)
    ax2.plot(positions_arr[0, 0], -positions_arr[0, 2], 'go', markersize=10)
    ax2.plot(goal[0], -goal[2], 'r*', markersize=15)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Side View')
    ax2.grid(True, alpha=0.3)
    
    # 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(positions_arr[:, 0], positions_arr[:, 1], positions_arr[:, 2], 'b-', linewidth=2)
    ax3.scatter(positions_arr[0, 0], positions_arr[0, 1], positions_arr[0, 2], 
               c='g', s=100, label='Start')
    ax3.scatter(goal[0], goal[1], goal[2], c='r', marker='*', s=200, label='Goal')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D View')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved trajectory visualization to {save_path}")
    plt.close()

def compute_metrics(episode_data):
    """
    Compute performance metrics for an episode
    
    Args:
        episode_data: List of step dicts
    
    Returns:
        Dict of metrics
    """
    total_reward = sum(step['reward'] for step in episode_data)
    episode_length = len(episode_data)
    
    final_info = episode_data[-1]['info']
    success = final_info['distance_to_goal'] < 2.0
    collision = final_info['collision']
    
    # Compute path length
    positions = [step['info']['position'] for step in episode_data]
    path_length = 0
    for i in range(len(positions)-1):
        p1 = np.array(positions[i])
        p2 = np.array(positions[i+1])
        path_length += np.linalg.norm(p2 - p1)
    
    # Compute smoothness (sum of action changes)
    actions = [step['action'] for step in episode_data]
    smoothness = 0
    for i in range(len(actions)-1):
        a1 = np.array(actions[i])
        a2 = np.array(actions[i+1])
        smoothness += np.linalg.norm(a2 - a1)
    smoothness = smoothness / len(actions) if len(actions) > 0 else 0
    
    metrics = {
        'total_reward': total_reward,
        'episode_length': episode_length,
        'success': success,
        'collision': collision,
        'final_distance': final_info['distance_to_goal'],
        'path_length': path_length,
        'smoothness': smoothness
    }
    
    return metrics


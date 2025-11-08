# train_baseline.py
import numpy as np
import torch
import os
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.airsim_env import AirSimDroneEnv
from algorithms.td3 import TD3
from algorithms.replay_buffer import ReplayBuffer

class Trainer:
    """
    Training pipeline for baseline drone navigation
    No adversarial attacks yet - just clean training
    """
    
    def __init__(
        self,
        env,
        agent,
        replay_buffer,
        max_episodes=1000,
        max_steps=500,
        batch_size=128,
        learning_starts=2000,
        save_dir='./checkpoints',
        log_interval=10,
        eval_interval=50,
        eval_episodes=10
    ):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.eval_rewards = []
        
        # Training stats
        self.total_steps = 0
        self.training_start_time = None
    
    def train(self):
        """Main training loop"""
        
        print("="*60)
        print("Starting Baseline Training (No Adversarial Attacks)")
        print("="*60)
        print(f"Max episodes: {self.max_episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        print(f"Learning starts after: {self.learning_starts} steps")
        print(f"Batch size: {self.batch_size}")
        print("="*60)
        
        self.training_start_time = time.time()
        
        for episode in range(self.max_episodes):
            episode_start = time.time()
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment
            obs = self.env.reset()
            depth = obs['depth_image']
            state = obs['self_state']
            
            for step in range(self.max_steps):
                self.total_steps += 1
                
                # Select action
                if self.total_steps < self.learning_starts:
                    # Random exploration initially
                    action = self.env.action_space.sample()
                else:
                    # Use policy with exploration noise
                    action = self.agent.select_action(depth, state, add_noise=True)
                
                # Execute action
                next_obs, reward, done, info = self.env.step(action)
                next_depth = next_obs['depth_image']
                next_state = next_obs['self_state']
                
                # Store transition
                self.replay_buffer.add(
                    depth, state, action, reward,
                    next_depth, next_state, float(done)
                )
                
                # Update state
                depth = next_depth
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Train agent
                if self.total_steps >= self.learning_starts:
                    metrics = self.agent.train(self.replay_buffer, self.batch_size)
                
                if done:
                    break
            
            # Log episode stats
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # Calculate success (reached goal)
            success = info['distance_to_goal'] < self.env.goal_thresh
            
            # Print progress
            if episode % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                avg_length = np.mean(self.episode_lengths[-self.log_interval:])
                elapsed = time.time() - self.training_start_time
                
                print(f"\n[Episode {episode}/{self.max_episodes}]")
                print(f"  Total steps: {self.total_steps}")
                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Avg reward ({self.log_interval} eps): {avg_reward:.2f}")
                print(f"  Episode length: {episode_steps}")
                print(f"  Success: {success}")
                print(f"  Distance to goal: {info['distance_to_goal']:.2f}m")
                print(f"  Elapsed time: {elapsed/60:.1f} min")
                print(f"  Buffer size: {len(self.replay_buffer)}")
            
            # Evaluate agent
            if episode % self.eval_interval == 0 and self.total_steps >= self.learning_starts:
                eval_reward, eval_success = self.evaluate()
                self.eval_rewards.append((episode, eval_reward))
                self.success_rates.append((episode, eval_success))
                
                print(f"\n{'='*60}")
                print(f"EVALUATION @ Episode {episode}")
                print(f"  Avg reward: {eval_reward:.2f}")
                print(f"  Success rate: {eval_success*100:.1f}%")
                print(f"{'='*60}\n")
                
                # Save checkpoint
                self.save_checkpoint(episode, eval_reward, eval_success)
        
        # Final evaluation
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        final_reward, final_success = self.evaluate()
        print(f"Final evaluation:")
        print(f"  Avg reward: {final_reward:.2f}")
        print(f"  Success rate: {final_success*100:.1f}%")
        print("="*60)
        
        # Save final model
        self.save_checkpoint('final', final_reward, final_success)
        
        # Plot training curves
        self.plot_training_curves()
    
    def evaluate(self):
        """Evaluate agent without exploration noise"""
        eval_rewards = []
        eval_successes = []
        
        for _ in range(self.eval_episodes):
            obs = self.env.reset()
            depth = obs['depth_image']
            state = obs['self_state']
            
            episode_reward = 0
            done = False
            
            while not done:
                # Deterministic action (no noise)
                action = self.agent.select_action(depth, state, add_noise=False)
                
                next_obs, reward, done, info = self.env.step(action)
                depth = next_obs['depth_image']
                state = next_obs['self_state']
                
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_successes.append(info['distance_to_goal'] < self.env.goal_thresh)
        
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_successes)
        
        return avg_reward, success_rate
    
    def save_checkpoint(self, episode, reward, success_rate):
        """Save model checkpoint"""
        filename = os.path.join(
            self.save_dir,
            f'baseline_ep{episode}_r{reward:.1f}_sr{success_rate:.2f}.pth'
        )
        self.agent.save(filename)
        print(f"✓ Saved checkpoint: {filename}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        # Evaluation rewards
        if len(self.eval_rewards) > 0:
            episodes, rewards = zip(*self.eval_rewards)
            axes[1, 0].plot(episodes, rewards, marker='o')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Avg Reward')
            axes[1, 0].set_title('Evaluation Rewards')
            axes[1, 0].grid(True)
        
        # Success rates
        if len(self.success_rates) > 0:
            episodes, rates = zip(*self.success_rates)
            axes[1, 1].plot(episodes, [r*100 for r in rates], marker='o')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_title('Success Rate Over Training')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300)
        print(f"✓ Saved training curves: {save_path}")
        
        plt.close()

# Main training script
if __name__ == "__main__":
    
    # Initialize environment
    print("Initializing environment...")
    env = AirSimDroneEnv()
    
    # Initialize agent
    print("Initializing TD3 agent...")
    agent = TD3(
        state_dim=8,
        action_dim=4,
        max_action=[3.0, 3.0, 2.0, 0.3],
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4
    )
    
    # Initialize replay buffer
    print("Initializing replay buffer...")
    replay_buffer = ReplayBuffer(
        max_size=50000,
        depth_shape=(80, 100),
        state_dim=8,
        action_dim=4
    )
    
    # Initialize trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        max_episodes=1000,
        max_steps=500,
        batch_size=128,
        learning_starts=2000,
        save_dir='./checkpoints/baseline',
        log_interval=10,
        eval_interval=50,
        eval_episodes=10
    )
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    finally:
        env.close()
        print("Environment closed.")


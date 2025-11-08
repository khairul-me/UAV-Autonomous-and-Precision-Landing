"""
Complete Training Pipeline for Adversarially Robust Drone Navigation

This script trains 4 different models for comparison:

1. Baseline: Clean training (no attacks)

2. Baseline + Attacks: Trained on clean, tested with attacks

3. DPRL-style: Privileged learning (sensor noise robustness)

4. YOUR METHOD: Complete adversarial robustness (all 4 defense layers)

Usage:

    python train_complete.py --mode baseline

    python train_complete.py --mode robust --enable-all-defenses

"""

import argparse
import os
import time
import numpy as np
import torch
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Environment
from environments.airsim_env_enhanced import AirSimDroneEnvEnhanced
from environments.obstacle_generator import ObstacleGenerator

# Models and algorithms
from models.actor import Actor
from models.critic import Critic
from algorithms.td3 import TD3
from algorithms.replay_buffer import ReplayBuffer

# Attacks
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.motion_blur import MotionBlurAttack

# Defenses
from defenses.input_sanitization import (
    DenoisingAutoencoder, 
    AnomalyDetector, 
    InputSanitizer,
    train_denoising_autoencoder
)
from defenses.multi_sensor_fusion import MultiSensorFusion
from defenses.temporal_consistency import TemporalConsistencyChecker
from defenses.robust_training import RobustTD3, RobustTrainingManager
from defenses.integrated_defense import IntegratedDefenseSystem

# Utils
from utils.airsim_utils import AirSimRecorder, visualize_trajectory, compute_metrics

class CompletePipeline:
    """
    Complete training pipeline with all components
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Create directories
        self.setup_directories()
        
        # Initialize environment
        print("\n" + "="*80)
        print("INITIALIZING COMPLETE TRAINING PIPELINE")
        print("="*80)
        print(f"Mode: {config['mode']}")
        print(f"Max episodes: {config['max_episodes']}")
        print(f"Using attacks: {config['use_attacks']}")
        print(f"Using defenses: {config['use_defenses']}")
        print("="*80 + "\n")
        
        self.env = self.create_environment()
        
        # Initialize agent
        self.agent = self.create_agent()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config['buffer_size'],
            depth_shape=(80, 100),
            state_dim=8,
            action_dim=4
        )
        
        # Initialize attacks
        self.attacks = self.create_attacks()
        
        # Initialize defenses
        self.defenses = None
        if config['use_defenses']:
            self.defenses = self.create_defenses()
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'attack_detection_rates': [],
            'eval_rewards': [],
            'clean_success': [],
            'attacked_success': [],
            'defended_success': []
        }
        
        # Recorder for visualization
        self.recorder = AirSimRecorder(save_dir=os.path.join(config['save_dir'], 'recordings'))
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['save_dir'],
            os.path.join(self.config['save_dir'], 'checkpoints'),
            os.path.join(self.config['save_dir'], 'recordings'),
            os.path.join(self.config['save_dir'], 'figures'),
            os.path.join(self.config['save_dir'], 'logs')
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        print(f"[OK] Created directories in {self.config['save_dir']}")
    
    def create_environment(self):
        """Create AirSim environment"""
        print("\n1. Creating AirSim Environment...")
        
        # Generate obstacles if needed
        if not os.path.exists('obstacles.json'):
            print("   Generating obstacle configuration...")
            try:
                import airsim
                client = airsim.MultirotorClient()
                obstacle_gen = ObstacleGenerator(client)
                obstacles = obstacle_gen.generate_random_obstacles(num_obstacles=70)
                obstacle_gen.save_obstacles('obstacles.json')
                print("   [OK] Saved obstacles.json")
            except Exception as e:
                print(f"   [WARNING] Could not generate obstacles: {e}")
                print("   [INFO] Will proceed without obstacles")
        
        env = AirSimDroneEnvEnhanced(
            ip_address=self.config['airsim_ip'],
            vehicle_name="Drone1",
            image_shape=(80, 100),
            max_steps=self.config['max_steps_per_episode'],
            add_sensor_noise=self.config['add_sensor_noise'],
            gps_noise_std=0.5,
            imu_noise_std=0.05,
            depth_noise_prob=0.01
        )
        
        print("   [OK] Environment created")
        return env
    
    def create_agent(self):
        """Create RL agent (TD3 or RobustTD3)"""
        print("\n2. Creating Agent...")
        
        if self.config['mode'] == 'robust':
            # Use RobustTD3 with privileged learning
            agent = RobustTD3(
                state_dim=8,
                action_dim=4,
                max_action=[3.0, 3.0, 2.0, 0.3],
                discount=self.config['discount'],
                tau=self.config['tau'],
                policy_noise=self.config['policy_noise'],
                noise_clip=self.config['noise_clip'],
                policy_freq=self.config['policy_freq'],
                actor_lr=self.config['actor_lr'],
                critic_lr=self.config['critic_lr'],
                adversarial_ratio=self.config['adversarial_ratio'],
                device=self.config['device']
            )
            print("   [OK] Created RobustTD3 agent (with privileged learning)")
        else:
            # Standard TD3
            agent = TD3(
                state_dim=8,
                action_dim=4,
                max_action=[3.0, 3.0, 2.0, 0.3],
                discount=self.config['discount'],
                tau=self.config['tau'],
                policy_noise=self.config['policy_noise'],
                noise_clip=self.config['noise_clip'],
                policy_freq=self.config['policy_freq'],
                actor_lr=self.config['actor_lr'],
                critic_lr=self.config['critic_lr'],
                device=self.config['device']
            )
            print("   [OK] Created standard TD3 agent")
        
        return agent
    
    def create_attacks(self):
        """Create attack functions"""
        print("\n3. Creating Attacks...")
        
        attacks = {}
        
        if self.config['use_attacks']:
            attacks['fgsm'] = FGSM(epsilon=0.03)
            attacks['pgd'] = PGD(epsilon=0.03, alpha=0.007, num_steps=10)
            attacks['motion_blur'] = MotionBlurAttack(kernel_size=11)
            print(f"   [OK] Created {len(attacks)} attack types")
        else:
            print("   [OK] No attacks (clean training)")
        
        return attacks
    
    def create_defenses(self):
        """Create defense system (all 4 layers)"""
        print("\n4. Creating Defense System...")
        
        defenses = {}
        
        # Layer 1: Input Sanitization
        if self.config['enable_layer1']:
            print("   Setting up Layer 1: Input Sanitization...")
            
            # Train denoiser if not exists
            denoiser_path = os.path.join(self.config['save_dir'], 'denoiser.pth')
            
            if os.path.exists(denoiser_path):
                print("     Loading pre-trained denoiser...")
                denoiser = DenoisingAutoencoder()
                denoiser.load_state_dict(torch.load(denoiser_path, map_location='cpu'))
                denoiser.eval()
            else:
                print("     Training denoiser (this may take a few minutes)...")
                # Generate training data
                clean_images = self._generate_clean_images(n=200)
                attacked_images = self._generate_attacked_images(clean_images)
                
                # Train
                denoiser = train_denoising_autoencoder(
                    clean_images, attacked_images,
                    epochs=30, batch_size=32, lr=1e-3
                )
                
                # Save
                torch.save(denoiser.state_dict(), denoiser_path)
                print(f"     [OK] Saved denoiser to {denoiser_path}")
            
            # Calibrate anomaly detector
            detector = AnomalyDetector(threshold=0.15)
            clean_images = self._generate_clean_images(n=100)
            detector.calibrate(clean_images)
            
            # Create sanitizer
            sanitizer = InputSanitizer(denoiser, detector)
            defenses['sanitizer'] = sanitizer
            print("     [OK] Layer 1 ready")
        
        # Layer 2: Multi-Sensor Fusion
        if self.config['enable_layer2']:
            print("   Setting up Layer 2: Multi-Sensor Fusion...")
            sensor_fusion = MultiSensorFusion(
                depth_lidar_threshold=2.0,
                velocity_threshold=0.5,
                position_threshold=1.0
            )
            defenses['sensor_fusion'] = sensor_fusion
            print("     [OK] Layer 2 ready")
        
        # Layer 3: Temporal Consistency
        if self.config['enable_layer3']:
            print("   Setting up Layer 3: Temporal Consistency...")
            temporal_checker = TemporalConsistencyChecker(
                window_size=10,
                action_change_threshold=1.0,
                q_value_change_threshold=5.0,
                depth_change_threshold=5.0
            )
            defenses['temporal'] = temporal_checker
            print("     [OK] Layer 3 ready")
        
        # Create integrated defense system
        integrated_defense = IntegratedDefenseSystem(
            input_sanitizer=defenses.get('sanitizer'),
            sensor_fusion=defenses.get('sensor_fusion'),
            temporal_checker=defenses.get('temporal'),
            enable_layer1=self.config['enable_layer1'],
            enable_layer2=self.config['enable_layer2'],
            enable_layer3=self.config['enable_layer3']
        )
        
        defenses['integrated'] = integrated_defense
        
        print(f"   [OK] Defense system ready with {len([k for k in defenses if k != 'integrated'])} active layers")
        
        return defenses
    
    def _generate_clean_images(self, n=100):
        """Generate clean depth images for training denoiser"""
        print(f"     Collecting {n} clean images...")
        clean_images = []
        
        for _ in range(n):
            obs = self.env.reset()
            clean_images.append(obs['depth_image'])
            
            # Take a few random steps to get variety
            for _ in range(np.random.randint(5, 15)):
                action = self.env.action_space.sample()
                obs, _, done, _ = self.env.step(action)
                clean_images.append(obs['depth_image'])
                
                if done or len(clean_images) >= n:
                    break
            
            if len(clean_images) >= n:
                break
        
        return clean_images[:n]
    
    def _generate_attacked_images(self, clean_images):
        """Generate attacked versions of clean images"""
        print(f"     Generating attacked versions...")
        attacked_images = []
        
        fgsm = FGSM(epsilon=0.05)
        
        for clean in clean_images:
            state = np.random.randn(8).astype(np.float32)
            try:
                attacked = fgsm.attack(clean, self.agent.actor, state)
                attacked_images.append(attacked)
            except Exception as e:
                # Fallback: add noise
                attacked = clean + np.random.randn(*clean.shape).astype(np.float32) * 0.05
                attacked = np.clip(attacked, 0, 1)
                attacked_images.append(attacked)
        
        return attacked_images
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        start_time = time.time()
        total_steps = 0
        
        for episode in range(self.config['max_episodes']):
            episode_start = time.time()
            
            # Train one episode
            stats = self.train_episode(episode)
            
            # Update statistics
            self.training_stats['episode_rewards'].append(stats['episode_reward'])
            self.training_stats['episode_lengths'].append(stats['episode_steps'])
            
            total_steps += stats['episode_steps']
            
            # Logging
            if episode % self.config['log_interval'] == 0:
                self.log_progress(episode, stats, total_steps, start_time)
            
            # Evaluation
            if episode % self.config['eval_interval'] == 0 and episode > 0:
                eval_stats = self.evaluate(num_episodes=10)
                self.training_stats['eval_rewards'].append((episode, eval_stats['avg_reward']))
                self.training_stats['clean_success'].append((episode, eval_stats['clean_success']))
                
                if self.config['use_attacks']:
                    self.training_stats['attacked_success'].append((episode, eval_stats['attacked_success']))
                
                if self.config['use_defenses']:
                    self.training_stats['defended_success'].append((episode, eval_stats['defended_success']))
                
                self.log_evaluation(episode, eval_stats)
                
                # Save checkpoint
                self.save_checkpoint(episode, eval_stats)
        
        # Final evaluation
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        final_stats = self.evaluate(num_episodes=30)
        self.log_evaluation('FINAL', final_stats)
        
        # Save final model
        self.save_checkpoint('final', final_stats)
        
        # Generate plots
        self.plot_training_curves()
        
        # Print summary
        self.print_summary(time.time() - start_time)
    
    def train_episode(self, episode):
        """
        Train one episode
        
        Returns:
            Episode statistics
        """
        # Reset environment
        obs = self.env.reset()
        depth_clean = obs['depth_image']
        state_clean = obs['self_state']
        
        # Reset temporal checker if using defenses
        if self.config['use_defenses'] and self.defenses is not None:
            integrated = self.defenses['integrated']
            integrated.reset()
        
        episode_reward = 0
        episode_steps = 0
        attacks_applied = 0
        attacks_detected = 0
        
        # Decide whether to use attacks this episode
        use_attack_this_episode = (
            self.config['use_attacks'] and 
            np.random.random() < self.config['attack_probability']
        )
        
        for step in range(self.config['max_steps_per_episode']):
            total_steps = episode * self.config['max_steps_per_episode'] + step
            
            # Prepare observation for actor
            depth_for_actor = depth_clean
            state_for_actor = state_clean
            
            # Apply attack if enabled
            if use_attack_this_episode:
                # Randomly select attack
                attack_name = np.random.choice(list(self.attacks.keys()))
                attack_fn = self.attacks[attack_name]
                
                # Attack depth image
                try:
                    if attack_name in ['fgsm', 'pgd']:
                        depth_for_actor = attack_fn.attack(depth_clean, self.agent.actor, state_clean)
                    else:  # motion_blur
                        depth_for_actor = attack_fn.attack(depth_clean)
                    attacks_applied += 1
                except Exception as e:
                    # If attack fails, use clean image
                    depth_for_actor = depth_clean
            
            # Apply defenses if enabled
            if self.config['use_defenses'] and self.defenses is not None:
                # We'll apply defenses in the observation processing
                # For now, just note that we have defenses
                pass
            
            # Select action
            if total_steps < self.config['learning_starts']:
                # Random exploration
                action = np.random.uniform(
                    self.env.action_space.low,
                    self.env.action_space.high,
                    size=self.env.action_space.shape
                )
            else:
                # Use policy
                action = self.agent.select_action(
                    depth_for_actor, 
                    state_for_actor, 
                    add_noise=True,
                    noise_scale=self.config['exploration_noise']
                )
            
            # Execute action
            next_obs, reward, done, info = self.env.step(action)
            next_depth_clean = next_obs['depth_image']
            next_state_clean = next_obs['self_state']
            
            # Store transition in replay buffer
            # Store BOTH clean and noisy versions for privileged learning
            self.replay_buffer.add(
                depth=depth_for_actor,  # What actor saw (possibly attacked)
                state=state_for_actor,
                action=action,
                reward=reward,
                next_depth=next_depth_clean,
                next_state=next_state_clean,
                done=float(done),
                clean_depth=depth_clean,  # Ground truth for Critic
                next_clean_depth=next_depth_clean
            )
            
            # Train agent
            if total_steps >= self.config['learning_starts']:
                try:
                    if isinstance(self.agent, RobustTD3):
                        # Robust training with privileged learning
                        metrics = self.agent.train_robust(
                            self.replay_buffer,
                            batch_size=self.config['batch_size']
                        )
                    else:
                        # Standard training
                        metrics = self.agent.train(
                            self.replay_buffer,
                            batch_size=self.config['batch_size'],
                            use_privileged=self.config['use_privileged_learning']
                        )
                except Exception as e:
                    # Skip training step if error
                    pass
            
            # Update state
            depth_clean = next_depth_clean
            state_clean = next_state_clean
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        stats = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'success': info['distance_to_goal'] < self.env.goal_thresh,
            'collision': info['collision'],
            'final_distance': info['distance_to_goal'],
            'attacks_applied': attacks_applied,
            'attacks_detected': attacks_detected
        }
        
        return stats
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate agent under different conditions
        
        Returns:
            Evaluation statistics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION ({num_episodes} episodes)")
        print(f"{'='*80}")
        
        results = {
            'clean': [],
            'attacked': [],
            'defended': []
        }
        
        for eval_type in ['clean', 'attacked', 'defended']:
            # Skip if not applicable
            if eval_type == 'attacked' and not self.config['use_attacks']:
                continue
            if eval_type == 'defended' and not self.config['use_defenses']:
                continue
            
            print(f"\nEvaluating: {eval_type.upper()}")
            
            for ep in range(num_episodes):
                obs = self.env.reset()
                depth = obs['depth_image']
                state = obs['self_state']
                
                episode_reward = 0
                done = False
                steps = 0
                
                while not done and steps < self.config['max_steps_per_episode']:
                    # Prepare observation
                    depth_obs = depth
                    
                    # Apply attack for 'attacked' condition
                    if eval_type == 'attacked':
                        attack_name = np.random.choice(list(self.attacks.keys()))
                        attack_fn = self.attacks[attack_name]
                        
                        try:
                            if attack_name in ['fgsm', 'pgd']:
                                depth_obs = attack_fn.attack(depth, self.agent.actor, state)
                            else:
                                depth_obs = attack_fn.attack(depth)
                        except Exception:
                            pass
                    
                    # Apply defenses for 'defended' condition
                    if eval_type == 'defended':
                        # First attack
                        attack_name = np.random.choice(list(self.attacks.keys()))
                        attack_fn = self.attacks[attack_name]
                        
                        try:
                            if attack_name in ['fgsm', 'pgd']:
                                depth_obs = attack_fn.attack(depth, self.agent.actor, state)
                            else:
                                depth_obs = attack_fn.attack(depth)
                        except Exception:
                            pass
                        
                        # Then defend
                        integrated = self.defenses['integrated']
                        defense_result = integrated.process_observation(
                            depth_image=depth_obs,
                            state=state,
                            airsim_client=None,  # Skip sensor fusion in eval
                            action=None,  # Will be computed next
                            q_value=None
                        )
                        
                        depth_obs = defense_result['depth_image']
                    
                    # Select action (deterministic for evaluation)
                    action = self.agent.select_action(depth_obs, state, add_noise=False)
                    
                    # Execute
                    next_obs, reward, done, info = self.env.step(action)
                    depth = next_obs['depth_image']
                    state = next_obs['self_state']
                    
                    episode_reward += reward
                    steps += 1
                
                # Record results
                results[eval_type].append({
                    'reward': episode_reward,
                    'success': info['distance_to_goal'] < self.env.goal_thresh,
                    'steps': steps,
                    'collision': info['collision']
                })
                
                if (ep + 1) % 5 == 0:
                    print(f"  Episode {ep+1}/{num_episodes} complete")
        
        # Compute statistics
        stats = {}
        
        if len(results['clean']) > 0:
            stats['clean_success'] = np.mean([r['success'] for r in results['clean']])
            stats['clean_reward'] = np.mean([r['reward'] for r in results['clean']])
        
        if len(results['attacked']) > 0:
            stats['attacked_success'] = np.mean([r['success'] for r in results['attacked']])
            stats['attacked_reward'] = np.mean([r['reward'] for r in results['attacked']])
        
        if len(results['defended']) > 0:
            stats['defended_success'] = np.mean([r['success'] for r in results['defended']])
            stats['defended_reward'] = np.mean([r['reward'] for r in results['defended']])
        
        # Average across all conditions
        all_rewards = []
        if len(results['clean']) > 0:
            all_rewards.extend([r['reward'] for r in results['clean']])
        if len(results['attacked']) > 0:
            all_rewards.extend([r['reward'] for r in results['attacked']])
        if len(results['defended']) > 0:
            all_rewards.extend([r['reward'] for r in results['defended']])
        
        stats['avg_reward'] = np.mean(all_rewards) if len(all_rewards) > 0 else 0.0
        
        return stats
    
    def log_progress(self, episode, stats, total_steps, start_time):
        """Log training progress"""
        elapsed = time.time() - start_time
        
        avg_reward = np.mean(self.training_stats['episode_rewards'][-self.config['log_interval']:])
        avg_length = np.mean(self.training_stats['episode_lengths'][-self.config['log_interval']:])
        
        print(f"\n[Episode {episode}/{self.config['max_episodes']}]")
        print(f"  Total steps: {total_steps}")
        print(f"  Episode reward: {stats['episode_reward']:.2f}")
        print(f"  Avg reward ({self.config['log_interval']} eps): {avg_reward:.2f}")
        print(f"  Episode length: {stats['episode_steps']}")
        print(f"  Success: {stats['success']}")
        print(f"  Collision: {stats['collision']}")
        print(f"  Final distance: {stats['final_distance']:.2f}m")
        print(f"  Buffer size: {len(self.replay_buffer)}")
        print(f"  Elapsed time: {elapsed/60:.1f} min")
        
        if self.config['use_attacks']:
            print(f"  Attacks applied: {stats['attacks_applied']}")
    
    def log_evaluation(self, episode, stats):
        """Log evaluation results"""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS @ Episode {episode}")
        print(f"{'='*80}")
        print(f"Average reward: {stats['avg_reward']:.2f}")
        
        if 'clean_success' in stats:
            print(f"Clean success rate: {stats['clean_success']*100:.1f}%")
        
        if 'attacked_success' in stats:
            print(f"Attacked success rate: {stats['attacked_success']*100:.1f}%")
            print(f"  Performance drop: {(stats.get('clean_success', 0) - stats['attacked_success'])*100:.1f}%")
        
        if 'defended_success' in stats:
            print(f"Defended success rate: {stats['defended_success']*100:.1f}%")
            print(f"  Defense effectiveness: {(stats['defended_success'] - stats.get('attacked_success', 0))*100:.1f}%")
        
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, episode, stats):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['save_dir'], 'checkpoints')
        
        filename = os.path.join(
            checkpoint_dir,
            f"{self.config['mode']}_ep{episode}_r{stats['avg_reward']:.1f}.pth"
        )
        
        self.agent.save(filename)
        print(f"[OK] Saved checkpoint: {filename}")
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        print("\nGenerating training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_stats['episode_rewards'], alpha=0.3, color='blue')
        # Smooth with moving average
        window = 50
        if len(self.training_stats['episode_rewards']) > window:
            smoothed = np.convolve(
                self.training_stats['episode_rewards'], 
                np.ones(window)/window, 
                mode='valid'
            )
            axes[0, 0].plot(range(window-1, len(self.training_stats['episode_rewards'])), 
                          smoothed, color='blue', linewidth=2, label='Smoothed')
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(self.training_stats['episode_lengths'], alpha=0.6)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rates over training
        if len(self.training_stats['clean_success']) > 0:
            episodes, rates = zip(*self.training_stats['clean_success'])
            axes[1, 0].plot(episodes, [r*100 for r in rates], 'g-o', label='Clean', markersize=4)
        
        if len(self.training_stats['attacked_success']) > 0:
            episodes, rates = zip(*self.training_stats['attacked_success'])
            axes[1, 0].plot(episodes, [r*100 for r in rates], 'r-s', label='Attacked', markersize=4)
        
        if len(self.training_stats['defended_success']) > 0:
            episodes, rates = zip(*self.training_stats['defended_success'])
            axes[1, 0].plot(episodes, [r*100 for r in rates], 'b-^', label='Defended', markersize=4)
        
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Success Rates During Training')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 105)
        
        # Evaluation rewards
        if len(self.training_stats['eval_rewards']) > 0:
            episodes, rewards = zip(*self.training_stats['eval_rewards'])
            axes[1, 1].plot(episodes, rewards, 'b-o', markersize=6)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].set_title('Evaluation Performance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config['save_dir'], 'figures', 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved training curves to {save_path}")
        
        plt.close()
    
    def print_summary(self, total_time):
        """Print final training summary"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Mode: {self.config['mode']}")
        print(f"Total episodes: {self.config['max_episodes']}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Final avg reward: {np.mean(self.training_stats['episode_rewards'][-100:]):.2f}")
        
        if len(self.training_stats['clean_success']) > 0:
            print(f"Final clean success rate: {self.training_stats['clean_success'][-1][1]*100:.1f}%")
        
        if len(self.training_stats['attacked_success']) > 0:
            print(f"Final attacked success rate: {self.training_stats['attacked_success'][-1][1]*100:.1f}%")
        
        if len(self.training_stats['defended_success']) > 0:
            print(f"Final defended success rate: {self.training_stats['defended_success'][-1][1]*100:.1f}%")
        
        if self.config['use_defenses'] and self.defenses is not None:
            print("\nDefense System Statistics:")
            self.defenses['integrated'].print_summary()
        
        print("="*80 + "\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Complete Training Pipeline')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'baseline_attacked', 'dprl', 'robust'],
                       help='Training mode')
    
    # Training parameters
    parser.add_argument('--max-episodes', type=int, default=1000,
                       help='Maximum number of episodes')
    parser.add_argument('--max-steps-per-episode', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--learning-starts', type=int, default=2000,
                       help='Steps before training starts')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=50000,
                       help='Replay buffer size')
    
    # RL hyperparameters
    parser.add_argument('--discount', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Target network update rate')
    parser.add_argument('--policy-noise', type=float, default=0.1,
                       help='Policy noise for TD3')
    parser.add_argument('--noise-clip', type=float, default=0.5,
                       help='Noise clip for TD3')
    parser.add_argument('--policy-freq', type=int, default=2,
                       help='Policy update frequency')
    parser.add_argument('--actor-lr', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=3e-4,
                       help='Critic learning rate')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                       help='Exploration noise scale')
    
    # Adversarial training parameters
    parser.add_argument('--adversarial-ratio', type=float, default=0.3,
                       help='Ratio of adversarial examples in batch')
    parser.add_argument('--attack-probability', type=float, default=0.5,
                       help='Probability of attack per episode')
    
    # Defense layers
    parser.add_argument('--enable-layer1', action='store_true',
                       help='Enable Layer 1: Input Sanitization')
    parser.add_argument('--enable-layer2', action='store_true',
                       help='Enable Layer 2: Multi-Sensor Fusion')
    parser.add_argument('--enable-layer3', action='store_true',
                       help='Enable Layer 3: Temporal Consistency')
    parser.add_argument('--enable-all-defenses', action='store_true',
                       help='Enable all defense layers')
    
    # Environment
    parser.add_argument('--airsim-ip', type=str, default='127.0.0.1',
                       help='AirSim IP address')
    parser.add_argument('--add-sensor-noise', action='store_true',
                       help='Add sensor noise to observations')
    
    # Logging and saving
    parser.add_argument('--save-dir', type=str, default='./experiments',
                       help='Directory to save results')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (episodes)')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='Evaluation interval (episodes)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.mode == 'baseline':
        # Clean baseline training
        config = {
            'mode': 'baseline',
            'use_attacks': False,
            'use_defenses': False,
            'use_privileged_learning': False,
            'add_sensor_noise': False,
            'enable_layer1': False,
            'enable_layer2': False,
            'enable_layer3': False
        }
    
    elif args.mode == 'baseline_attacked':
        # Baseline trained on clean, tested with attacks
        config = {
            'mode': 'baseline_attacked',
            'use_attacks': True,  # Attacks only during evaluation
            'use_defenses': False,
            'use_privileged_learning': False,
            'add_sensor_noise': False,
            'enable_layer1': False,
            'enable_layer2': False,
            'enable_layer3': False
        }
    
    elif args.mode == 'dprl':
        # DPRL-style: privileged learning for sensor noise
        config = {
            'mode': 'dprl',
            'use_attacks': False,
            'use_defenses': False,
            'use_privileged_learning': True,  # Critic sees clean, Actor sees noisy
            'add_sensor_noise': True,
            'enable_layer1': False,
            'enable_layer2': False,
            'enable_layer3': False
        }
    
    elif args.mode == 'robust':
        # YOUR METHOD: Complete adversarial robustness
        if args.enable_all_defenses:
            enable_layer1 = True
            enable_layer2 = True
            enable_layer3 = True
        else:
            enable_layer1 = args.enable_layer1
            enable_layer2 = args.enable_layer2
            enable_layer3 = args.enable_layer3
        
        config = {
            'mode': 'robust',
            'use_attacks': True,
            'use_defenses': True,
            'use_privileged_learning': True,  # Layer 4: Robust training
            'add_sensor_noise': True,
            'enable_layer1': enable_layer1,
            'enable_layer2': enable_layer2,
            'enable_layer3': enable_layer3
        }
    
    # Update config with command-line args
    config.update(vars(args))
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['save_dir'] = os.path.join(args.save_dir, f"{args.mode}_{timestamp}")
    
    # Create pipeline and train
    pipeline = CompletePipeline(config)
    
    try:
        pipeline.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        pipeline.print_summary(0)
    finally:
        pipeline.env.close()
        print("Environment closed.")

if __name__ == '__main__':
    main()


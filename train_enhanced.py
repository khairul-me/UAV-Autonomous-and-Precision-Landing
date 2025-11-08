"""
Enhanced training script leveraging the new multi-camera observation stack,
comprehensive logging, and monitoring utilities.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import airsim
import numpy as np
import torch
import torch.nn as nn

from environments.advanced_obstacles import AdvancedObstacleGenerator
from environments.observations import ObservationBuilder
from utils.episode_logger import EpisodeLogger, TrainingLogger
from utils.multi_camera import MultiCameraFeatureExtractor, MultiCameraManager
from utils.training_monitor import TrainingMonitor


# ---------------------------------------------------------------------- #
# Lightweight RL components for vector observations
# ---------------------------------------------------------------------- #
class EnhancedActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        self.action_scale = torch.tensor([3.0, 3.0, 2.0, 0.3], dtype=torch.float32)
        self.action_bias = torch.zeros(act_dim, dtype=torch.float32)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.net(obs))
        if self.action_scale.device != raw.device:
            self.action_scale = self.action_scale.to(raw.device)
            self.action_bias = self.action_bias.to(raw.device)
        return raw * self.action_scale + self.action_bias


class EnhancedCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class EnhancedReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = 100_000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.from_numpy(self.obs_buf[idxs]),
            acts=torch.from_numpy(self.acts_buf[idxs]),
            rews=torch.from_numpy(self.rews_buf[idxs]),
            next_obs=torch.from_numpy(self.next_obs_buf[idxs]),
            done=torch.from_numpy(self.done_buf[idxs]),
        )
        return batch

    def __len__(self):
        return self.size


class EnhancedTD3:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        policy_freq: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ):
        self.device = device
        self.actor = EnhancedActor(obs_dim, act_dim).to(device)
        self.actor_target = EnhancedActor(obs_dim, act_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = EnhancedCritic(obs_dim, act_dim).to(device)
        self.critic2 = EnhancedCritic(obs_dim, act_dim).to(device)
        self.critic1_target = EnhancedCritic(obs_dim, act_dim).to(device)
        self.critic2_target = EnhancedCritic(obs_dim, act_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, obs: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        if noise_scale > 0.0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action += noise
            action = np.clip(
                action,
                [-3.0, -3.0, -2.0, -0.3],
                [3.0, 3.0, 2.0, 0.3],
            )
        return action.astype(np.float32)

    def train(self, replay_buffer: EnhancedReplayBuffer, batch_size: int = 128):
        if len(replay_buffer) < batch_size:
            return {}

        self.total_it += 1
        batch = replay_buffer.sample(batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            noise = torch.randn_like(acts) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(
                torch.tensor([-3.0, -3.0, -2.0, -0.3], device=self.device),
                torch.tensor([3.0, 3.0, 2.0, 0.3], device=self.device),
            )
            target_q1 = self.critic1_target(next_obs, next_action)
            target_q2 = self.critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rews + (1 - done) * self.discount * target_q

        current_q1 = self.critic1(obs, acts)
        current_q2 = self.critic2(obs, acts)

        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(obs)
            actor_loss = -self.critic1(obs, actor_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ---------------------------------------------------------------------- #
# Enhanced trainer orchestrator
# ---------------------------------------------------------------------- #
class EnhancedTrainer:
    def __init__(self, args):
        self.args = args
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        print("[OK] AirSim connection established")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[OK] Using device: {self.device}")

        self._init_sensors()
        self._init_obstacles()
        self._init_models()
        self._init_logging()

        print("[OK] Enhanced trainer initialised")

    def _init_sensors(self):
        self.camera_manager = MultiCameraManager(self.client)
        self.feature_extractor = MultiCameraFeatureExtractor().to(self.device)
        self.obs_builder = ObservationBuilder(self.client, self.camera_manager, self.feature_extractor, self.device)

    def _init_obstacles(self):
        generator = AdvancedObstacleGenerator(self.client)
        obstacle_file = Path(f"obstacles_{self.args.scenario}.json")
        if obstacle_file.exists():
            self.obstacles = generator.load_obstacles(str(obstacle_file))
        else:
            self.obstacles = generator.generate_scenario(self.args.scenario, save_path=str(obstacle_file))

    def _init_models(self):
        self.obs_dim = 70
        self.act_dim = 4
        self.agent = EnhancedTD3(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
            discount=self.args.discount,
            tau=self.args.tau,
        )
        self.buffer = EnhancedReplayBuffer(self.obs_dim, self.act_dim, size=self.args.buffer_size)

    def _init_logging(self):
        self.episode_logger = EpisodeLogger(log_dir="logs/episodes")
        self.training_logger = TrainingLogger(log_dir="logs/training")
        self.monitor: Optional[TrainingMonitor] = None
        if self.args.monitor:
            self.monitor = TrainingMonitor()
            self.monitor.set_obstacles(self.obstacles)
            self.monitor.start()

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self):
        for episode in range(self.args.episodes):
            episode_stats = self._run_episode(episode)
            self.training_logger.log_episode(episode, episode_stats)
            if self.monitor:
                self.monitor.update_episode(episode, episode_stats)

            if episode % self.args.eval_interval == 0 and episode > 0:
                eval_results = self._evaluate()
                self.training_logger.log_evaluation(episode, eval_results)

            if episode % self.args.checkpoint_interval == 0:
                self._save_checkpoint(episode)

        if self.monitor:
            self.monitor.stop()
        print("[OK] Training complete")

    def _run_episode(self, episode: int) -> Dict[str, float]:
        goal = self._generate_goal()
        self.obs_builder.reset(goal)

        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-5, 2).join()

        self.episode_logger.start_episode(episode, {"goal": goal.tolist(), "scenario": self.args.scenario})

        total_reward = 0.0
        step = 0
        done = False

        obs_struct = self.obs_builder.build(current_time=0.0)
        obs_vector = obs_struct.to_vector()

        while not done and step < self.args.max_steps:
            action = self.agent.select_action(obs_vector, noise_scale=self.args.exploration_noise)
            self._execute_action(action)

            next_obs_struct = self.obs_builder.build(current_time=(step + 1) * self.args.control_dt)
            next_obs_vector = next_obs_struct.to_vector()

            reward, done = self._compute_reward(obs_struct, next_obs_struct)
            collision = self._check_collision()
            if collision:
                done = True

            self.buffer.add(obs_vector, action, reward, next_obs_vector, float(done))

            if len(self.buffer) > self.args.learning_starts:
                train_metrics = self.agent.train(self.buffer, batch_size=self.args.batch_size)
            else:
                train_metrics = {}

            self._log_step(step, obs_struct, action, reward, train_metrics)
            if self.monitor:
                self.monitor.update_step({"position": self._get_position().tolist(), "action": action.tolist()})

            obs_struct = next_obs_struct
            obs_vector = next_obs_vector
            total_reward += reward
            step += 1

        success = obs_struct.goal_distance < 5.0
        final_distance = obs_struct.goal_distance
        collision = self._check_collision()

        self.episode_logger.end_episode(success, collision, final_distance, total_reward)
        self.client.armDisarm(False)

        return {
            "reward": total_reward,
            "length": step,
            "success": success,
            "collision": collision,
            "goal_distance": final_distance,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _execute_action(self, action: np.ndarray):
        self.client.moveByVelocityAsync(
            float(action[0]),
            float(action[1]),
            float(action[2]),
            duration=self.args.control_dt,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(action[3])),
        ).join()

    def _compute_reward(self, obs: ObservationBuilder, next_obs: ObservationBuilder):
        progress = obs.goal_distance - next_obs.goal_distance
        reward = progress * 10.0
        done = False

        if next_obs.goal_distance < 5.0:
            reward += 10.0
            done = True

        if self._check_collision():
            reward -= 5.0
            done = True

        min_obstacle = min(next_obs.obstacle_distances.values()) if next_obs.obstacle_distances else 100.0
        if min_obstacle < 3.0:
            reward -= 0.5

        return reward, done

    def _check_collision(self) -> bool:
        collision = self.client.simGetCollisionInfo()
        return collision.has_collided

    def _generate_goal(self) -> np.ndarray:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = 65.0
        return np.array([distance * np.cos(angle), distance * np.sin(angle), -5.0], dtype=np.float32)

    def _get_position(self) -> np.ndarray:
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _log_step(self, step: int, obs: ObservationBuilder, action: np.ndarray, reward: float, metrics: Dict[str, float]):
        (position,) = [self._get_position()]
        self.episode_logger.log_step(
            step,
            {
                "position": position.tolist(),
                "velocity": obs.velocity.tolist(),
                "orientation": obs.orientation_euler.tolist(),
                "action": action.tolist(),
                "reward": reward,
                "goal_distance": obs.goal_distance,
                "closest_obstacle": min(obs.obstacle_distances.values()) if obs.obstacle_distances else 100.0,
                "sensor_health": obs.sensor_health,
                "q_values": metrics.get("critic1_loss"),
                "observation": obs.to_vector(),
            },
        )

    def _evaluate(self, episodes: int = 5):
        successes = 0
        total_reward = 0.0
        total_length = 0
        collisions = 0

        noise_backup = self.args.exploration_noise
        self.args.exploration_noise = 0.0

        for _ in range(episodes):
            stats = self._run_episode(-1)
            successes += int(stats["success"])
            total_reward += stats["reward"]
            total_length += stats["length"]
            collisions += int(stats["collision"])

        self.args.exploration_noise = noise_backup

        return {
            "success_rate": successes / episodes,
            "avg_reward": total_reward / episodes,
            "avg_length": total_length / episodes,
            "collision_rate": collisions / episodes,
        }

    def _save_checkpoint(self, episode: int):
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(
            {
                "episode": episode,
                "actor": self.agent.actor.state_dict(),
                "critic1": self.agent.critic1.state_dict(),
                "critic2": self.agent.critic2.state_dict(),
            },
            f"checkpoints/enhanced_{episode:06d}.pth",
        )
        print(f"[OK] Checkpoint saved for episode {episode}")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced multi-camera training pipeline")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--scenario", type=str, default="dprl", choices=["dprl", "corridor", "forest", "urban", "sparse", "dense"])
    parser.add_argument("--monitor", action="store_true", help="Enable real-time dashboard")

    parser.add_argument("--max-steps", dest="max_steps", type=int, default=500)
    parser.add_argument("--control-dt", dest="control_dt", type=float, default=0.2)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=100000)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", dest="exploration_noise", type=float, default=0.1)
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=1000)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", dest="checkpoint_interval", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = EnhancedTrainer(args)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[WARN] Training interrupted by user")
    finally:
        trainer.client.enableApiControl(False)
        trainer.client.armDisarm(False)


if __name__ == "__main__":
    main()


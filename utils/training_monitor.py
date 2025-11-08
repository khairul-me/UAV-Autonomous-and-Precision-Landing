"""
Real-time training monitor providing live plots of key metrics.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


class TrainingMonitor:
    """Threaded dashboard for visualising training progress."""

    def __init__(self, update_interval: float = 1.0, save_dir: str = "outputs/monitoring"):
        self.update_interval = update_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.data_queue: "queue.Queue" = queue.Queue()

        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.lengths: List[int] = []
        self.successes: List[int] = []
        self.collisions: List[int] = []
        self.q_values: List[float] = []
        self.losses: List[float] = []

        self.current_trajectory: List[List[float]] = []
        self.current_actions: List[List[float]] = []

        self.obstacles: Optional[List[Dict[str, float]]] = None

        self.fig: Optional[plt.Figure] = None
        self.axes: Dict[str, plt.Axes] = {}

        self.running = False
        self.thread: Optional[threading.Thread] = None

        print(f"[OK] TrainingMonitor initialised (update_interval={self.update_interval}s)")

    # ------------------------------------------------------------------ #
    # External API
    # ------------------------------------------------------------------ #
    def set_obstacles(self, obstacle_positions: List[Dict[str, float]]):
        self.obstacles = obstacle_positions

    def update_episode(self, episode: int, data: Dict[str, float]):
        self.data_queue.put(("episode", episode, data))

    def update_step(self, step_data: Dict[str, List[float]]):
        self.data_queue.put(("step", step_data))

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self.thread.start()
        print("[OK] Training monitor started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        plt.close("all")
        print("[OK] Training monitor stopped")

    def save_current_plot(self, filename: Optional[str] = None):
        if self.fig is None:
            return
        filename = filename or f"monitor_{int(time.time())}.png"
        save_path = self.save_dir / filename
        self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Dashboard snapshot saved to {save_path}")

    # ------------------------------------------------------------------ #
    # Internal processing
    # ------------------------------------------------------------------ #
    def _run_dashboard(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle("Training Monitor (Real-Time)", fontsize=16, fontweight="bold")
        grid = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        self.axes["reward"] = self.fig.add_subplot(grid[0, 0])
        self.axes["length"] = self.fig.add_subplot(grid[0, 1])
        self.axes["success"] = self.fig.add_subplot(grid[0, 2])
        self.axes["trajectory"] = self.fig.add_subplot(grid[1, :2])
        self.axes["actions"] = self.fig.add_subplot(grid[1, 2])
        self.axes["q_values"] = self.fig.add_subplot(grid[2, 0])
        self.axes["loss"] = self.fig.add_subplot(grid[2, 1])
        self.axes["metrics"] = self.fig.add_subplot(grid[2, 2])

        FuncAnimation(
            self.fig,
            self._update_plots,
            interval=int(self.update_interval * 1000),
            blit=False,
        )

        plt.show()

    def _update_plots(self, _frame):
        while not self.data_queue.empty():
            try:
                message = self.data_queue.get_nowait()
            except queue.Empty:
                break

            if message[0] == "episode":
                _, episode, data = message
                self._process_episode_data(episode, data)
            elif message[0] == "step":
                _, step_data = message
                self._process_step_data(step_data)

        self._plot_reward()
        self._plot_length()
        self._plot_success_rate()
        self._plot_trajectory()
        self._plot_actions()
        self._plot_q_values()
        self._plot_loss()
        self._plot_metrics()

        return []

    def _process_episode_data(self, episode: int, data: Dict[str, float]):
        self.episodes.append(episode)
        self.rewards.append(data.get("reward", 0.0))
        self.lengths.append(data.get("length", 0))
        self.successes.append(1 if data.get("success", False) else 0)
        self.collisions.append(1 if data.get("collision", False) else 0)

        if "q_value" in data:
            self.q_values.append(data["q_value"])
        if "loss" in data:
            self.losses.append(max(data["loss"], 1e-8))  # avoid log-scale issues

        self.current_trajectory = []
        self.current_actions = []

    def _process_step_data(self, step_data: Dict[str, List[float]]):
        if "position" in step_data:
            self.current_trajectory.append(step_data["position"])
        if "action" in step_data:
            self.current_actions.append(step_data["action"])

    # ------------------------------------------------------------------ #
    # Plot helpers
    # ------------------------------------------------------------------ #
    def _plot_reward(self):
        ax = self.axes["reward"]
        ax.clear()
        if self.rewards:
            ax.plot(self.episodes, self.rewards, "b-", alpha=0.3, linewidth=0.75)
            if len(self.rewards) > 10:
                window = min(50, len(self.rewards))
                rolling = np.convolve(self.rewards, np.ones(window) / window, mode="valid")
                ax.plot(self.episodes[window - 1 :], rolling, "r-", linewidth=2, label=f"Avg(last {window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_length(self):
        ax = self.axes["length"]
        ax.clear()
        if self.lengths:
            ax.plot(self.episodes, self.lengths, "g-", alpha=0.3, linewidth=0.75)
            if len(self.lengths) > 10:
                window = min(50, len(self.lengths))
                rolling = np.convolve(self.lengths, np.ones(window) / window, mode="valid")
                ax.plot(self.episodes[window - 1 :], rolling, "darkgreen", linewidth=2, label=f"Avg(last {window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Episode Length")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_success_rate(self):
        ax = self.axes["success"]
        ax.clear()
        if self.successes:
            window = min(100, len(self.successes))
            rates = []
            for idx in range(len(self.successes)):
                start = max(0, idx - window + 1)
                rates.append(np.mean(self.successes[start : idx + 1]) * 100)
            ax.plot(self.episodes, rates, "purple", linewidth=2)
            ax.axhline(80, color="r", linestyle="--", alpha=0.5, label="Target 80%")
            ax.fill_between(self.episodes, 0, rates, alpha=0.3, color="purple")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Success Rate (window {min(100, len(self.successes))})")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_trajectory(self):
        ax = self.axes["trajectory"]
        ax.clear()
        if self.obstacles:
            for obs in self.obstacles:
                circle = plt.Circle((obs["x"], obs["y"]), obs["radius"], color="red", alpha=0.2)
                ax.add_patch(circle)

        if len(self.current_trajectory) > 1:
            trajectory = np.array(self.current_trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="Path")
            ax.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], "r^", label="Current")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Current Episode Trajectory")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_actions(self):
        ax = self.axes["actions"]
        ax.clear()
        if self.current_actions:
            actions = np.array(self.current_actions)
            steps = np.arange(actions.shape[0])
            labels = ["vx", "vy", "vz", "yaw_rate"]
            for idx in range(actions.shape[1]):
                ax.plot(steps, actions[:, idx], linewidth=1.5, label=labels[idx])
        ax.set_xlabel("Step")
        ax.set_ylabel("Action value")
        ax.set_title("Actions (current episode)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_q_values(self):
        ax = self.axes["q_values"]
        ax.clear()
        if self.q_values:
            episodes = self.episodes[-len(self.q_values) :]
            ax.plot(episodes, self.q_values, "orange", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Q-value")
        ax.set_title("Q-values")
        ax.grid(True, alpha=0.3)

    def _plot_loss(self):
        ax = self.axes["loss"]
        ax.clear()
        if self.losses:
            episodes = self.episodes[-len(self.losses) :]
            ax.plot(episodes, self.losses, "brown", linewidth=2)
            ax.set_yscale("log")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        ax.set_title("Loss (log scale)")
        ax.grid(True, alpha=0.3)

    def _plot_metrics(self):
        ax = self.axes["metrics"]
        ax.clear()
        ax.axis("off")

        if not self.episodes:
            return

        window = min(100, len(self.rewards))
        recent_rewards = self.rewards[-window:] if self.rewards else [0.0]
        recent_success = self.successes[-window:] if self.successes else [0]
        recent_collisions = self.collisions[-window:] if self.collisions else [0]

        metrics = f"""
        Metrics (Episode {self.episodes[-1]})
        -------------------------------
        Avg reward (last {window}): {np.mean(recent_rewards):.2f}
        Success rate (last {window}): {np.mean(recent_success) * 100:.1f}%
        Collision rate (last {window}): {np.mean(recent_collisions) * 100:.1f}%
        Total episodes: {len(self.episodes)}
        Best reward: {np.max(self.rewards):.2f}
        Worst reward: {np.min(self.rewards):.2f}
        """
        ax.text(0.05, 0.5, metrics, fontsize=10, family="monospace", va="center")


# ---------------------------------------------------------------------- #
# Diagnostic script
# ---------------------------------------------------------------------- #
def test_monitor():
    print("Testing TrainingMonitor...")
    monitor = TrainingMonitor(update_interval=0.5)

    # Simulated obstacles
    obstacles = [{"x": i * 5.0, "y": j * 5.0, "radius": 2.5} for i in range(-3, 4) for j in range(-3, 4)]
    monitor.set_obstacles(obstacles)
    monitor.start()

    try:
        for episode in range(10):
            for step in range(30):
                monitor.update_step(
                    {
                        "position": [
                            step * 0.5 + np.random.randn() * 0.1,
                            step * 0.3 + np.random.randn() * 0.1,
                            -5.0,
                        ],
                        "action": (np.random.randn(4) * 0.5).tolist(),
                    }
                )
                time.sleep(0.02)

            monitor.update_episode(
                episode,
                {
                    "reward": np.random.randn() * 2 + 5 + episode * 0.1,
                    "length": 30,
                    "success": bool(np.random.rand() > 0.3),
                    "collision": bool(np.random.rand() > 0.7),
                    "q_value": np.random.randn() + 10 + episode * 0.2,
                    "loss": max(np.exp(-episode * 0.05), 1e-4),
                },
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[WARN] Monitor test interrupted")
    finally:
        monitor.stop()
        monitor.save_current_plot("monitor_test.png")
        print("[OK] TrainingMonitor test completed")


if __name__ == "__main__":
    import numpy as np

    test_monitor()


# defenses/temporal_consistency.py
import numpy as np
import torch
from collections import deque


class TemporalConsistencyChecker:
    """
    Temporal Consistency Defense
    Detects adversarial attacks by finding sudden unrealistic changes
    
    Insight: Adversarial perturbations often cause sudden jumps in:
    1. Predicted actions
    2. Estimated Q-values  
    3. Perceived obstacle distances
    
    Natural sensor noise is smooth, but attacks cause discontinuities!
    """
    
    def __init__(self, 
                 window_size=10,
                 action_change_threshold=1.0,
                 q_value_change_threshold=5.0,
                 depth_change_threshold=10.0):
        """
        Args:
            window_size: Number of recent observations to track
            action_change_threshold: Max allowed action change between steps
            q_value_change_threshold: Max allowed Q-value change
            depth_change_threshold: Max allowed average depth change (meters)
        """
        self.window_size = window_size
        self.action_threshold = action_change_threshold
        self.q_threshold = q_value_change_threshold
        self.depth_threshold = depth_change_threshold
        
        # History buffers (using deque for efficient FIFO)
        self.action_history = deque(maxlen=window_size)
        self.q_value_history = deque(maxlen=window_size)
        self.depth_history = deque(maxlen=window_size)
        self.detection_history = deque(maxlen=window_size)
    
    def reset(self):
        """Reset all history buffers"""
        self.action_history.clear()
        self.q_value_history.clear()
        self.depth_history.clear()
        self.detection_history.clear()
    
    def check_consistency(self, current_action, current_q_value, current_depth):
        """
        Check temporal consistency of current observation
        
        Args:
            current_action: Action predicted by policy [4]
            current_q_value: Q-value estimate (float)
            current_depth: Depth image [H, W]
        
        Returns:
            {
                'is_consistent': bool,
                'anomalies': list of detected anomaly types,
                'confidence': confidence in attack detection (0-1),
                'should_reject': whether to reject current observation
            }
        """
        anomalies = []
        confidence = 0.0
        
        # Need at least 2 observations for comparison
        if len(self.action_history) < 1:
            self.action_history.append(current_action)
            self.q_value_history.append(current_q_value)
            self.depth_history.append(self._compute_depth_stats(current_depth))
            self.detection_history.append(False)
            
            return {
                'is_consistent': True,
                'anomalies': [],
                'confidence': 0.0,
                'should_reject': False
            }
        
        # Check 1: Action consistency
        action_consistent, action_change = self._check_action_consistency(current_action)
        if not action_consistent:
            anomalies.append('action_jump')
            confidence += 0.4
        
        # Check 2: Q-value consistency
        q_consistent, q_change = self._check_q_value_consistency(current_q_value)
        if not q_consistent:
            anomalies.append('q_value_jump')
            confidence += 0.3
        
        # Check 3: Depth consistency
        depth_consistent, depth_change = self._check_depth_consistency(current_depth)
        if not depth_consistent:
            anomalies.append('depth_jump')
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        
        # Update history
        self.action_history.append(current_action)
        self.q_value_history.append(current_q_value)
        self.depth_history.append(self._compute_depth_stats(current_depth))
        
        is_consistent = len(anomalies) == 0
        self.detection_history.append(not is_consistent)
        
        # Decide whether to reject observation
        # Reject if: (1) Multiple anomalies OR (2) Recent pattern of attacks
        should_reject = (
            len(anomalies) >= 2 or
            (not is_consistent and sum(self.detection_history) >= 3)
        )
        
        if not is_consistent:
            print(f"[WARNING] Temporal inconsistency detected!")
            print(f"  Anomalies: {anomalies}")
            print(f"  Confidence: {confidence:.2f}")
            if should_reject:
                print(f"  [CRITICAL] REJECTING observation!")
        
        return {
            'is_consistent': is_consistent,
            'anomalies': anomalies,
            'confidence': confidence,
            'should_reject': should_reject,
            'details': {
                'action_change': action_change,
                'q_change': q_change,
                'depth_change': depth_change
            }
        }
    
    def _check_action_consistency(self, current_action):
        """Check if action change is reasonable"""
        prev_action = self.action_history[-1]
        
        # Compute L2 distance between actions
        action_change = np.linalg.norm(current_action - prev_action)
        
        is_consistent = action_change < self.action_threshold
        
        return is_consistent, action_change
    
    def _check_q_value_consistency(self, current_q_value):
        """Check if Q-value change is reasonable"""
        prev_q_value = self.q_value_history[-1]
        
        q_change = abs(current_q_value - prev_q_value)
        
        is_consistent = q_change < self.q_threshold
        
        return is_consistent, q_change
    
    def _check_depth_consistency(self, current_depth):
        """Check if depth image change is reasonable"""
        current_stats = self._compute_depth_stats(current_depth)
        prev_stats = self.depth_history[-1]
        
        # Compare mean depth
        mean_change = abs(current_stats['mean'] - prev_stats['mean'])
        
        # Compare minimum depth (nearest obstacle)
        min_change = abs(current_stats['min'] - prev_stats['min'])
        
        # Significant change in either indicates possible attack
        depth_change = max(mean_change, min_change)
        
        is_consistent = depth_change < self.depth_threshold
        
        return is_consistent, depth_change
    
    def _compute_depth_stats(self, depth_image):
        """Compute statistical summary of depth image"""
        return {
            'mean': np.mean(depth_image),
            'std': np.std(depth_image),
            'min': np.min(depth_image),
            'max': np.max(depth_image)
        }
    
    def get_smoothed_action(self, current_action):
        """
        Get temporally smoothed action
        Uses exponential moving average of recent actions
        
        Args:
            current_action: Current predicted action [4]
        
        Returns:
            Smoothed action [4]
        """
        if len(self.action_history) == 0:
            return current_action
        
        # Exponential moving average
        alpha = 0.3  # Weight for current action
        
        # Compute weighted average of recent actions
        smoothed = alpha * current_action
        
        for i, past_action in enumerate(reversed(self.action_history)):
            weight = (1 - alpha) * (0.7 ** i)  # Exponential decay
            smoothed += weight * past_action
        
        return smoothed
    
    def get_attack_frequency(self):
        """
        Get frequency of detected attacks in recent history
        
        Returns:
            Proportion of recent observations flagged as attacks
        """
        if len(self.detection_history) == 0:
            return 0.0
        
        return sum(self.detection_history) / len(self.detection_history)


# Test
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("Testing Temporal Consistency Defense...")
    print("="*60)
    
    # Create checker
    checker = TemporalConsistencyChecker(
        window_size=10,
        action_change_threshold=1.0,
        q_value_change_threshold=5.0,
        depth_change_threshold=5.0
    )
    
    # Simulate normal flight sequence
    print("\n1. Testing normal flight (no attack)...")
    
    normal_actions = []
    normal_q_values = []
    detections = []
    
    for i in range(20):
        # Smooth trajectory
        action = np.array([1.0, 0.1*np.sin(i*0.5), 0.0, 0.05*np.cos(i*0.5)])
        q_value = 10.0 + np.random.randn() * 0.5  # Small noise
        depth = np.random.rand(80, 100) * 50 + 20  # Depth around 20-70m
        
        result = checker.check_consistency(action, q_value, depth)
        
        normal_actions.append(action)
        normal_q_values.append(q_value)
        detections.append(not result['is_consistent'])
    
    print(f"  Attack detection rate: {sum(detections)/len(detections)*100:.1f}%")
    print(f"  (Should be ~0% for normal flight)")
    
    # Reset and simulate attacked sequence
    print("\n2. Testing with adversarial attack...")
    checker.reset()
    
    attacked_actions = []
    attacked_q_values = []
    detections = []
    
    for i in range(20):
        if i == 10:
            # Inject attack at timestep 10!
            print(f"  [ATTACK] Injecting attack at step {i}")
            action = np.array([3.0, 2.5, -1.0, 0.3])  # Sudden extreme action
            q_value = -20.0  # Sudden drop
            depth = np.random.rand(80, 100) * 100  # Completely different depths
        else:
            # Normal behavior
            action = np.array([1.0, 0.1*np.sin(i*0.5), 0.0, 0.05*np.cos(i*0.5)])
            q_value = 10.0 + np.random.randn() * 0.5
            depth = np.random.rand(80, 100) * 50 + 20
        
        result = checker.check_consistency(action, q_value, depth)
        
        attacked_actions.append(action)
        attacked_q_values.append(q_value)
        detections.append(not result['is_consistent'])
        
        if not result['is_consistent']:
            print(f"    Step {i}: Anomalies detected: {result['anomalies']}")
    
    print(f"\n  Attack detection rate: {sum(detections)/len(detections)*100:.1f}%")
    print(f"  Attack frequency in window: {checker.get_attack_frequency()*100:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot actions (x-velocity component)
    normal_vx = [a[0] for a in normal_actions]
    attacked_vx = [a[0] for a in attacked_actions]
    
    axes[0].plot(normal_vx, 'b-', label='Normal', linewidth=2)
    axes[0].plot(attacked_vx, 'r-', label='With Attack', linewidth=2)
    axes[0].axvline(x=10, color='k', linestyle='--', label='Attack Injected')
    axes[0].set_ylabel('X Velocity (m/s)')
    axes[0].set_title('Action Consistency Check')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Q-values
    axes[1].plot(normal_q_values, 'b-', label='Normal', linewidth=2)
    axes[1].plot(attacked_q_values, 'r-', label='With Attack', linewidth=2)
    axes[1].axvline(x=10, color='k', linestyle='--')
    axes[1].set_ylabel('Q-Value')
    axes[1].set_title('Q-Value Consistency Check')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot detections
    axes[2].scatter(range(len(detections)), detections, c=['red' if d else 'green' for d in detections], s=100)
    axes[2].axvline(x=10, color='k', linestyle='--')
    axes[2].set_ylabel('Attack Detected')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Detection Results')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_consistency_test.png', dpi=150)
    print("\n[OK] Saved visualization to temporal_consistency_test.png")
    
    print("\n" + "="*60)
    print("[OK] Temporal Consistency Defense Test Complete!")


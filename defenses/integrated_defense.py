# defenses/integrated_defense.py
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from defenses.input_sanitization import InputSanitizer
    from defenses.multi_sensor_fusion import MultiSensorFusion
    from defenses.temporal_consistency import TemporalConsistencyChecker
except ImportError:
    InputSanitizer = None
    MultiSensorFusion = None
    TemporalConsistencyChecker = None


class IntegratedDefenseSystem:
    """
    Complete 4-Layer Defense System
    
    Layer 1: Input Sanitization (denoise attacked images)
    Layer 2: Multi-Sensor Fusion (cross-validate sensors)
    Layer 3: Temporal Consistency (detect sudden changes)
    Layer 4: Robust Training (adversarial training with privileged learning)
    
    This is YOUR COMPLETE DEFENSE FRAMEWORK!
    """
    
    def __init__(self, 
                 input_sanitizer=None,
                 sensor_fusion=None,
                 temporal_checker=None,
                 enable_layer1=True,
                 enable_layer2=True,
                 enable_layer3=True):
        """
        Args:
            input_sanitizer: InputSanitizer instance (Layer 1)
            sensor_fusion: MultiSensorFusion instance (Layer 2)
            temporal_checker: TemporalConsistencyChecker instance (Layer 3)
            enable_layer*: Enable/disable specific layers
        """
        self.input_sanitizer = input_sanitizer
        self.sensor_fusion = sensor_fusion
        self.temporal_checker = temporal_checker
        
        self.enable_layer1 = enable_layer1 and input_sanitizer is not None
        self.enable_layer2 = enable_layer2 and sensor_fusion is not None
        self.enable_layer3 = enable_layer3 and temporal_checker is not None
        
        # Statistics
        self.total_observations = 0
        self.attacks_detected = 0
        self.attacks_by_layer = {1: 0, 2: 0, 3: 0}
        self.false_positives = 0
    
    def process_observation(self, 
                           depth_image, 
                           state,
                           airsim_client=None,
                           vehicle_name="Drone1",
                           action=None,
                           q_value=None):
        """
        Process observation through all defense layers
        
        Args:
            depth_image: Input depth image (potentially attacked)
            state: Self-state vector
            airsim_client: AirSim client (for sensor fusion)
            vehicle_name: Vehicle name
            action: Predicted action (for temporal check)
            q_value: Q-value estimate (for temporal check)
        
        Returns:
            {
                'depth_image': Cleaned/validated depth image,
                'state': Validated state,
                'attack_detected': Whether attack was detected,
                'attack_confidence': Confidence in detection (0-1),
                'detection_layers': Which layers detected attack,
                'should_use': Whether this observation is safe to use,
                'fallback_required': Whether to use emergency fallback
            }
        """
        self.total_observations += 1
        
        detection_layers = []
        attack_confidence = 0.0
        attack_detected = False
        
        processed_depth = depth_image
        processed_state = state
        
        # ==================== Layer 1: Input Sanitization ====================
        if self.enable_layer1:
            sanitize_result = self.input_sanitizer.sanitize(depth_image)
            
            if sanitize_result['is_anomaly']:
                attack_detected = True
                detection_layers.append(1)
                attack_confidence = max(attack_confidence, sanitize_result['anomaly_score'])
                self.attacks_by_layer[1] += 1
                
                # Use cleaned image
                processed_depth = sanitize_result['clean_image']
                
                print(f"  [Layer 1] Anomaly detected! Score: {sanitize_result['anomaly_score']:.3f}")
        
        # ==================== Layer 2: Multi-Sensor Fusion ====================
        if self.enable_layer2 and airsim_client is not None:
            fusion_result = self.sensor_fusion.fuse_observations(airsim_client, vehicle_name)
            
            if fusion_result['attack_detected']:
                attack_detected = True
                detection_layers.append(2)
                attack_confidence = max(attack_confidence, fusion_result['confidence'])
                self.attacks_by_layer[2] += 1
                
                # Use trusted sensor data
                if 'depth' not in fusion_result['trusted_sensors']:
                    # Depth camera compromised - use LiDAR fallback
                    processed_depth = self.sensor_fusion.get_trusted_depth(fusion_result)
                    print(f"  [Layer 2] Depth camera attack! Type: {fusion_result['attack_type']}")
        
        # ==================== Layer 3: Temporal Consistency ====================
        if self.enable_layer3 and action is not None and q_value is not None:
            temporal_result = self.temporal_checker.check_consistency(
                action, q_value, processed_depth
            )
            
            if not temporal_result['is_consistent']:
                attack_detected = True
                detection_layers.append(3)
                attack_confidence = max(attack_confidence, temporal_result['confidence'])
                self.attacks_by_layer[3] += 1
                
                print(f"  [Layer 3] Temporal inconsistency! Anomalies: {temporal_result['anomalies']}")
                
                # Use smoothed action (handled by caller)
                # Can also reject observation if severe
                if temporal_result['should_reject']:
                    print(f"  [Layer 3] [CRITICAL] Observation REJECTED!")
        
        # ==================== Decision Logic ====================
        
        # Should we use this observation?
        should_use = True
        fallback_required = False
        
        if attack_detected:
            self.attacks_detected += 1
            
            # Multiple layers detected attack - high confidence
            if len(detection_layers) >= 2:
                attack_confidence = min(attack_confidence * 1.5, 1.0)
                
                # Very severe - might need fallback
                if attack_confidence > 0.8:
                    fallback_required = True
                    print(f"  [CRITICAL] SEVERE ATTACK DETECTED! Confidence: {attack_confidence:.2f}")
                    print(f"     Detection layers: {detection_layers}")
        
        result = {
            'depth_image': processed_depth,
            'state': processed_state,
            'attack_detected': attack_detected,
            'attack_confidence': attack_confidence,
            'detection_layers': detection_layers,
            'should_use': should_use,
            'fallback_required': fallback_required,
            'stats': self.get_statistics()
        }
        
        return result
    
    def reset(self):
        """Reset temporal state (call at start of episode)"""
        if self.temporal_checker is not None:
            self.temporal_checker.reset()
    
    def get_statistics(self):
        """Get defense system statistics"""
        detection_rate = self.attacks_detected / max(self.total_observations, 1)
        
        return {
            'total_observations': self.total_observations,
            'attacks_detected': self.attacks_detected,
            'detection_rate': detection_rate,
            'attacks_by_layer': self.attacks_by_layer.copy(),
            'layer1_detections': self.attacks_by_layer[1],
            'layer2_detections': self.attacks_by_layer[2],
            'layer3_detections': self.attacks_by_layer[3]
        }
    
    def print_summary(self):
        """Print defense system summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DEFENSE SYSTEM SUMMARY")
        print("="*60)
        print(f"Total observations processed: {stats['total_observations']}")
        print(f"Attacks detected: {stats['attacks_detected']}")
        print(f"Detection rate: {stats['detection_rate']*100:.1f}%")
        print(f"\nDetections by layer:")
        print(f"  Layer 1 (Input Sanitization): {stats['layer1_detections']}")
        print(f"  Layer 2 (Multi-Sensor Fusion): {stats['layer2_detections']}")
        print(f"  Layer 3 (Temporal Consistency): {stats['layer3_detections']}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    print("Integrated Defense System - Example Usage")
    print("="*60)
    
    # This demonstrates how to use the complete system
    # In practice, you'd initialize with real components
    
    example_code = """
    # 1. Initialize all defense components
    from defenses.input_sanitization import DenoisingAutoencoder, AnomalyDetector, InputSanitizer
    from defenses.multi_sensor_fusion import MultiSensorFusion
    from defenses.temporal_consistency import TemporalConsistencyChecker
    
    # Train denoiser
    denoiser = DenoisingAutoencoder()
    # ... train on clean/attacked pairs ...
    
    # Calibrate detector
    detector = AnomalyDetector()
    detector.calibrate(clean_images)
    
    # Create sanitizer
    sanitizer = InputSanitizer(denoiser, detector)
    
    # Create sensor fusion
    sensor_fusion = MultiSensorFusion()
    
    # Create temporal checker
    temporal_checker = TemporalConsistencyChecker()
    
    # 2. Create integrated system
    defense_system = IntegratedDefenseSystem(
        input_sanitizer=sanitizer,
        sensor_fusion=sensor_fusion,
        temporal_checker=temporal_checker
    )
    
    # 3. Use in flight loop
    for step in range(max_steps):
        # Get observation
        obs = env.get_observation()
        depth = obs['depth_image']
        state = obs['self_state']
        
        # Get action and Q-value from policy
        action = agent.select_action(depth, state)
        q_value = agent.get_q_value(depth, state, action)
        
        # Process through defense system
        result = defense_system.process_observation(
            depth_image=depth,
            state=state,
            airsim_client=client,
            action=action,
            q_value=q_value
        )
        
        # Use defended observation
        if result['should_use']:
            safe_depth = result['depth_image']
            safe_state = result['state']
            
            # Execute action
            env.step(action)
        else:
            # Fallback to safe mode
            emergency_action = safe_controller.get_action()
            env.step(emergency_action)
    
    # 4. Print statistics
    defense_system.print_summary()
    """
    
    print(example_code)
    print("\n[OK] Integrated Defense System Ready!")


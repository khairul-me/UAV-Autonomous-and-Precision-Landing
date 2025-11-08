# Complete Training Pipeline - Usage Guide

## Overview

This guide explains how to use the complete training pipeline for adversarially robust drone navigation.

## Files

- `train_complete.py` - Main training script with all modes
- `quick_test.py` - Quick verification script
- `train_all.py` - Batch training for all models

## Quick Start

### 1. Quick Test (Verify Everything Works)

```bash
python quick_test.py
```

This will:
- Test environment creation
- Test agent initialization
- Test replay buffer
- Test attacks
- Test training step
- Run a full episode

### 2. Train Baseline (Clean Training)

```bash
python train_complete.py --mode baseline --max-episodes 500
```

### 3. Train Your Complete Robust Method

```bash
python train_complete.py --mode robust \
    --enable-all-defenses \
    --max-episodes 1000 \
    --adversarial-ratio 0.3 \
    --attack-probability 0.5
```

## Training Modes

### Mode 1: Baseline
Clean training without attacks or defenses.

```bash
python train_complete.py --mode baseline --max-episodes 500
```

### Mode 2: Baseline + Attacks
Baseline trained on clean, tested with attacks (demonstrates vulnerability).

```bash
python train_complete.py --mode baseline_attacked --max-episodes 500
```

### Mode 3: DPRL-style
Privileged learning for sensor noise robustness.

```bash
python train_complete.py --mode dprl \
    --add-sensor-noise \
    --max-episodes 500
```

### Mode 4: Robust (YOUR METHOD)
Complete adversarial robustness with all 4 defense layers.

```bash
python train_complete.py --mode robust \
    --enable-all-defenses \
    --max-episodes 1000
```

## Defense Layers

You can enable individual defense layers:

```bash
# Only Layer 1 (Input Sanitization)
python train_complete.py --mode robust \
    --enable-layer1 \
    --max-episodes 1000

# Layer 1 + Layer 3 (Input Sanitization + Temporal Consistency)
python train_complete.py --mode robust \
    --enable-layer1 \
    --enable-layer3 \
    --max-episodes 1000

# All layers
python train_complete.py --mode robust \
    --enable-all-defenses \
    --max-episodes 1000
```

## Advanced Options

### Training Parameters

```bash
python train_complete.py --mode robust \
    --max-episodes 1000 \
    --max-steps-per-episode 500 \
    --learning-starts 2000 \
    --batch-size 128 \
    --buffer-size 50000
```

### RL Hyperparameters

```bash
python train_complete.py --mode robust \
    --discount 0.99 \
    --tau 0.005 \
    --actor-lr 3e-4 \
    --critic-lr 3e-4 \
    --exploration-noise 0.1
```

### Adversarial Training

```bash
python train_complete.py --mode robust \
    --adversarial-ratio 0.3 \
    --attack-probability 0.5
```

### Logging and Saving

```bash
python train_complete.py --mode robust \
    --save-dir ./experiments \
    --log-interval 10 \
    --eval-interval 50
```

### Device Selection

```bash
# Use GPU (default)
python train_complete.py --mode robust --device cuda

# Use CPU
python train_complete.py --mode robust --device cpu
```

## Batch Training

Train all 4 models sequentially:

```bash
python train_all.py
```

This will:
1. Train baseline (500 episodes)
2. Train baseline + attacks (500 episodes)
3. Train DPRL-style (500 episodes)
4. Train robust method (1000 episodes)

**Estimated time: 8-12 hours**

## Output Structure

Each training run creates a directory structure:

```
experiments/
└── {mode}_{timestamp}/
    ├── checkpoints/
    │   ├── {mode}_ep0_r{reward}.pth
    │   ├── {mode}_ep50_r{reward}.pth
    │   └── ...
    ├── recordings/
    │   └── episode_*.npy
    ├── figures/
    │   └── training_curves.png
    └── logs/
        └── training.log
```

## Examples

### Example 1: Quick Test
```bash
python quick_test.py
```

### Example 2: Train Baseline
```bash
python train_complete.py --mode baseline --max-episodes 500
```

### Example 3: Train Robust Method
```bash
python train_complete.py --mode robust \
    --enable-all-defenses \
    --max-episodes 1000 \
    --adversarial-ratio 0.3 \
    --attack-probability 0.5
```

### Example 4: Train with Selected Defenses
```bash
python train_complete.py --mode robust \
    --enable-layer1 \
    --enable-layer3 \
    --max-episodes 1000
```

### Example 5: DPRL-style Training
```bash
python train_complete.py --mode dprl \
    --add-sensor-noise \
    --max-episodes 500
```

### Example 6: Train All Models
```bash
python train_all.py
```

### Example 7: Resume from Checkpoint
```bash
python train_complete.py --mode robust \
    --enable-all-defenses \
    --max-episodes 1000 \
    --load-checkpoint ./experiments/robust_20250101_120000/checkpoints/robust_ep500_r25.3.pth
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure all dependencies are installed:
```bash
pip install airsim torch numpy opencv-python matplotlib
```

### Issue: "AirSim connection failed"
**Solution**: Make sure AirSim is running and the environment is loaded.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU:
```bash
python train_complete.py --mode robust --batch-size 64 --device cpu
```

### Issue: "Training is too slow"
**Solution**: 
- Reduce `--max-episodes`
- Reduce `--max-steps-per-episode`
- Disable some defense layers
- Use GPU if available

## Next Steps

After training:

1. **Compare Results**: Use evaluation scripts to compare different models
2. **Generate Figures**: Create publication-ready figures
3. **Analyze Performance**: Review training curves and metrics
4. **Test on New Scenarios**: Evaluate on unseen environments

## Notes

- Training can take several hours depending on hardware
- Make sure AirSim is running before starting training
- Checkpoints are saved automatically for recovery
- Evaluation runs every `--eval-interval` episodes
- All results are saved in the experiment directory


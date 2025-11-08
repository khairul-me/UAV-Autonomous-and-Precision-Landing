# Pre-Training System Verification Guide

## Overview

Before starting training, we need to verify that all systems are working correctly. This guide walks you through:

1. **System Verification** - Check all components
2. **Demo Flight** - Run a 1-2 minute test flight with video recording
3. **Review Results** - Verify everything is working

## Quick Start

### Option 1: Automated (Recommended)

```powershell
.\startup_check.ps1
```

This will:
- Check AirSim connection
- Run all system checks
- Optionally run demo flight
- Confirm readiness for training

### Option 2: Manual Step-by-Step

#### Step 1: System Verification

```powershell
python preflight_check.py
```

This checks:
- ✓ AirSim Connection
- ✓ Drone Control
- ✓ Camera System
- ✓ Depth Perception
- ✓ Sensor Suite (IMU, GPS, Barometer, LiDAR)
- ✓ Obstacle Configuration
- ✓ Action Execution
- ✓ State Observation

**Expected Output:**
```
================================================================================
SYSTEM VERIFICATION
================================================================================
Checking all systems...

[AirSim Connection]
  [OK] Connected to AirSim successfully

[Drone Control]
  [OK] Drone armed and ready for API control

[Camera System]
  [OK] Camera working: 320x240

[Depth Perception]
  [OK] Depth camera OK: range [0.50, 85.23]m

...

================================================================================
VERIFICATION SUMMARY
================================================================================
[OK] Passed: 8/8
[FAIL] Failed: 0/8

[OK] ALL SYSTEMS OPERATIONAL
Ready to proceed with demo and training.
```

#### Step 2: Demo Flight

```powershell
python demo_flight.py
```

This will:
- Take off to 5m altitude
- Fly for 2 minutes with obstacle avoidance
- Record video with telemetry overlay
- Save flight data and visualizations
- Land safely

**Features:**
- Live visualization window (press 'Q' to stop early)
- Real-time telemetry display
- Trajectory visualization
- Video recording (`demo_flight.mp4`)
- Telemetry data (`telemetry.npy`)
- Analysis plots (`telemetry_plots.png`)

**Output Directory:**
```
demo_output_YYYYMMDD_HHMMSS/
├── demo_flight.mp4          # Recorded video
├── telemetry.npy            # Flight data
└── telemetry_plots.png      # Visualization
```

#### Step 3: Review Results

After the demo flight, check:

1. **Video Recording** (`demo_flight.mp4`)
   - Verify video was recorded
   - Check telemetry overlay is visible
   - Verify smooth flight behavior

2. **Telemetry Plots** (`telemetry_plots.png`)
   - Check trajectory makes sense
   - Verify speed is reasonable
   - Check obstacle detection works
   - Verify no crashes

3. **Console Output**
   - Check for any errors
   - Verify all systems reported OK
   - Confirm flight completed successfully

## Troubleshooting

### Issue: "Cannot connect to AirSim"

**Solution:**
1. Make sure Unreal Engine is running with AirSim plugin
2. Make sure the environment is loaded (Blocks or Neighborhood)
3. Check that AirSim API is enabled in settings

### Issue: "Drone failed to arm"

**Solution:**
1. Make sure you're using the correct vehicle type (Multirotor)
2. Check AirSim settings for API control permissions
3. Try resetting the simulation

### Issue: "Camera not working"

**Solution:**
1. Check camera settings in `settings.json`
2. Verify camera is enabled for the vehicle
3. Check camera name matches code ("0" or "front_center")

### Issue: "Video not recording"

**Solution:**
1. Check if OpenCV video writer is working
2. Verify codec is available (mp4v)
3. Check disk space for output directory

### Issue: "Demo flight crashes"

**Solution:**
1. Check AirSim connection is stable
2. Verify environment has obstacles configured
3. Check for any errors in console output
4. Try reducing flight duration for testing

## Expected Demo Flight Behavior

During the 2-minute demo flight, you should see:

1. **Takeoff**: Drone rises to 5m altitude
2. **Navigation**: Drone moves toward goal (50, 50, -5)
3. **Obstacle Avoidance**: When obstacle detected (< 5m), drone slows and adjusts
4. **Circular Pattern**: When goal reached, drone flies in circular pattern
5. **Landing**: Drone lands safely

**Telemetry Display:**
- Position (X, Y, Z)
- Velocity (Vx, Vy, Vz)
- Speed (m/s)
- Nearest obstacle distance
- Distance to goal
- Flight status (CLEAR/OBSTACLE NEAR)

## Next Steps

After successful verification:

1. **Quick Test**: Run `python quick_test.py` to verify training pipeline
2. **Start Training**: 
   - Baseline: `python train_complete.py --mode baseline --max-episodes 100`
   - Robust: `python train_complete.py --mode robust --enable-all-defenses --max-episodes 1000`

## Files Created

- `preflight_check.py` - System verification script
- `demo_flight.py` - Demo flight with recording
- `startup_check.ps1` - Automated startup script (Windows)
- `PRE_TRAINING_VERIFICATION.md` - This guide

## Notes

- Demo flight duration: 2 minutes (120 seconds)
- Video recording: 20 FPS, 1280x720 resolution
- Telemetry sampling: Every 0.05 seconds
- Press 'Q' during flight to stop early
- All outputs saved in timestamped directory

## Safety

- Always verify system checks pass before flight
- Monitor console output during demo
- Check video recording after completion
- Verify safe landing before proceeding


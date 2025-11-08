# AirSim Not Running - Start AirSim First

## Issue
The preflight check shows that AirSim is not running. You need to start AirSim before running the verification.

## How to Start AirSim

### Option 1: Use Existing Launch Scripts

If you have launch scripts, run:

```powershell
# Launch Blocks environment
.\launch_blocks.ps1

# OR launch Blocks with safe mode
.\launch_blocks_safe.ps1

# OR launch drone-only mode
.\launch_drone_only.ps1
```

### Option 2: Manual Launch

1. **Open Unreal Engine**
   - Launch Unreal Engine 4.27 or 5.0
   - Make sure AirSim plugin is installed and enabled

2. **Load Environment**
   - Open the Blocks environment (or your preferred environment)
   - Wait for the environment to fully load
   - You should see the drone in the scene

3. **Verify AirSim is Running**
   - The AirSim API server should start automatically
   - You should see "AirSim is running" in the console
   - The drone should be visible in the scene

### Option 3: Check if AirSim is Already Running

Try to verify the connection:

```powershell
python -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('AirSim is running!')"
```

## After AirSim is Running

Once AirSim is running, you can proceed:

```powershell
# Run preflight check
python preflight_check.py

# If all checks pass, run demo flight
python demo_flight.py
```

## Troubleshooting

### Issue: "WSAECONNREFUSED" Error
- **Cause**: AirSim API server is not running
- **Solution**: Start Unreal Engine with AirSim environment

### Issue: "Retry connection over the limit"
- **Cause**: AirSim is not responding on port 41451
- **Solution**: 
  1. Check if Unreal Engine is running
  2. Check if environment is loaded
  3. Try restarting Unreal Engine

### Issue: AirSim Plugin Not Found
- **Cause**: AirSim plugin is not installed or enabled
- **Solution**: Install AirSim plugin for Unreal Engine

## Quick Checklist

- [ ] Unreal Engine is running
- [ ] AirSim plugin is enabled
- [ ] Environment (Blocks/Neighborhood) is loaded
- [ ] Drone is visible in the scene
- [ ] AirSim API server is running (check console)

## Next Steps

1. Start AirSim using one of the methods above
2. Wait for environment to fully load
3. Run `python preflight_check.py` again
4. If all checks pass, run `python demo_flight.py`


# Current Status - Refreshed

## What's Running Right Now

**Last Check:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

### Processes:
- Check Task Manager or run: `Get-Process | Where-Object { $_.ProcessName -like "*Block*" -or $_.ProcessName -like "*AirSim*" }`

### Port Status:
- Port 41451 (AirSim API): Check if listening
- Command: `Get-NetTCPConnection -LocalPort 41451 -State Listen`

## Quick Status Check Commands

```powershell
# Check what's running
Get-Process | Where-Object { $_.ProcessName -like "*Block*" -or $_.ProcessName -like "*AirSim*" }

# Check API port
Get-NetTCPConnection -LocalPort 41451 -State Listen

# Check settings
Get-Content "$env:USERPROFILE\Documents\AirSim\settings.json"
```

## Files Created

All installation files and scripts are in: `E:\Drone\`

See `COMPLETE_INSTALLATION_SUMMARY.md` for full details.


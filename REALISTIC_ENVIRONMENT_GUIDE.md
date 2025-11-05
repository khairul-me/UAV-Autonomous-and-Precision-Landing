# 🏙️ REALISTIC ENVIRONMENT SETUP GUIDE

## Overview

This setup uses **AirSimNH (Neighborhood)** - a realistic urban environment with:
- Houses and buildings
- Streets and roads
- Trees and vegetation
- Realistic lighting and shadows
- Much more realistic than simple Blocks environment

---

## 🚀 Quick Start

### Step 1: Launch Realistic Environment

```powershell
.\launch_realistic_env.ps1
```

This will:
- ✅ Configure settings for realistic environment
- ✅ Launch AirSimNH (Neighborhood)
- ✅ Set up proper drone configuration

**Wait 3-5 minutes** for AirSimNH to fully load.

### Step 2: Run Autonomous Flight

```powershell
.\run_realistic_flight.bat
```

Or directly:
```powershell
venv\Scripts\python.exe autonomous_flight_realistic.py
```

---

## 🎮 Features

### ✅ Autonomous Flight
- **Spiral Exploration Pattern**: Gradually expands exploration area
- **Safe Altitude**: 15m altitude (safe for urban environment)
- **Intelligent Waypoints**: Visits multiple locations automatically
- **Smooth Movement**: 3.0 m/s speed for realistic flight

### ✅ Recording & Data
- **Video Recording**: MP4 video with flight overlay
- **Flight Data**: JSON file with positions and timestamps
- **Summary Report**: Text file with flight statistics
- **Frame Capture**: Every 0.3 seconds during flight

### ✅ Safety Features
- **Emergency Exit**: Press `Ctrl+C` at any time
- **Emergency Landing**: Automatic safe landing on exit
- **Return to Start**: Returns to origin before landing
- **Error Handling**: Comprehensive try-catch blocks

---

## 🛑 EXIT FUNCTIONALITY

### How to Exit Safely

**During Flight:**
- Press **`Ctrl+C`** at any time
- The drone will:
  1. Stop current waypoint
  2. Perform emergency landing
  3. Clean up and disarm
  4. Save all recorded data

**During Setup:**
- Press **`Ctrl+C`** to cancel before flight starts

**After Flight:**
- Press **Enter** to exit the program

---

## 📋 Flight Parameters

### Default Settings:
- **Duration**: 120 seconds (2 minutes)
- **Altitude**: 15 meters
- **Speed**: 3.0 m/s
- **Waypoint Interval**: 8 seconds
- **Frame Capture**: Every 0.3 seconds

### Customization:
When you run the script, you can specify:
- **Duration**: 30-600 seconds (default: 120)
- Longer flights = more exploration = larger spiral

---

## 📁 Output Files

All recordings saved to:
```
flight_recordings/realistic_[timestamp]/
├── flight_recording.mp4    (Video with overlay)
├── flight_data.json        (Position/timestamp data)
└── summary.txt            (Flight statistics)
```

---

## 🔧 Troubleshooting

### "Cannot connect to AirSim"
- **Solution**: Make sure AirSimNH.exe is running
- **Wait**: 2-5 minutes after launching AirSimNH
- **Check**: Look for "Asset database ready!" in console

### "Takeoff failed"
- **Solution**: Check if drone is visible in environment
- **Check**: Verify settings.json has correct drone configuration

### "Exit not working"
- **Solution**: Press `Ctrl+C` directly in the console window
- **Make sure**: Console window has focus

---

## 📊 Flight Pattern

The drone follows a **spiral exploration pattern**:
1. Starts at origin (0, 0)
2. Gradually increases exploration radius
3. Visits waypoints in spiral pattern
4. Maximum radius: 50 meters
5. Returns to start before landing

**Example Waypoints:**
- Waypoint 1: (2m, 0m) - Small radius
- Waypoint 2: (4m, 2m) - Expanding
- Waypoint 3: (6m, 6m) - Further out
- ...continues spiraling outward...

---

## ✅ Verification Checklist

- [x] `launch_realistic_env.ps1` - Created and tested
- [x] `autonomous_flight_realistic.py` - Syntax verified
- [x] `run_realistic_flight.bat` - Created
- [x] Exit functionality - Ctrl+C implemented
- [x] Emergency landing - Implemented
- [x] Error handling - Comprehensive
- [x] Recording functionality - Working
- [x] Settings configuration - Automated

---

## 🎯 Next Steps

1. **Launch Environment**: Run `launch_realistic_env.ps1`
2. **Wait for Load**: 3-5 minutes
3. **Start Flight**: Run `run_realistic_flight.bat`
4. **Monitor Flight**: Watch the console for waypoint updates
5. **Exit Anytime**: Press `Ctrl+C` to exit safely
6. **Review Data**: Check `flight_recordings/` folder

---

**Everything is ready! Enjoy realistic autonomous drone flights! 🚁🏙️**

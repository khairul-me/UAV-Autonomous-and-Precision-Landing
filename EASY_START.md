# 🚁 Easy Start Guide - Keyboard Control for Drone

## ⚠️ IMPORTANT: Ensure Drone Mode (NO CARS!)

Before starting, make sure AirSim is set to **DRONE MODE ONLY**:

### Step 0: Fix Drone Mode (Do This First!)
```powershell
cd E:\Drone
.\fix_drone_mode.ps1
```

This will:
- Kill any running AirSim processes
- Ensure settings are set to Multirotor (NO CARS)
- Launch AirSim with drone-only mode

**Wait 2-5 minutes** for AirSim to fully load. You should see a **DRONE (quadcopter)**, NOT a car!

## Step 1: Close AirSim (if it's running)
- Close the Blocks.exe window completely

## Step 2: Start AirSim
Run this command:
```powershell
cd E:\Drone
.\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe
```
**Wait 2-5 minutes** for it to fully load. You'll see the Blocks environment window with a **DRONE**.

## Step 3: Run Keyboard Control

### Option A: Easy Way (Recommended - No PowerShell issues!)
Just double-click this file:
```
run_keyboard_control.bat
```

### Option B: Command Line
Open PowerShell and run:
```powershell
cd E:\Drone
.\venv\Scripts\python.exe keyboard_control.py
```

**Note:** If you get a PowerShell execution policy error with `Activate.ps1`, use Option A or Option B above instead!

## Step 4: Fly!
- Press **[C]** to claim control (enables API + arms drone)
- Press **[T]** to take off
- Use **[WASD]** to move
- Press **[L]** to land
- Press **[ESC]** to quit

---

## Quick Commands (Copy & Paste)

**Fix Drone Mode (Do this first!):**
```powershell
cd E:\Drone; .\fix_drone_mode.ps1
```

**Launch AirSim:**
```powershell
cd E:\Drone; .\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe
```

**Start Keyboard Control:**
- **Easiest:** Double-click `run_keyboard_control.bat`
- **Or PowerShell:** `cd E:\Drone; .\venv\Scripts\python.exe keyboard_control.py`

---

## Controls Summary

**[C]** = Claim Control (do this first!)  
**[T]** = Take Off  
**[W]** = Forward  
**[S]** = Backward  
**[A]** = Left  
**[D]** = Right  
**[R]** = Up  
**[F]** = Down  
**[H]** = Hover  
**[L]** = Land  
**[ESC]** = Quit  

---

## Troubleshooting

**If you see CARS instead of DRONES:**
1. Close AirSim completely
2. Run: `.\fix_drone_mode.ps1`
3. Wait for AirSim to restart with drone mode
4. You should see a quadcopter drone, NOT a car!

**If you get PowerShell execution policy errors:**
- Use `run_keyboard_control.bat` instead (just double-click it)
- Or use: `.\venv\Scripts\python.exe keyboard_control.py` directly

**That's it!** 🎉

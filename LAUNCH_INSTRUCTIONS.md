# 🚀 Launch Instructions

## Current Status

Blocks.exe has been launched! However, it needs time to fully initialize.

## ⏱️ Initialization Time

**First Launch:** 2-5 minutes
- Shader compilation
- Texture loading  
- AirSim plugin initialization

## ✅ How to Know Blocks is Ready

1. **Visual Check:** The Blocks window should be fully visible with the 3D environment
2. **No Loading Screens:** All loading screens should be gone
3. **Window Title:** Should show "Blocks" (not "Loading...")

## 🧪 Testing Connection

### Option 1: Wait and Test (Recommended)

1. **Wait 2-3 minutes** for Blocks to fully load
2. **Look at the Blocks window** - it should show the 3D environment
3. **Then run the test:**

```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python test_airsim.py
```

### Option 2: Use Test Script

After Blocks is fully loaded:

```powershell
.\test_connection.ps1
```

## 🔍 Troubleshooting Connection Errors

If you get "Connection refused" or "WSAECONNREFUSED":

### Solution 1: Wait Longer
- First launch takes time
- Wait 2-5 minutes total
- Ensure Blocks window is fully visible

### Solution 2: Check Blocks Window
- Is the window visible?
- Is it still loading/showing splash screen?
- If crashed, restart: `.\launch_blocks.ps1`

### Solution 3: Manual Launch
If Blocks didn't start automatically:

```powershell
cd E:\Drone\AirSim\Blocks\WindowsNoEditor
.\Blocks.exe
```

Wait for it to load, then test connection.

## 📊 Expected Test Output

When everything works:

```
============================================================
AirSim Installation Test
============================================================

[1/5] Connecting to AirSim...
[OK] Connection established successfully!

[2/5] Checking API version...
[OK] AirSim API Version: 1.8.1

[3/5] Testing camera access...
[OK] Camera available: front_center

[4/5] Capturing test image...
[OK] Saved scene image: test_output/test_scene.png
[OK] Saved depth image: test_output/test_depth.png
[OK] Saved segmentation image: test_output/test_segmentation.png

[5/5] Testing vehicle state access...
[OK] Vehicle state retrieved

============================================================
[SUCCESS] ALL TESTS PASSED!
============================================================
```

## 🎯 Quick Commands

**Check if Blocks is running:**
```powershell
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue
```

**Launch Blocks:**
```powershell
.\launch_blocks.ps1
```

**Test Connection:**
```powershell
.\test_connection.ps1
```

---

**Current Status:** Blocks launched, waiting for initialization...


# Complete AirSim Installation Summary - Step by Step

## What Was Done - Clear Explanation for Anyone

This document explains everything that was installed and configured, step by step, with exact file paths.

---

## PART 1: Initial Setup and Prerequisites

### Step 1: Python Installation
**What was done:**
- Installed Python 3.11.9 using Windows Package Manager (winget)
- **Location:** `C:\Users\Khairul\AppData\Local\Programs\Python\Python311\`
- **Verification:** Python is now available system-wide

**Command used:**
```powershell
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
```

---

### Step 2: Directory Structure Creation
**What was done:**
Created the following directory structure for the project:

```
E:\Drone\
├── AirSim\                    # Main AirSim directory
│   ├── Blocks\                # Blocks environment (for drones)
│   │   └── WindowsNoEditor\
│   │       └── Blocks\
│   │           └── Binaries\Win64\
│   │               └── Blocks.exe  (160 MB - DRONE environment)
│   └── AirSimNH\              # AirSimNH environment (for cars)
│       └── AirSimNH\
│           └── WindowsNoEditor\
│               └── AirSimNH\
│                   └── Binaries\Win64\
│                       └── AirSimNH.exe  (161 MB - CAR environment)
├── venv\                      # Python virtual environment
├── test_output\                # Test images output folder
└── [various scripts and config files]
```

**Files created:**
- `E:\Drone\AirSim\` - Created
- `E:\Drone\AirSim\Blocks\` - Created
- `E:\Drone\AirSim\AirSimNH\` - Created

---

### Step 3: Python Virtual Environment Setup
**What was done:**
- Created a Python virtual environment to isolate project dependencies
- **Location:** `E:\Drone\venv\`
- **Activation:** `E:\Drone\venv\Scripts\Activate.ps1`

**What this means:**
- All Python packages are installed in this isolated environment
- Prevents conflicts with other Python projects
- Makes it easy to manage dependencies

---

### Step 4: Python Package Installation
**What was done:**
Installed all required Python packages in the virtual environment.

**Location:** `E:\Drone\venv\Lib\site-packages\`

**Packages installed:**
1. **AirSim Python API** (`airsim`)
   - Version: 1.8.1
   - Purpose: Connect to and control AirSim simulator
   - Location: `E:\Drone\venv\Lib\site-packages\airsim\`

2. **PyTorch** (`torch`)
   - Version: 2.7.1+cu118 (with CUDA support)
   - Purpose: Deep learning framework
   - Location: `E:\Drone\venv\Lib\site-packages\torch\`

3. **OpenCV** (`opencv-python`, `opencv-contrib-python`)
   - Version: 4.12.0.88
   - Purpose: Image processing
   - Location: `E:\Drone\venv\Lib\site-packages\cv2\`

4. **NumPy** (`numpy`)
   - Version: 2.2.6
   - Purpose: Numerical computations
   - Location: `E:\Drone\venv\Lib\site-packages\numpy\`

5. **Other packages:**
   - `pandas` - Data analysis
   - `matplotlib` - Plotting
   - `msgpack-rpc-python` - Communication protocol for AirSim
   - All dependencies listed in `requirements.txt`

**Command used:**
```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## PART 2: AirSim Environment Downloads

### Step 5: Downloading Blocks Environment (For Drones)
**What was done:**
- Downloaded Blocks.zip from GitHub releases
- **Source:** https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/Blocks.zip
- **Size:** 247 MB
- **Download location:** `C:\Users\Khairul\Downloads\Blocks.zip`
- **Extracted to:** `E:\Drone\AirSim\Blocks\`

**What Blocks is:**
- A 3D environment with colored blocks
- **Designed for DRONE simulation** (Multirotor mode)
- Includes Unreal Engine runtime
- **Main executable:** `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe`

---

### Step 6: Downloading AirSimNH Environment (For Cars)
**What was done:**
- Downloaded AirSimNH.zip from GitHub releases
- **Source:** https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/AirSimNH.zip
- **Size:** 1.6 GB
- **Download location:** `C:\Users\Khairul\Downloads\AirSimNH.zip`
- **Extracted to:** `E:\Drone\AirSim\AirSimNH\`

**What AirSimNH is:**
- A neighborhood environment with roads and buildings
- **Designed for CAR simulation** (Car mode)
- **Main executable:** `E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64\AirSimNH.exe`

**Important:** This is the CAR environment. We don't want this for drones!

---

## PART 3: Configuration Files

### Step 7: Creating AirSim Settings File
**What was done:**
Created settings.json file to configure AirSim for drone mode.

**Location:** `C:\Users\Khairul\Documents\AirSim\settings.json`

**Content:**
```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",        ← This sets it to DRONE mode
  "ClockSpeed": 1,
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",  ← This is the drone type
      "X": 0,
      "Y": 0,
      "Z": 0,
      "Yaw": 0
    }
  },
  "ApiServerPort": 41451          ← Port for Python API connection
}
```

**What this does:**
- Tells AirSim to use Multirotor (drone) mode
- Sets up a drone vehicle named "Drone1"
- Configures the API server port

**Also created in:**
- `E:\Drone\AirSim\Blocks\WindowsNoEditor\settings.json`
- `E:\Drone\settings.json`

(These are backup locations, but the main one is in Documents)

---

## PART 4: Fixing Issues

### Step 8: Fixing Missing DirectX DLL
**Problem encountered:**
- When trying to launch Blocks, got error: "XAPOFX1_5.dll was not found"
- This is a DirectX component needed for Unreal Engine

**What was done:**
- Installed DirectX End-User Runtime using winget
- **Command:** `winget install Microsoft.DirectX --silent`
- **Result:** DLL now available at `C:\WINDOWS\System32\XAPOFX1_5.dll`

**After this fix:**
- Blocks.exe was able to launch successfully
- No more DLL errors

---

### Step 9: Fixing AirSim Plugin in Blocks
**Problem encountered:**
- Blocks.zip from GitHub doesn't include the AirSim plugin DLLs
- This means Blocks can show the environment visually, but Python can't connect to control it
- Port 41451 (AirSim API server) was not listening

**What was done:**
- Copied AirSim plugin from AirSimNH to Blocks
- **Source:** `E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Plugins\AirSim\`
- **Destination:** `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\`

**What this does:**
- Gives Blocks the AirSim plugin DLLs
- Enables the API server on port 41451
- Allows Python to connect and control the drone

---

## PART 5: Scripts Created

### Step 10: Test Scripts
**What was created:**

1. **`E:\Drone\test_airsim.py`**
   - Tests basic AirSim connection
   - Captures test images (scene, depth, segmentation)
   - Saves to `E:\Drone\test_output\`

2. **`E:\Drone\test_drone.py`**
   - Tests drone-specific connection
   - Uses MultirotorClient (for drones)

3. **`E:\Drone\test_drone_simple.py`**
   - Simplified drone connection test

4. **`E:\Drone\FLY_DRONE_NOW.py`**
   - Complete flight script with retry logic
   - Takes off, flies, lands

5. **`E:\Drone\MAKE_IT_FLY.py`**
   - Complete flight sequence
   - Makes drone move forward, right, return, land

6. **`E:\Drone\fly_drone.py`**
   - Another flight test script

7. **`E:\Drone\diagnose_drone.py`**
   - Diagnostic script to check connection issues

---

### Step 11: PowerShell Scripts
**What was created:**

1. **`E:\Drone\install_airsim.ps1`**
   - Automated installation script
   - Checks Python, creates venv, installs packages

2. **`E:\Drone\setup_airsim.ps1`**
   - Original setup script
   - Sets up Python environment

3. **`E:\Drone\launch_blocks.ps1`**
   - Launches Blocks environment

4. **`E:\Drone\launch_drone.ps1`**
   - Launches Blocks for drone simulation

5. **`E:\Drone\quick_start.ps1`**
   - Quick verification script

6. **`E:\Drone\download_airsim.ps1`**
   - Helper script for downloading AirSim binaries

---

## PART 6: Current Status

### What's Working:
✅ **Python 3.11.9** - Installed and working
✅ **Virtual Environment** - Created at `E:\Drone\venv\`
✅ **All Python Packages** - Installed (AirSim, PyTorch, OpenCV, etc.)
✅ **Blocks Environment** - Downloaded and extracted
✅ **Settings Configuration** - Set to Multirotor (drone) mode
✅ **DirectX** - Fixed and installed
✅ **AirSim Plugin** - Copied to Blocks from AirSimNH

### Current Issue:
⚠️ **API Connection** - Python can't connect to Blocks API server
- Port 41451 is not listening
- Blocks shows drone visually but API server isn't running
- This prevents Python from controlling the drone

### Why This Happens:
Blocks.zip from GitHub doesn't include the compiled AirSim plugin DLLs. Even though we copied them from AirSimNH, Blocks may need to be restarted or the plugin needs to initialize properly.

---

## PART 7: File Locations Summary

### Important Files:

**Python Virtual Environment:**
- Location: `E:\Drone\venv\`
- Python: `E:\Drone\venv\Scripts\python.exe`
- Packages: `E:\Drone\venv\Lib\site-packages\`

**AirSim Settings:**
- Main: `C:\Users\Khairul\Documents\AirSim\settings.json`
- Backup: `E:\Drone\AirSim\Blocks\WindowsNoEditor\settings.json`
- Backup: `E:\Drone\settings.json`

**Blocks Environment (DRONE):**
- Executable: `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe`
- Plugin: `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\`
- Config: `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Config\`

**AirSimNH Environment (CAR - NOT USED FOR DRONES):**
- Executable: `E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64\AirSimNH.exe`
- Plugin: `E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Plugins\AirSim\`

**Test Scripts:**
- `E:\Drone\test_airsim.py`
- `E:\Drone\test_drone.py`
- `E:\Drone\MAKE_IT_FLY.py`
- `E:\Drone\FLY_DRONE_NOW.py`

---

## PART 8: What Each Environment Does

### Blocks Environment (What We Want for Drones):
- **Location:** `E:\Drone\AirSim\Blocks\`
- **Executable:** `Blocks.exe` (160 MB)
- **Visual:** 3D environment with colored blocks
- **Default Mode:** Supports Multirotor (drone) mode
- **Purpose:** UAV/drone simulation
- **Status:** Shows drone on screen, but API connection needs fixing

### AirSimNH Environment (What We DON'T Want - It's for Cars):
- **Location:** `E:\Drone\AirSim\AirSimNH\`
- **Executable:** `AirSimNH.exe` (161 MB)
- **Visual:** Neighborhood with roads and buildings
- **Default Mode:** Car mode
- **Purpose:** Car/vehicle simulation
- **Status:** This is why cars appear - we accidentally launched this instead of Blocks

---

## PART 9: The Problem

### Why Cars Keep Appearing:

**Root Cause:**
1. AirSimNH is configured for **car mode** by default
2. Blocks is configured for **drone mode** by default
3. When we launch AirSimNH, we get cars
4. When we launch Blocks, we should get drones

**What Happened:**
- Sometimes AirSimNH was launched instead of Blocks
- AirSimNH always shows cars (it's designed for that)
- Even with settings.json set to Multirotor, AirSimNH environment is car-based

**Solution:**
- **ONLY launch Blocks.exe** (not AirSimNH.exe)
- Blocks will show drones
- Settings.json will work with Blocks

---

## PART 10: Next Steps to Make Drone Fly

### What Needs to Happen:

1. **Ensure Blocks is running** (not AirSimNH)
   - Check: Look for `Blocks.exe` in Task Manager
   - Path should be: `E:\Drone\AirSim\Blocks\...\Blocks.exe`

2. **Wait for Blocks to fully initialize**
   - Takes 2-3 minutes
   - AirSim plugin needs to start API server on port 41451

3. **Verify API server is running**
   - Port 41451 should be listening
   - Check with: `Get-NetTCPConnection -LocalPort 41451`

4. **Run flight script**
   - Script: `E:\Drone\MAKE_IT_FLY.py`
   - This will connect, take off, fly, and land

---

## Summary

**What was installed:**
1. Python 3.11.9
2. Virtual environment with all packages
3. Blocks environment (drone)
4. AirSimNH environment (car - not used for drones)
5. Settings configured for Multirotor mode
6. DirectX runtime
7. AirSim plugin copied to Blocks

**Current status:**
- Blocks shows drone on screen ✅
- Python packages installed ✅
- Settings configured correctly ✅
- API connection not working ⚠️ (needs Blocks to initialize API server)

**To make it work:**
- Launch Blocks (not AirSimNH)
- Wait for full initialization
- Run `python MAKE_IT_FLY.py`

---

This is everything that was done, step by step, with file paths.


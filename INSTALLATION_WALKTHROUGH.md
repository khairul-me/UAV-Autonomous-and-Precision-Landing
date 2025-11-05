# AirSim Installation Walkthrough - Step-by-Step Guide

This guide walks you through installing Microsoft AirSim on Windows using **pre-built binaries** and the **Blocks test environment**, which is the fastest way to get started.

## Prerequisites Check

Before starting, verify you have:

- ✅ **Windows 10/11 (64-bit)**
- ✅ **Python 3.8+** installed
- ✅ **NVIDIA GPU drivers** up to date (for RTX 3060)
- ✅ **~5GB free space** for AirSim binaries
- ✅ **Internet connection** for downloads

---

## Installation Steps

### Step 1: Download Pre-built AirSim Binaries (Recommended)

**Why pre-built?** Building from source takes 30-60 minutes and requires Visual Studio. Pre-built binaries get you running in minutes.

1. **Visit AirSim Releases Page:**
   - Go to: https://github.com/microsoft/AirSim/releases
   - Look for the **latest stable release** (e.g., v1.8.1 or newer)

2. **Download Required Files:**
   - Download **`AirSim.zip`** (or `AirSim-Windows.zip`) - This contains the AirSim plugin and binaries
   - Download **`Blocks.zip`** - This is the pre-built Blocks test environment

3. **Extract to Project Directory:**
   ```powershell
   # If files are in Downloads, extract them:
   # Extract AirSim.zip to: E:\Drone\AirSim
   # Extract Blocks.zip to: E:\Drone\AirSim\Blocks
   ```

**Alternative Quick Method:**
```powershell
# Navigate to your project
cd E:\Drone

# Create directories
New-Item -ItemType Directory -Force -Path "AirSim"
New-Item -ItemType Directory -Force -Path "AirSim\Blocks"

# Manually extract downloaded ZIPs to these locations
```

---

### Step 2: Setup Python Environment

1. **Create Virtual Environment:**
   ```powershell
   cd E:\Drone
   python -m venv venv
   ```

2. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Note:** If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install Python Dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install AirSim Python API:**
   ```powershell
   # If using pre-built binaries, install from PyPI
   pip install msgpack-rpc-python airsim
   
   # OR if you have AirSim source with PythonClient folder:
   cd E:\Drone\AirSim\PythonClient
   pip install -e .
   ```

---

### Step 3: Launch Blocks Test Environment

1. **Navigate to Blocks Directory:**
   ```powershell
   cd E:\Drone\AirSim\Blocks
   ```

2. **Launch Blocks Environment:**
   - **Option A:** Double-click `Blocks.exe` (or `run.bat` if available)
   - **Option B:** Run from PowerShell:
     ```powershell
     .\Blocks.exe
     ```

3. **First Launch Notes:**
   - First launch may take **2-5 minutes** (compiling shaders)
   - A window should open showing the Blocks environment
   - **Keep this window open** while testing/using AirSim
   - You should see a simple 3D environment with colored blocks

---

### Step 4: Verify Installation

1. **Open a NEW PowerShell window** (keep Blocks.exe running)

2. **Navigate to project and activate venv:**
   ```powershell
   cd E:\Drone
   .\venv\Scripts\Activate.ps1
   ```

3. **Run Test Script:**
   ```powershell
   python test_airsim.py
   ```

4. **Expected Output:**
   ```
   ============================================================
   AirSim Installation Test
   ============================================================
   
   [1/5] Connecting to AirSim...
   ✓ Connection established successfully!
   
   [2/5] Checking API version...
   ✓ AirSim API Version: 1.x.x
   
   [3/5] Testing camera access...
   ✓ Camera available: front_center
   
   [4/5] Capturing test image...
   ✓ Saved scene image: test_output/test_scene.png
   ✓ Saved depth image: test_output/test_depth.png
   ✓ Saved segmentation image: test_output/test_segmentation.png
   
   [5/5] Testing vehicle state access...
   ✓ Vehicle state retrieved
   
   ============================================================
   ✓ ALL TESTS PASSED!
   ============================================================
   ```

5. **Check Test Output:**
   - Navigate to `E:\Drone\test_output\`
   - You should see 3 images:
     - `test_scene.png` - RGB camera view
     - `test_depth.png` - Depth visualization
     - `test_segmentation.png` - Semantic segmentation

---

### Step 5: Verify GPU/CUDA (Optional but Recommended)

1. **Check CUDA Installation:**
   ```powershell
   nvcc --version
   ```
   Should show CUDA version (11.x or 12.x)

2. **Verify PyTorch CUDA:**
   ```powershell
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
   ```
   
   Expected output:
   ```
   CUDA available: True
   Device: NVIDIA GeForce RTX 3060
   ```

---

## Troubleshooting Common Issues

### Issue 1: "Connection Refused" or Timeout

**Symptoms:** `test_airsim.py` fails to connect

**Solutions:**
1. ✅ Ensure `Blocks.exe` is running (check Task Manager)
2. ✅ Wait 30-60 seconds after launching Blocks.exe before running test
3. ✅ Check Windows Firewall - allow Blocks.exe through firewall
4. ✅ Try restarting Blocks.exe

### Issue 2: "ModuleNotFoundError: No module named 'airsim'"

**Symptoms:** Python can't find airsim module

**Solutions:**
1. ✅ Ensure virtual environment is activated: `.\venv\Scripts\Activate.ps1`
2. ✅ Reinstall: `pip install msgpack-rpc-python airsim`
3. ✅ Verify: `python -c "import airsim; print(airsim.__version__)"`

### Issue 3: Blocks.exe Won't Launch

**Symptoms:** Crashes or fails to start

**Solutions:**
1. ✅ Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. ✅ Update GPU drivers (NVIDIA GeForce Experience)
3. ✅ Add Windows Defender exclusions for `E:\Drone\AirSim`
4. ✅ Check Windows Event Viewer for error details

### Issue 4: "DLL missing" Errors

**Symptoms:** MSVCP140.dll, VCRUNTIME140.dll missing

**Solutions:**
1. ✅ Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. ✅ Restart computer after installation

### Issue 5: CUDA Not Available in PyTorch

**Symptoms:** `torch.cuda.is_available()` returns False

**Solutions:**
1. ✅ Verify CUDA is installed: `nvcc --version`
2. ✅ Reinstall PyTorch with CUDA:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace `cu118` with your CUDA version: `cu121` for CUDA 12.1, etc.)

---

## Quick Verification Checklist

After installation, verify these:

- [ ] Blocks.exe launches successfully
- [ ] Python test script connects to AirSim
- [ ] Test images are generated (scene, depth, segmentation)
- [ ] Python dependencies installed (airsim, torch, opencv, numpy)
- [ ] CUDA available for PyTorch (if GPU acceleration needed)

---

## Next Steps After Installation

### 1. Explore AirSim Examples
```powershell
# Check if AirSim examples folder exists
cd E:\Drone\AirSim\PythonClient\examples
# Run example scripts to learn AirSim API
```

### 2. Basic Flight Test
Create a simple flight script:
```python
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Move forward
client.moveToPositionAsync(5, 0, -5, 5).join()

# Land
client.landAsync().join()
```

### 3. Start Data Collection Pipeline
- Review `INSTALLATION_GUIDE.md` for advanced setup
- Plan procedural generation strategy
- Design domain randomization parameters

---

## File Structure After Installation

```
E:\Drone\
├── AirSim\
│   ├── Blocks\              # Blocks test environment (Blocks.exe)
│   ├── PythonClient\        # Python API (if source installed)
│   └── Unreal\              # AirSim plugin (if building from source)
├── venv\                    # Python virtual environment
├── test_output\              # Generated test images
├── requirements.txt
├── test_airsim.py
└── INSTALLATION_WALKTHROUGH.md (this file)
```

---

## Alternative: Building from Source (Advanced)

If you need latest features or want to customize AirSim:

1. **Install Unreal Engine 4.27:**
   - Download Epic Games Launcher
   - Install UE 4.27 (requires Epic Games account)

2. **Build AirSim:**
   ```powershell
   cd E:\Drone
   git clone https://github.com/microsoft/AirSim.git
   cd AirSim
   git submodule update --init --recursive
   .\build.cmd
   ```
   
   **Time:** 30-60 minutes

3. **See `INSTALLATION_GUIDE.md` for detailed source build instructions**

---

## Getting Help

- **AirSim Documentation:** https://microsoft.github.io/AirSim/
- **AirSim GitHub:** https://github.com/microsoft/AirSim
- **AirSim Issues:** https://github.com/microsoft/AirSim/issues
- **Unreal Engine Docs:** https://docs.unrealengine.com/4.27/en-US/

---

## Summary

You've now completed the basic AirSim installation using pre-built binaries! 

**What you have:**
- ✅ AirSim Blocks environment running
- ✅ Python API configured
- ✅ Basic functionality verified

**Ready for:**
- 🚁 Testing UAV flight control
- 📸 Capturing synthetic images
- 🎯 Beginning data collection for your research

**Next phase:** Implement procedural generation and domain randomization for your 100K+ image dataset!


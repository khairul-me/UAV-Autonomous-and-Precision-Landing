# Microsoft AirSim Installation Guide for Windows

## System Requirements
- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) - CUDA compatible
- **RAM**: 128GB (more than sufficient)
- **Python**: 3.8 or higher

---

## Step 1: Install Prerequisites

### 1.1 Install Visual Studio Build Tools (Required for AirSim)
1. Download **Visual Studio 2019 or 2022 Community** (free)
   - URL: https://visualstudio.microsoft.com/downloads/
2. During installation, select:
   - **Desktop development with C++** workload
   - **Windows 10/11 SDK** (latest version)
   - **CMake tools for Windows**

### 1.2 Install Git
1. Download from: https://git-scm.com/download/win
2. Install with default options

### 1.3 Install CUDA Toolkit (for GPU acceleration)
1. Download **CUDA 11.x or 12.x** (compatible with RTX 3060)
   - URL: https://developer.nvidia.com/cuda-downloads
2. Select Windows → x86_64 → 10/11 → exe (local)
3. Install with default options

---

## Step 2: Install Unreal Engine 4.27

### Option A: Install via Epic Games Launcher (Recommended)
1. Download Epic Games Launcher: https://www.unrealengine.com/download
2. Sign in/create account
3. Go to **Unreal Engine** → **Library** → **+ Add Version**
4. Select **4.27** → **Install**
5. Choose installation location (recommend: `C:\Program Files\Epic Games\UE_4.27`)
   - **Note**: Requires ~25GB disk space

### Option B: Source Build (For advanced users)
If you need to modify Unreal Engine itself:
```powershell
# Clone Unreal Engine (requires Epic Games account linked to GitHub)
git clone -b 4.27 https://github.com/EpicGames/UnrealEngine.git
# Follow Epic's build instructions
```

---

## Step 3: Install AirSim

### Option A: Pre-built Binaries (Easiest - Recommended for Start)
1. Download pre-built AirSim release:
   - Go to: https://github.com/microsoft/AirSim/releases
   - Download latest stable release (e.g., `AirSim.zip`)
   - Extract to: `E:\Drone\AirSim`

2. Download Blocks Environment:
   - From same releases page, download `Blocks.zip`
   - Extract to: `E:\Drone\AirSim\Blocks`

### Option B: Build from Source (If you need latest features)
```powershell
# Clone AirSim repository
cd E:\Drone
git clone https://github.com/microsoft/AirSim.git
cd AirSim

# Update submodules
git submodule update --init --recursive

# Build AirSim (this will take 30-60 minutes)
.\build.cmd
```

---

## Step 4: Setup Python Environment

### 4.1 Create Virtual Environment
```powershell
cd E:\Drone
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4.2 Install Python Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3 Install AirSim Python API
```powershell
# If using pre-built binaries, copy API from AirSim folder
# If building from source, it will be in AirSim\PythonClient
cd E:\Drone\AirSim\PythonClient
pip install -e .
```

---

## Step 5: Test Installation

### 5.1 Launch Blocks Environment
1. Navigate to: `E:\Drone\AirSim\Blocks`
2. Double-click `Blocks.exe` (or run from command line)
3. Wait for Unreal Engine to load (first launch takes 2-5 minutes)

### 5.2 Run Test Script
```powershell
cd E:\Drone
python test_airsim.py
```

Expected output:
- Connection established
- Camera images captured
- Depth/segmentation data available

---

## Step 6: Verify GPU Acceleration

### Check CUDA in Python
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## Troubleshooting

### Common Issues:

1. **"MSVCP140.dll missing"**
   - Install: https://aka.ms/vs/17/release/vc_redist.x64.exe

2. **Unreal Engine won't launch**
   - Check Windows Defender exclusions
   - Verify GPU drivers are up to date

3. **Python connection timeout**
   - Ensure Blocks.exe is running first
   - Check firewall settings

4. **CUDA errors**
   - Verify CUDA installation: `nvcc --version`
   - Reinstall PyTorch with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

---

## Next Steps

After successful installation:
1. Run `test_airsim.py` to validate pipeline
2. Explore `examples/` directory for sample scripts
3. Begin procedural generation setup (see `PROCEDURAL_GENERATION.md`)

---

## Research Pipeline Structure

```
E:\Drone\
├── AirSim\              # AirSim binaries/source
├── Blocks\              # Blocks environment
├── venv\                # Python virtual environment
├── data\                # Generated datasets (100K+ images)
├── scripts\             # Procedural generation scripts
├── models\              # Trained models
└── papers\              # Research documentation
```

---

## References

- AirSim Documentation: https://microsoft.github.io/AirSim/
- AirSim GitHub: https://github.com/microsoft/AirSim
- Unreal Engine Docs: https://docs.unrealengine.com/4.27/en-US/
- PyTorch Installation: https://pytorch.org/get-started/locally/



# UAV Synthetic Data Research with Microsoft AirSim

Research project for generating large-scale synthetic training datasets using Microsoft AirSim for autonomous UAV systems.

## Project Overview

**Goal**: Generate 100K+ synthetic images with depth/segmentation data using procedural environment generation and domain randomization for sim-to-real transfer in UAV autonomy.

**System Specs**:
- Windows 10/11
- NVIDIA RTX 3060 (12GB VRAM)
- 128GB RAM
- Python 3.8+

## Quick Start

### For First-Time Installation (Recommended)

**Start here:** Follow the step-by-step walkthrough for installing pre-built binaries:

1. **Read Installation Walkthrough**
   ```powershell
   # Open and follow: INSTALLATION_WALKTHROUGH.md
   # This provides detailed step-by-step instructions
   ```

2. **Quick Verification**
   ```powershell
   .\quick_start.ps1
   ```

3. **Install AirSim Binaries**
   - Download from: https://github.com/microsoft/AirSim/releases
   - Extract `AirSim.zip` to `E:\Drone\AirSim`
   - Extract `Blocks.zip` to `E:\Drone\AirSim\Blocks`

4. **Setup Python Environment**
   ```powershell
   .\setup_airsim.ps1
   ```

5. **Launch Blocks Environment**
   - Navigate to `E:\Drone\AirSim\Blocks`
   - Run `Blocks.exe` (keep it running)

6. **Test Installation**
   ```powershell
   # In a NEW terminal window:
   cd E:\Drone
   .\venv\Scripts\Activate.ps1
   python test_airsim.py
   ```

### Installation Guides Available

- **`INSTALLATION_WALKTHROUGH.md`** - Step-by-step guide for pre-built binaries (START HERE)
- **`INSTALLATION_GUIDE.md`** - Comprehensive guide including source build options

## Project Structure

```
E:\Drone\
├── INSTALLATION_GUIDE.md    # Detailed installation instructions
├── requirements.txt          # Python dependencies
├── setup_airsim.ps1         # Automated setup script
├── test_airsim.py           # Installation validation
├── AirSim\                  # AirSim binaries/source
├── Blocks\                  # Blocks test environment
├── venv\                    # Python virtual environment
├── data\                    # Generated datasets (created during research)
├── scripts\                 # Procedural generation scripts
└── models\                  # Trained models
```

## Research Phases

### Phase 1: Installation & Validation (Weeks 1-2)
- ✅ AirSim installation
- ✅ Unreal Engine 4.27 setup
- ✅ Python environment configuration
- ✅ Pipeline validation

### Phase 2: Procedural Generation (Weeks 3-6)
- Implement domain randomization
- Develop procedural environment generation
- Create data collection pipeline

### Phase 3: Methodology & Publication (Weeks 7-12)
- Novel methodology development
- Dataset generation (100K+ images)
- Paper submission (ICRA/IROS)
- Open-source framework release

## Key Features

- **Synthetic Data Generation**: Automated dataset creation with depth/segmentation
- **Domain Randomization**: Weather, lighting, object placement variations
- **Sim-to-Real Transfer**: Bridging simulation and real-world UAV deployment
- **Scalable Pipeline**: Designed for 100K+ image generation

## Documentation

- [Installation Guide](INSTALLATION_GUIDE.md) - Complete setup instructions
- AirSim Docs: https://microsoft.github.io/AirSim/
- Unreal Engine Docs: https://docs.unrealengine.com/4.27/en-US/

## Requirements

See `requirements.txt` for Python dependencies. Key packages:
- `airsim` - AirSim Python API
- `torch` - PyTorch for deep learning
- `opencv-python` - Image processing
- `numpy` - Numerical computations

## Contributing

This is a research project. For issues or questions:
1. Check AirSim GitHub: https://github.com/microsoft/AirSim
2. Review installation troubleshooting section
3. Consult AirSim documentation

## License

Research project - see individual component licenses (AirSim: MIT, Unreal Engine: Epic Games License)

---

**Status**: Phase 1 - Installation & Setup



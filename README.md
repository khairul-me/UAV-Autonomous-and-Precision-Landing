# UAV Synthetic Data Research with Microsoft AirSim

> **Note**: This branch (`airsim-comprehensive-flight`) focuses on AirSim synthetic data generation and autonomous flight research. The `main` branch contains the Autonomous Drone Delivery System project. Both are separate research efforts under the same repository.

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

### Phase 0: AirSim Foundation Setup (COMPLETE ✅)
- ✅ AirSim installation and configuration
- ✅ Multi-sensor data capture (RGB, Depth, Segmentation, IMU, GPS, Magnetometer, Barometer)
- ✅ Comprehensive data logging system
- ✅ Autonomous flight patterns
- ✅ Real-time video recording

### Phase 1: Baseline Navigation System (In Progress)
- Manual control and data collection
- Vision-based object detection
- Classical navigation controller
- Deep RL navigation agent (optional)

### Phase 2: Adversarial Attack Implementation
- Digital attack generation (FGSM, PGD, C&W, UAP)
- Physical adversarial patch generation
- Multi-modal attacks
- Adaptive attacks

### Phase 3: Defense Implementation
- Input-level defenses
- Model-level defenses
- Multi-sensor fusion (key innovation)
- Behavioral defenses

## Key Features

- **Synthetic Data Generation**: Automated dataset creation with depth/segmentation
- **Multi-Sensor Capture**: RGB, Depth, Segmentation, IMU, GPS, Magnetometer, Barometer
- **Comprehensive Logging**: Synchronized timestamps, control commands, sensor data
- **Autonomous Flight**: Configurable flight patterns and exploration strategies
- **Real-time Recording**: Video recording with flight info overlays
- **Domain Randomization**: Weather, lighting, object placement variations
- **Sim-to-Real Transfer**: Bridging simulation and real-world UAV deployment
- **Scalable Pipeline**: Designed for 100K+ image generation

## Main Scripts

- **`autonomous_flight_comprehensive.py`** - Main comprehensive flight script with all sensors
- **`keyboard_control.py`** - Manual keyboard control for drone
- **`settings_comprehensive.json`** - Full sensor configuration
- **`run_comprehensive_flight.bat`** - Quick launcher for autonomous flight

## Documentation

- [Installation Guide](INSTALLATION_GUIDE.md) - Complete setup instructions
- [Phase 0 README](README_PHASE0.md) - Phase 0 implementation details
- [Troubleshooting Guide](README_TROUBLESHOOTING.md) - Common issues and solutions
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

**Author**: Md Khairul Islam  
**Institution**: Hobart and William Smith Colleges, Geneva, NY  
**Status**: Phase 0 Complete, Phase 1 In Progress

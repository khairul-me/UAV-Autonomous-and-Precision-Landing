# 🔧 Blocks Launch Solution

## The Problem

**Blocks.exe keeps exiting immediately** because **Blocks.zip doesn't include the AirSim plugin DLLs**.

Blocks.exe needs the AirSim plugin to run. The pre-built Blocks.zip from GitHub releases is just the Unreal Engine environment, but **doesn't include the AirSim plugin**.

## Solutions

### Option 1: Build AirSim Plugin (Recommended for Research)

Since you need full AirSim functionality for your research, you should build AirSim from source with the plugin:

1. **Install Unreal Engine 4.27** (if not already installed)
   - Download Epic Games Launcher
   - Install UE 4.27

2. **Build AirSim Plugin:**
   ```powershell
   cd E:\Drone
   git clone https://github.com/microsoft/AirSim.git AirSim_source
   cd AirSim_source
   git submodule update --init --recursive
   .\build.cmd
   ```

3. **Copy plugin to Blocks:**
   - After build completes, copy the plugin DLLs to Blocks

### Option 2: Use AirSim with Unreal Engine (Alternative)

If you have Unreal Engine 4.27 installed:

1. **Clone AirSim:**
   ```powershell
   cd E:\Drone
   git clone https://github.com/microsoft/AirSim.git
   ```

2. **Build Blocks environment with AirSim:**
   - Open Unreal Engine 4.27
   - Create/open Blocks project
   - Add AirSim plugin
   - Package for Windows

### Option 3: Use Pre-built AirSim Binaries (If Available)

Check if there's a complete AirSim+Blocks package:
- Some releases might include both
- Check: https://github.com/microsoft/AirSim/releases

## Quick Check

To verify if Blocks has AirSim plugin:

```powershell
cd E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins
Get-ChildItem -Recurse -Filter "*AirSim*.dll"
```

If no DLLs found → **Plugin is missing** → Need to build/install it.

## Current Status

✅ **Installed:**
- Python 3.11.9
- All Python packages (AirSim API, PyTorch, etc.)
- Blocks.zip extracted
- Directory structure ready

❌ **Missing:**
- AirSim plugin DLLs in Blocks environment

## Recommendation

For your research project (100K+ images, procedural generation), you'll need:
1. **Full AirSim build** with plugin
2. **Unreal Engine 4.27** for custom environments
3. **Ability to modify** environments for domain randomization

**Next Step:** Build AirSim from source or install Unreal Engine 4.27 and set up the plugin.

---

**Note:** The Blocks.zip from GitHub is just the base environment. The AirSim plugin needs to be built separately or obtained from a complete package.


# 🚀 AirSim API Connection - Quick Fix Guide

## 🎯 The Problem in 30 Seconds

**Issue:** Port 41451 not listening → Python can't connect to AirSim

**Root Cause:** Pre-built Blocks.zip from GitHub is missing compiled AirSim plugin DLL files

**Why Blocks Shows but API Doesn't Work:**
- ✅ Blocks.exe runs (Unreal Engine works)
- ✅ 3D environment displays
- ❌ AirSim plugin DLLs missing → API server can't start
- ❌ Port 41451 never listens → Python connection fails

---

## ✅ Quick Verification

Run this to confirm the issue:

```powershell
cd E:\Drone
.\quick_diagnostic.ps1
```

**Or manually:**
```powershell
# Check for plugin DLLs (should be EMPTY - that's the problem)
Get-ChildItem -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim" -Filter "*.dll" -Recurse

# Check if port is listening (should be EMPTY)
netstat -an | Select-String "41451"
```

---

## 🔧 The Solution (Choose One)

### **Option 1: Build from Source (RECOMMENDED) ⭐**

**Why:** Most reliable, gives you full control for research

**Prerequisites:**
- Visual Studio 2019/2022 Community (free)
- Unreal Engine 4.27 (free, via Epic Games Launcher)
- Git (usually already installed)

**Time:** 1-2 hours (mostly waiting for build)

**Steps:**
```powershell
# 1. Install prerequisites (if not already)
# - Visual Studio: https://visualstudio.microsoft.com/downloads/
# - UE 4.27: Epic Games Launcher

# 2. Build AirSim
cd E:\Drone
git clone https://github.com/microsoft/AirSim.git AirSim_source
cd AirSim_source
git checkout v1.8.1
git submodule update --init --recursive
.\build.cmd

# 3. Copy plugin to Blocks
$source = "E:\Drone\AirSim_source\Unreal\Plugins\AirSim"
$dest = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"
Rename-Item -Path $dest -NewName "AirSim_backup" -ErrorAction SilentlyContinue
Copy-Item -Path $source -Destination $dest -Recurse -Force

# 4. Verify DLLs exist
Get-ChildItem -Path $dest -Filter "*.dll" -Recurse
# Should now show DLL files!

# 5. Test
# Launch Blocks.exe, wait for load, then:
cd E:\Drone
.\venv\Scripts\Activate.ps1
python test_airsim.py
```

**Full details:** See `API_CONNECTION_TROUBLESHOOTING.md`

---

### **Option 2: Find Complete Pre-built Package**

**Check:**
- GitHub releases: https://github.com/microsoft/AirSim/releases
- Community builds (forums, Discord)
- Look for "complete" or "full" Windows builds

**Warning:** May not exist or be outdated

---

## 📋 Verification Checklist

After implementing the fix:

```powershell
# 1. Check DLLs exist (should show files now)
Get-ChildItem -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim" -Filter "*.dll" -Recurse

# 2. Launch Blocks.exe and wait 2-5 minutes

# 3. Check port is listening (in new terminal)
netstat -an | Select-String "41451"
# Should show: TCP    0.0.0.0:41451           0.0.0.0:0              LISTENING

# 4. Test connection
cd E:\Drone
.\venv\Scripts\Activate.ps1
python test_airsim.py
```

---

## ❓ FAQ

**Q: Why does Blocks run but API doesn't work?**
A: Blocks.zip includes Unreal Engine runtime but not compiled AirSim plugin DLLs. The plugin needs to be built separately.

**Q: Can I copy DLLs from AirSimNH?**
A: No - AirSimNH.zip also lacks compiled DLLs. Both need to be built.

**Q: Is Python 3.11.9 compatible?**
A: Yes! The issue is missing DLLs, not version incompatibility.

**Q: How long does building take?**
A: 30-60 minutes, mostly automated. You just need to wait.

**Q: Will this work for my research project?**
A: Yes! Building from source gives you the most control and reliability for generating 100K+ images.

---

## 📚 Full Documentation

For complete details, see:
- **`API_CONNECTION_TROUBLESHOOTING.md`** - Comprehensive guide with all diagnostic steps and solutions
- **`COMPLETE_INSTALLATION_SUMMARY.md`** - Everything that's been installed

---

## 🆘 Still Stuck?

1. Run `.\quick_diagnostic.ps1` to check current status
2. Review `API_CONNECTION_TROUBLESHOOTING.md` for detailed steps
3. Check Unreal Engine logs: `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Saved\Logs\Blocks.log`
4. Search AirSim GitHub issues: https://github.com/microsoft/AirSim/issues

---

**Bottom Line:** Build AirSim from source → Get plugin DLLs → Copy to Blocks → API works! 🎉

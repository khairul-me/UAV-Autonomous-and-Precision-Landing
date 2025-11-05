# 🚁 AirSim Installation - START HERE

**Quick Reference for Immediate Installation**

---

## 📋 Immediate Action Items

### Step 1: Download AirSim Pre-built Binaries

1. **Go to:** https://github.com/microsoft/AirSim/releases
2. **Download these files:**
   - `AirSim.zip` (or `AirSim-Windows.zip`)
   - `Blocks.zip`
3. **Extract:**
   - `AirSim.zip` → `E:\Drone\AirSim\`
   - `Blocks.zip` → `E:\Drone\AirSim\Blocks\`

### Step 2: Setup Python Environment

```powershell
cd E:\Drone
.\setup_airsim.ps1
```

This will:
- Create virtual environment
- Install all Python dependencies
- Verify prerequisites

### Step 3: Launch Blocks Environment

```powershell
cd E:\Drone\AirSim\Blocks
.\Blocks.exe
```

**Important:** Keep this window open while using AirSim!

### Step 4: Test Installation

**Open a NEW PowerShell window:**

```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python test_airsim.py
```

**Expected Result:**
- ✅ Connection successful
- ✅ Test images saved to `test_output/`
- ✅ All tests passed

---

## 📚 Detailed Guides

- **`INSTALLATION_WALKTHROUGH.md`** ⭐ **START HERE** - Complete step-by-step walkthrough
- **`INSTALLATION_GUIDE.md`** - Comprehensive reference (includes source build)
- **`quick_start.ps1`** - Quick verification script

---

## ✅ Success Criteria

After installation, you should have:

- ✅ `Blocks.exe` running successfully
- ✅ Python test script connects to AirSim
- ✅ Test images generated (scene, depth, segmentation)
- ✅ All dependencies installed (airsim, torch, opencv, numpy)

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Ensure `Blocks.exe` is running |
| Module not found | Activate venv: `.\venv\Scripts\Activate.ps1` |
| DLL missing | Install: https://aka.ms/vs/17/release/vc_redist.x64.exe |
| Blocks won't launch | Update GPU drivers, check Windows Defender |

**See `INSTALLATION_WALKTHROUGH.md` for detailed troubleshooting**

---

## 🎯 What's Next?

After successful installation:

1. **Explore AirSim API** - Review examples in `AirSim/PythonClient/examples/`
2. **Test Flight Controls** - Create basic takeoff/land script
3. **Plan Data Collection** - Design procedural generation pipeline
4. **Begin Research Phase** - Start implementing domain randomization

---

## 📞 Getting Help

- **AirSim Docs:** https://microsoft.github.io/AirSim/
- **GitHub Issues:** https://github.com/microsoft/AirSim/issues
- **Your Project Docs:** See `INSTALLATION_WALKTHROUGH.md`

---

**Ready to start?** Follow `INSTALLATION_WALKTHROUGH.md` step-by-step! 🚀


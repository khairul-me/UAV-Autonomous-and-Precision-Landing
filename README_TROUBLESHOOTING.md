# AirSim API Connection Troubleshooting - Complete Package

## 🎯 What You Have Now

A complete systematic troubleshooting package to diagnose and fix your AirSim API connection issue.

---

## 📚 Documentation Files

### 1. **SYSTEMATIC_TROUBLESHOOTING.md**
   - **Purpose:** Root cause analysis
   - **Contents:**
     - Why port 41451 isn't listening
     - Is copying plugin from AirSimNH correct?
     - Version compatibility analysis
     - Diagnostic steps explained
     - Solution options overview

### 2. **SYSTEMATIC_FIX_PLAN.md**
   - **Purpose:** Step-by-step fix guide
   - **Contents:**
     - Phase 1: Comprehensive diagnostics
     - Phase 2: Root cause analysis
     - Phase 3: Fix options (4 different approaches)
     - Phase 4: Validation steps
     - Success criteria
     - Rollback procedures

### 3. **QUICK_START_DIAGNOSTICS.md**
   - **Purpose:** Quick reference to start troubleshooting
   - **Contents:**
     - Commands to run immediately
     - Expected output
     - Next steps based on results

---

## 🔧 Diagnostic Tools

### 1. **diagnose_connection.py**
   - **Purpose:** Comprehensive Python diagnostic script
   - **What it checks:**
     - Python version
     - AirSim module import
     - Port 41451 availability
     - Blocks process status
     - Plugin files
     - Log files
     - Settings configuration
     - Connection test

   **Usage:**
   ```powershell
   cd E:\Drone
   .\venv\Scripts\Activate.ps1
   python diagnose_connection.py
   ```

### 2. **check_logs.ps1**
   - **Purpose:** Analyze Unreal Engine log files
   - **What it does:**
     - Finds log directories
     - Searches for AirSim messages
     - Shows plugin loading status
     - Displays errors

   **Usage:**
   ```powershell
   .\check_logs.ps1
   ```

### 3. **verify_plugin.ps1**
   - **Purpose:** Verify AirSim plugin installation
   - **What it checks:**
     - Plugin files exist
     - File sizes are correct
     - Compares with AirSimNH plugin
     - Shows what's missing

   **Usage:**
   ```powershell
   .\verify_plugin.ps1
   ```

---

## 🚀 Quick Start

### Step 1: Launch Blocks
```powershell
$exe = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor"
Start-Process -FilePath $exe -WorkingDirectory $workingDir
```

### Step 2: Wait 3 Minutes
Blocks needs time to initialize. The AirSim plugin may take 2-3 minutes to load.

### Step 3: Run Diagnostics
```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python diagnose_connection.py
```

### Step 4: Review Results
The diagnostic will tell you:
- ✅ What's working
- ❌ What's broken
- 🔧 What to fix

### Step 5: Follow Fix Plan
Based on diagnostic results, follow the appropriate fix option in `SYSTEMATIC_FIX_PLAN.md`.

---

## 📋 Diagnostic Checklist

Run these in order:

- [ ] **Python Version:** Should be 3.8+
- [ ] **AirSim Module:** Should import without errors
- [ ] **Settings:** Should be Multirotor mode, port 41451
- [ ] **Plugin Files:** Should exist in Blocks/Plugins/AirSim/
- [ ] **Blocks Process:** Should be running
- [ ] **Port 41451:** Should be listening
- [ ] **Log Files:** Should show AirSim messages
- [ ] **Connection:** Should connect successfully

---

## 🔍 Common Issues & Solutions

### Issue: Plugin Files Missing
**Diagnostic shows:** `[ERROR] Plugin directory does not exist`
**Fix:** See `SYSTEMATIC_FIX_PLAN.md` Option 1 - Copy plugin from AirSimNH

### Issue: Plugin Not Loading
**Diagnostic shows:** No AirSim messages in logs
**Fix:** See `SYSTEMATIC_FIX_PLAN.md` Option 2 - Check plugin compatibility

### Issue: API Server Not Starting
**Diagnostic shows:** Plugin loaded but port 41451 not listening
**Fix:** See `SYSTEMATIC_FIX_PLAN.md` Option 3 - Fix API server startup

### Issue: All Else Fails
**Diagnostic shows:** Everything checks but still doesn't work
**Fix:** See `SYSTEMATIC_FIX_PLAN.md` Option 4 - Build from source

---

## 📊 Expected Results

### ✅ Working Setup:
```
[OK] Port 41451 is LISTENING
[OK] Blocks process found
[OK] Plugin files exist
[OK] AirSim messages in logs
[SUCCESS] Connected to AirSim API!
```

### ❌ Problem Setup:
```
[ERROR] Port 41451 is NOT listening
[ERROR] No AirSim messages in logs
[ERROR] Plugin files missing
[ERROR] Connection failed: Connection refused
```

---

## 🎓 Understanding the Problem

### Root Cause Analysis (from SYSTEMATIC_TROUBLESHOOTING.md):

**Why Port 41451 Isn't Listening:**
1. Plugin not loaded - DLLs not being loaded by Unreal Engine
2. Plugin failed to initialize - Loaded but API server didn't start
3. Wrong plugin version - Incompatible plugin files
4. Missing dependencies - Plugin DLLs missing required files
5. Architecture mismatch - 32-bit vs 64-bit DLL mismatch
6. Configuration issue - Plugin not enabled in Blocks config

**Is Copying Plugin from AirSimNH Correct?**
- **Potential Issue:** AirSimNH and Blocks may have different Unreal Engine configurations
- **Better Approach:** Verify plugin structure, check Blocks-specific config files
- **Solution:** Copy plugin carefully, verify all files, check compatibility

**Version Compatibility:**
- Python 3.11.9 + AirSim 1.8.1: ✅ Compatible
- AirSim 1.8.1: Released 2024, supports Python 3.8+
- Unreal Engine 4.27: Blocks uses UE4.27, AirSim 1.8.1 is compatible

---

## 📝 Log File Locations

Check these locations for Unreal Engine logs:

1. `%LOCALAPPDATA%\AirSim\Blocks\Saved\Logs\`
2. `E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Saved\Logs\`
3. `%LOCALAPPDATA%\AirSim\Saved\Logs\`

**Key log file:** `Blocks.log`

**What to look for:**
- `[AirSim] Plugin loaded successfully`
- `[AirSim] API server starting on port 41451`
- `[AirSim] API server ready`
- Error messages about missing DLLs
- Plugin loading failures

---

## 🔧 Manual Verification Commands

### Check Port Status:
```powershell
Get-NetTCPConnection -LocalPort 41451 -State Listen
```

### Check Blocks Process:
```powershell
Get-Process | Where-Object { $_.Path -like "*Block*" }
```

### Check Plugin Files:
```powershell
Test-Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\Binaries\Win64\AirSim.dll"
```

### Check Settings:
```powershell
Get-Content "$env:USERPROFILE\Documents\AirSim\settings.json" | ConvertFrom-Json
```

---

## 🎯 Next Steps

1. **Read QUICK_START_DIAGNOSTICS.md** - Start here!
2. **Run diagnose_connection.py** - See what's wrong
3. **Check logs** - Find error messages
4. **Verify plugin** - Ensure files are correct
5. **Follow fix plan** - Apply appropriate solution
6. **Validate** - Test connection
7. **Report results** - Document what worked

---

## 💡 Tips

- **Be patient:** Blocks takes 2-3 minutes to initialize
- **Check logs first:** Logs often contain the exact error
- **Verify plugin files:** Missing files are the most common issue
- **Try different ports:** Sometimes port 41451 is blocked
- **Check firewall:** Windows Firewall may block connections
- **Restart Blocks:** After fixing plugin, always restart Blocks

---

## 📞 If You're Still Stuck

After running all diagnostics:

1. **Save diagnostic output** to a file
2. **Save log files** (especially Blocks.log)
3. **Document what you tried** from the fix plan
4. **Check AirSim GitHub issues** for similar problems
5. **Consider building from source** (Option 4 in fix plan)

---

## ✅ Success Criteria

You'll know it's working when:

- [ ] Port 41451 is listening
- [ ] `python diagnose_connection.py` shows all checks passing
- [ ] `client.confirmConnection()` succeeds
- [ ] Can get vehicle state
- [ ] Can capture images
- [ ] Can send control commands
- [ ] Drone flies in Blocks environment

---

**Good luck! The diagnostic tools will guide you to the solution.**


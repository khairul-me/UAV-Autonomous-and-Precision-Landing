# 🔧 AirSim API Connection Troubleshooting Guide

## 📋 Executive Summary

**Root Cause Identified:** The pre-built Blocks.zip and AirSimNH.zip from GitHub releases do NOT include compiled AirSim plugin DLL files. The plugin structure exists (`.uplugin` file and content assets) but the compiled binary DLLs that enable the API server are missing.

**Impact:** Without the plugin DLLs, Blocks.exe runs and shows the 3D environment, but the AirSim API server cannot start on port 41451, making Python connections impossible.

---

## 🔍 ROOT CAUSE ANALYSIS

### Why Port 41451 Isn't Listening

1. **Missing Compiled Plugin DLLs**
   - ✅ Plugin directory exists: `Blocks\Plugins\AirSim\`
   - ✅ Plugin descriptor exists: `AirSim.uplugin`
   - ✅ Content assets exist: 390+ files
   - ❌ **No compiled DLL files found**
   - ❌ Plugin cannot load → API server cannot start

2. **What the Plugin DLLs Do**
   - The AirSim plugin DLLs contain the compiled C++ code that:
     - Initializes the API server on port 41451
     - Handles msgpack-rpc communication
     - Bridges Unreal Engine with the Python API
   - Without these DLLs, Blocks is just a plain Unreal Engine environment

3. **Why Copying from AirSimNH Didn't Work**
   - AirSimNH.zip also lacks compiled DLLs
   - Both pre-built binaries from GitHub are incomplete
   - Copying incomplete plugin between environments doesn't help

4. **Version Compatibility**
   - Python 3.11.9 is compatible with AirSim 1.8.1
   - The issue is NOT a version incompatibility
   - The issue is missing compiled binaries

---

## 🔬 DIAGNOSTIC STEPS

### Step 1: Verify Plugin DLL Presence

**PowerShell Command:**
```powershell
# Check Blocks plugin
Get-ChildItem -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim" -Filter "*.dll" -Recurse

# Expected: Empty (0 results) = PROBLEM CONFIRMED
# Expected in working install: Multiple DLL files in Binaries\Win64\ subdirectory
```

**Success Criteria:**
- ❌ **Current State:** 0 DLL files found
- ✅ **Working State:** Should find DLLs like:
  - `AirSim.dll`
  - `AirSimModule.dll`
  - Other plugin dependency DLLs

---

### Step 2: Check Unreal Engine Logs

**Log Location:**
```
E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Saved\Logs\Blocks.log
```

**PowerShell Command to Check Latest Log:**
```powershell
$logPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Saved\Logs"
$latestLog = Get-ChildItem -Path $logPath -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $latestLog.FullName | Select-String -Pattern "AirSim|Plugin|API|41451|Error|Failed" -CaseSensitive:$false | Select-Object -First 20
```

**What to Look For:**
- ❌ **Plugin failed to load:** "AirSim plugin failed to initialize"
- ❌ **Missing module:** "Module 'AirSim' could not be loaded"
- ❌ **DLL missing:** "Could not find module 'AirSim.dll'"
- ✅ **Success:** "AirSim plugin initialized" or "API server started on port 41451"

---

### Step 3: Check Process and Network Status

**Run Diagnostic Script:**
```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python diagnose_api.py
```

**Or Manual Checks:**

**Check if Blocks is Running:**
```powershell
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue
```

**Check Port 41451 Status:**
```powershell
netstat -an | Select-String "41451"
# Should show: TCP    0.0.0.0:41451           0.0.0.0:0              LISTENING
# Current: No results = Port not listening
```

**Check Network Connections:**
```powershell
Get-NetTCPConnection -LocalPort 41451 -ErrorAction SilentlyContinue
# Should show listening connection
# Current: Empty = API server not running
```

---

### Step 4: Verify Settings.json

**Check All Settings Locations:**
```powershell
# Main settings (primary)
$settings1 = "$env:USERPROFILE\Documents\AirSim\settings.json"

# Blocks-specific settings
$settings2 = "E:\Drone\AirSim\Blocks\WindowsNoEditor\settings.json"

# Project settings
$settings3 = "E:\Drone\settings.json"

foreach ($s in @($settings1, $settings2, $settings3)) {
    if (Test-Path $s) {
        Write-Host "Found: $s"
        Get-Content $s | ConvertFrom-Json | Select-Object SimMode, ApiServerPort
    }
}
```

**Expected Configuration:**
- `SimMode`: `"Multirotor"`
- `ApiServerPort`: `41451`

---

### Step 5: Check Unreal Engine Console

**When Blocks is Running:**

1. Press **`** (backtick/tilde key) to open Unreal Engine console
2. Type: `stat plugins`
3. Look for AirSim plugin in the list
4. Check if it shows as "Loaded" or "Failed to Load"

**Alternative Console Commands:**
```
stat unit
log list
plugin list
```

---

### Step 6: Firewall Check

**Check Windows Firewall Rules:**
```powershell
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*AirSim*" -or $_.DisplayName -like "*Blocks*"}
```

**Test Localhost Connection (No Firewall Involved):**
```powershell
Test-NetConnection -ComputerName localhost -Port 41451
# Should show TcpTestSucceeded: True if API server is running
# Current: TcpTestSucceeded: False
```

---

## 🔧 SOLUTION OPTIONS

### **Solution 1: Build AirSim from Source (RECOMMENDED for Research)**

This is the most reliable solution and gives you full control for your research project.

#### Prerequisites:
1. **Visual Studio 2019/2022 Community** (free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select:
     - **Desktop development with C++**
     - **Windows 10/11 SDK** (latest)
     - **CMake tools for Windows**

2. **Unreal Engine 4.27** (required for building)
   - Download Epic Games Launcher: https://www.unrealengine.com/download
   - Install UE 4.27 via launcher

3. **Git** (if not already installed)

#### Build Steps:

```powershell
# 1. Clone AirSim repository
cd E:\Drone
git clone https://github.com/microsoft/AirSim.git AirSim_source
cd AirSim_source

# 2. Checkout version 1.8.1 (to match your Python API)
git checkout v1.8.1

# 3. Update submodules
git submodule update --init --recursive

# 4. Build AirSim plugin
.\build.cmd

# This will take 30-60 minutes
# After completion, plugin DLLs will be in: AirSim_source\Unreal\Plugins\AirSim\Binaries\Win64\
```

#### Copy Plugin to Blocks:

```powershell
# Source: Built plugin
$sourcePlugin = "E:\Drone\AirSim_source\Unreal\Plugins\AirSim"

# Destination: Blocks environment
$destPlugin = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"

# Backup existing plugin first
Rename-Item -Path $destPlugin -NewName "AirSim_backup" -ErrorAction SilentlyContinue

# Copy entire plugin directory
Copy-Item -Path $sourcePlugin -Destination $destPlugin -Recurse -Force
```

**Validation:**
```powershell
# Verify DLLs are present
Get-ChildItem -Path $destPlugin -Filter "*.dll" -Recurse
# Should now show multiple DLL files
```

---

### **Solution 2: Use AirSim with Unreal Engine Editor (Alternative)**

If you have UE 4.27 installed, you can package Blocks with the plugin:

#### Steps:

1. **Clone AirSim:**
```powershell
cd E:\Drone
git clone https://github.com/microsoft/AirSim.git AirSim_source
```

2. **Build Plugin for UE 4.27:**
```powershell
cd AirSim_source
.\build.cmd
```

3. **Open Blocks in Unreal Engine:**
   - Launch Unreal Engine 4.27
   - File → Open Project
   - Navigate to: `E:\Drone\AirSim_source\Unreal\Environments\Blocks\Blocks.uproject`
   - The plugin should auto-load

4. **Package for Windows:**
   - File → Package Project → Windows → Windows (64-bit)
   - This creates a complete executable with the plugin compiled in

---

### **Solution 3: Download Complete Pre-built Package (If Available)**

Some community builds or alternative releases might include complete binaries:

**Check:**
- AirSim releases page: https://github.com/microsoft/AirSim/releases
- Look for releases labeled "Windows-Complete" or similar
- Check community forums/discussions for working builds

**Warning:** These may not be officially maintained or up-to-date.

---

### **Solution 4: Use AirSimNH as Workaround (NOT RECOMMENDED)**

**Why Not Recommended:**
- AirSimNH is designed for car simulation
- Environment is not suitable for drone research
- Also lacks compiled DLLs (same problem)

**If You Must Try:**
- AirSimNH also needs plugin DLLs built
- Same build process required
- Better to fix Blocks properly

---

## 📝 STEP-BY-STEP FIX (Recommended Approach)

### Phase 1: Preparation (15-30 minutes)

1. **Install Visual Studio Build Tools**
   ```powershell
   # Download and install from:
   # https://visualstudio.microsoft.com/downloads/
   # Select "Desktop development with C++" workload
   ```

2. **Install Unreal Engine 4.27**
   ```powershell
   # Download Epic Games Launcher
   # Install UE 4.27 (requires Epic account, free)
   ```

3. **Verify Prerequisites:**
   ```powershell
   # Check Git
   git --version
   
   # Check CMake (comes with Visual Studio)
   cmake --version
   ```

### Phase 2: Build AirSim Plugin (30-60 minutes)

1. **Clone and Build:**
   ```powershell
   cd E:\Drone
   git clone https://github.com/microsoft/AirSim.git AirSim_source
   cd AirSim_source
   git checkout v1.8.1
   git submodule update --init --recursive
   .\build.cmd
   ```

2. **Monitor Build:**
   - Watch for errors
   - Build should complete without critical errors
   - Final message should indicate success

### Phase 3: Install Plugin to Blocks (5 minutes)

1. **Backup Existing Plugin:**
   ```powershell
   $pluginPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"
   if (Test-Path $pluginPath) {
       Rename-Item -Path $pluginPath -NewName "AirSim_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
   }
   ```

2. **Copy Built Plugin:**
   ```powershell
   $source = "E:\Drone\AirSim_source\Unreal\Plugins\AirSim"
   $dest = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"
   Copy-Item -Path $source -Destination $dest -Recurse -Force
   ```

3. **Verify DLLs:**
   ```powershell
   Get-ChildItem -Path $dest -Filter "*.dll" -Recurse
   # Should show multiple DLL files now
   ```

### Phase 4: Test Installation (10 minutes)

1. **Launch Blocks:**
   ```powershell
   cd E:\Drone\AirSim\Blocks\WindowsNoEditor
   .\Blocks.exe
   ```

2. **Wait for Full Load:**
   - Wait 2-5 minutes for initialization
   - Watch for Unreal Engine console errors (press ` key)

3. **Check Port:**
   ```powershell
   # In new PowerShell window
   netstat -an | Select-String "41451"
   # Should show: TCP    0.0.0.0:41451           0.0.0.0:0              LISTENING
   ```

4. **Test Connection:**
   ```powershell
   cd E:\Drone
   .\venv\Scripts\Activate.ps1
   python diagnose_api.py
   ```

5. **Run Full Test:**
   ```powershell
   python test_airsim.py
   ```

---

## ✅ VALIDATION CHECKLIST

After implementing the fix, verify each item:

- [ ] Plugin DLLs exist in `Blocks\Plugins\AirSim\Binaries\Win64\`
- [ ] Blocks.exe launches without errors
- [ ] Port 41451 is listening (check with `netstat`)
- [ ] `diagnose_api.py` reports "Port 41451 is LISTENING"
- [ ] `test_airsim.py` successfully connects
- [ ] Can capture images from AirSim
- [ ] Can retrieve vehicle state
- [ ] No errors in Unreal Engine console

---

## 🔄 ROLLBACK PROCEDURE

If something goes wrong during the build/install:

1. **Restore Original Plugin:**
   ```powershell
   $backup = Get-ChildItem -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins" -Filter "AirSim_backup_*" | Sort-Object Name -Descending | Select-Object -First 1
   if ($backup) {
       Remove-Item -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim" -Recurse -Force -ErrorAction SilentlyContinue
       Rename-Item -Path $backup.FullName -NewName "AirSim"
   }
   ```

2. **Clean Build Directory (if needed):**
   ```powershell
   Remove-Item -Path "E:\Drone\AirSim_source" -Recurse -Force -ErrorAction SilentlyContinue
   ```

---

## 📚 ADDITIONAL RESOURCES

### Official Documentation:
- AirSim Build Instructions: https://microsoft.github.io/AirSim/build_windows/
- AirSim API Reference: https://microsoft.github.io/AirSim/api/
- Unreal Engine 4.27 Docs: https://docs.unrealengine.com/4.27/en-US/

### Community Support:
- AirSim GitHub Issues: https://github.com/microsoft/AirSim/issues
- AirSim Discord: Check AirSim GitHub for invite link
- Stack Overflow: Tag questions with `airsim`

### Diagnostic Tools Created:
- `diagnose_api.py` - Comprehensive connection diagnostic
- This guide - Complete troubleshooting reference

---

## 🎯 EXPECTED OUTCOMES

### After Successful Fix:

1. **Blocks launches** and initializes normally
2. **Port 41451 listens** within 30-60 seconds after Blocks loads
3. **Python scripts connect** successfully
4. **API calls work** (get images, control drone, etc.)
5. **No connection errors** in Python scripts

### Success Indicators:

```powershell
# Port check should show:
netstat -an | Select-String "41451"
# TCP    0.0.0.0:41451           0.0.0.0:0              LISTENING

# Python test should show:
python test_airsim.py
# [OK] Connection established successfully!
# [OK] AirSim API Version: 1.8.1
# [OK] Camera available: front_center
# [OK] Test images saved
```

---

## 📞 NEXT STEPS

1. **Choose Solution 1** (Build from Source) - Most reliable
2. **Gather Prerequisites** - Visual Studio, UE 4.27
3. **Follow Phase-by-Phase Guide** above
4. **Validate at Each Step** using provided commands
5. **Test Connection** once plugin is installed

---

**Last Updated:** Based on diagnostic run showing 0 DLL files in plugin directories
**Status:** Root cause identified - Missing compiled plugin DLLs
**Recommended Action:** Build AirSim from source to generate plugin DLLs

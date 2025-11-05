# Systematic Fix Plan - Step by Step

## Phase 1: Comprehensive Diagnostics (Run First)

### Step 1.1: Run Python Diagnostic Script
```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python diagnose_connection.py
```

**What to look for:**
- Which checks pass/fail
- Specific error messages
- Plugin file status
- Port status

**Document results:** Save output to `diagnostic_results.txt`

---

### Step 1.2: Check Log Files
```powershell
.\check_logs.ps1
```

**What to look for:**
- AirSim plugin loading messages
- API server startup messages
- Error messages about missing DLLs
- Network binding errors

**Key messages to find:**
- `[AirSim] Plugin loaded successfully`
- `[AirSim] API server starting on port 41451`
- `[AirSim] API server ready`

**If plugin not loading:**
- Look for "Module 'AirSim' failed to load"
- Look for "Missing DLL" errors
- Check for "Plugin not found" messages

---

### Step 1.3: Verify Plugin Installation
```powershell
.\verify_plugin.ps1
```

**What to check:**
- All plugin files exist
- File sizes are reasonable (not 0 bytes)
- DLLs match between Blocks and AirSimNH
- Plugin descriptor is valid JSON

---

### Step 1.4: Check Process and Network
```powershell
# Check if Blocks is running
Get-Process | Where-Object { $_.Path -like "*Block*" }

# Check port status
Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue

# Check firewall rules
Get-NetFirewallApplicationFilter -Program "*Blocks.exe" -ErrorAction SilentlyContinue
```

---

## Phase 2: Root Cause Analysis

Based on diagnostic results, identify the issue:

### Scenario A: Plugin Files Missing
**Symptoms:**
- `verify_plugin.ps1` shows missing files
- Logs show "Plugin not found"

**Root Cause:** Blocks.zip doesn't include plugin, or plugin wasn't copied correctly

**Fix:** See Phase 3, Option 1

---

### Scenario B: Plugin Files Present But Not Loading
**Symptoms:**
- Plugin files exist
- No AirSim messages in logs
- Port 41451 not listening

**Root Cause:** 
- Plugin incompatible with Blocks
- Plugin descriptor incorrect
- DLL dependencies missing

**Fix:** See Phase 3, Option 2

---

### Scenario C: Plugin Loads But API Server Doesn't Start
**Symptoms:**
- Logs show plugin loaded
- No API server messages
- Port 41451 not listening

**Root Cause:**
- API server initialization failed
- Port binding issue
- Configuration problem

**Fix:** See Phase 3, Option 3

---

## Phase 3: Fix Options (Choose Based on Diagnostics)

### Option 1: Fix Plugin Copy (If Files Missing)

**Step 3.1.1: Stop Blocks**
```powershell
Get-Process | Where-Object { $_.Path -like "*Block*" } | Stop-Process -Force
```

**Step 3.1.2: Remove Existing Plugin**
```powershell
Remove-Item -Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim" -Recurse -Force -ErrorAction SilentlyContinue
```

**Step 3.1.3: Copy Plugin from AirSimNH**
```powershell
$source = "E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Plugins\AirSim"
$dest = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"

# Create destination directory
New-Item -ItemType Directory -Path $dest -Force | Out-Null

# Copy all files
Copy-Item -Path "$source\*" -Destination $dest -Recurse -Force

# Verify copy
Get-ChildItem -Path $dest -Recurse | Measure-Object -Property Length -Sum
```

**Step 3.1.4: Verify Copy Success**
```powershell
.\verify_plugin.ps1
```

**Step 3.1.5: Launch Blocks and Test**
```powershell
$exe = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor"
Start-Process -FilePath $exe -WorkingDirectory $workingDir

# Wait 3 minutes
Write-Host "Waiting 3 minutes for Blocks to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 180

# Test connection
python diagnose_connection.py
```

---

### Option 2: Check Plugin Compatibility (If Plugin Not Loading)

**Step 3.2.1: Check Plugin Descriptor**
```powershell
$uplugin = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\AirSim.uplugin"
Get-Content $uplugin | ConvertFrom-Json | Format-List
```

**Check:**
- EngineVersion matches (should be 4.27)
- Plugin name is "AirSim"
- Modules section is correct

**Step 3.2.2: Check DLL Dependencies**
```powershell
# Use Dependency Walker or check manually
$dll = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\Binaries\Win64\AirSim.dll"
Get-Item $dll | Select-Object VersionInfo
```

**Step 3.2.3: Check Unreal Engine Module Loading**
```powershell
# Launch Blocks and check console
# Press ` key to open console
# Look for module loading messages
```

**Step 3.2.4: Try Alternative Plugin Version**
If current plugin doesn't work, try getting plugin from:
- AirSim source repository
- Different AirSim version
- Community builds

---

### Option 3: Fix API Server Startup (If Plugin Loads But API Doesn't Start)

**Step 3.3.1: Check Settings.json**
```powershell
$settings = Get-Content "$env:USERPROFILE\Documents\AirSim\settings.json" | ConvertFrom-Json
$settings | Format-List
```

**Verify:**
- ApiServerPort is 41451
- SimMode is Multirotor
- Vehicles section is correct

**Step 3.3.2: Check for Port Conflicts**
```powershell
# Check if another process is using port 41451
Get-NetTCPConnection -LocalPort 41451 -ErrorAction SilentlyContinue
```

**Step 3.3.3: Try Different Port**
```json
{
  "ApiServerPort": 41452
}
```

**Step 3.3.4: Check Firewall**
```powershell
# Allow Blocks through firewall
New-NetFirewallRule -DisplayName "AirSim Blocks" -Direction Inbound -Program "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe" -Action Allow
```

---

### Option 4: Build from Source (Nuclear Option)

**When to use:**
- All other options fail
- Need guaranteed working plugin
- Want latest features

**Step 3.4.1: Install Unreal Engine 4.27**
- Download from Epic Games Launcher
- Install to: `C:\Program Files\Epic Games\UE_4.27\`

**Step 3.4.2: Clone AirSim Source**
```powershell
git clone https://github.com/microsoft/AirSim.git E:\Drone\AirSim\source
cd E:\Drone\AirSim\source
```

**Step 3.4.3: Build Plugin**
```powershell
# See AirSim documentation for build instructions
# Requires Visual Studio 2019 or 2022
```

**Step 3.4.4: Copy Built Plugin to Blocks**
```powershell
# Copy from build output to Blocks
```

---

## Phase 4: Validation

After each fix attempt:

### Step 4.1: Launch Blocks
```powershell
$exe = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor"
Start-Process -FilePath $exe -WorkingDirectory $workingDir
```

### Step 4.2: Wait for Initialization
```powershell
Write-Host "Waiting 3 minutes for initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 180
```

### Step 4.3: Check Port Status
```powershell
$port = Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue
if ($port) {
    Write-Host "[SUCCESS] Port 41451 is listening!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Port 41451 is not listening" -ForegroundColor Red
}
```

### Step 4.4: Run Diagnostic Script
```powershell
python diagnose_connection.py
```

### Step 4.5: Test Connection
```powershell
python -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('SUCCESS')"
```

### Step 4.6: Test Flight
```powershell
python MAKE_IT_FLY.py
```

---

## Success Criteria

**Plugin Loading:**
- [ ] Plugin DLLs exist and are correct size
- [ ] Logs show "Plugin loaded" message
- [ ] No errors in Unreal Engine console

**API Server:**
- [ ] Port 41451 is listening
- [ ] Logs show "API server ready"
- [ ] Process shows network connection

**Python Connection:**
- [ ] `client.confirmConnection()` succeeds
- [ ] Can get vehicle state
- [ ] Can capture images
- [ ] Can send control commands

**Flight Test:**
- [ ] Drone takes off
- [ ] Drone moves as commanded
- [ ] Drone lands successfully

---

## Rollback Plan

If something breaks:

1. **Stop Blocks:**
   ```powershell
   Get-Process | Where-Object { $_.Path -like "*Block*" } | Stop-Process -Force
   ```

2. **Restore Original Blocks:**
   ```powershell
   Remove-Item -Path "E:\Drone\AirSim\Blocks" -Recurse -Force
   # Re-extract Blocks.zip
   ```

3. **Reset Settings:**
   ```powershell
   Remove-Item -Path "$env:USERPROFILE\Documents\AirSim\settings.json" -Force
   # Let AirSim create default
   ```

---

## Next Steps

1. **Run diagnostics first** (Phase 1)
2. **Identify root cause** (Phase 2)
3. **Apply appropriate fix** (Phase 3)
4. **Validate** (Phase 4)
5. **Report results**

Start with Phase 1 and work through systematically.


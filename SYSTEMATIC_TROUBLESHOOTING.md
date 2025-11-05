# AirSim API Connection - Systematic Troubleshooting Guide

## Root Cause Analysis

### Why Port 41451 Isn't Listening

The AirSim API server runs as a plugin within Unreal Engine. If port 41451 isn't listening, it means:

1. **Plugin Not Loaded** - The AirSim plugin DLLs aren't being loaded by Unreal Engine
2. **Plugin Failed to Initialize** - Plugin loaded but failed to start the API server
3. **Wrong Plugin Version** - Incompatible plugin files copied from AirSimNH
4. **Missing Dependencies** - Plugin DLLs missing required dependencies
5. **Architecture Mismatch** - 32-bit vs 64-bit DLL mismatch
6. **Unreal Engine Configuration** - Blocks doesn't have plugin enabled in its config

### Is Copying Plugin from AirSimNH Correct?

**Potential Issue:** AirSimNH and Blocks may have different Unreal Engine configurations. The plugin might need Blocks-specific initialization or config files.

**Better Approach:** We should verify:
- If Blocks.zip actually includes the plugin
- If the plugin structure is correct for Blocks
- If there are Blocks-specific config files needed

### Version Compatibility

- **Python 3.11.9 + AirSim 1.8.1**: Should be compatible
- **AirSim 1.8.1**: Released in 2024, supports Python 3.8+
- **Unreal Engine 4.27**: Blocks uses UE4.27, AirSim 1.8.1 is compatible

---

## Diagnostic Steps

### Step 1: Check Unreal Engine Logs

**Location:** `%LOCALAPPDATA%\AirSim\Blocks\Saved\Logs\`

**What to look for:**
- Plugin loading messages
- API server startup messages
- Error messages about missing DLLs
- Network binding errors

**Key log files:**
- `Blocks.log` - Main Unreal Engine log
- `AirSim.log` - AirSim-specific log (if plugin loaded)

### Step 2: Verify Plugin Files

**Expected Structure:**
```
Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\
├── AirSim.uplugin          (Plugin descriptor)
├── Binaries\
│   └── Win64\
│       ├── AirSim.dll      (Main plugin DLL)
│       └── AirSim.lib
└── Content\
```

**What to check:**
- All DLLs present
- File sizes match expected (not 0 bytes)
- No corrupted files
- Correct architecture (64-bit)

### Step 3: Check Unreal Engine Console

**How to open:** Press `` ` `` (backtick) key in Blocks window

**What should appear:**
```
[AirSim] Plugin loaded successfully
[AirSim] API server starting on port 41451
[AirSim] API server ready
```

**If plugin not loaded:**
- No AirSim messages
- Error messages about missing modules

### Step 4: Check Windows Firewall

**What to check:**
- Blocks.exe allowed through firewall
- Port 41451 not blocked
- No antivirus blocking plugin DLLs

### Step 5: Verify Process and Network

**What to check:**
- Blocks.exe is running
- Process has network permissions
- No other process using port 41451
- Network adapter is active

---

## Verification Commands

### PowerShell Commands

```powershell
# Check if port 41451 is listening
Get-NetTCPConnection -LocalPort 41451 -State Listen -ErrorAction SilentlyContinue

# Check Blocks process
Get-Process | Where-Object { $_.Path -like "*Block*" }

# Check if plugin DLLs exist
Test-Path "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim\Binaries\Win64\AirSim.dll"

# Check log files
Get-ChildItem "$env:LOCALAPPDATA\AirSim\Blocks\Saved\Logs\" -ErrorAction SilentlyContinue

# Check firewall rules for Blocks
Get-NetFirewallApplicationFilter -Program "*Blocks.exe" -ErrorAction SilentlyContinue
```

### Python Diagnostic Script

See `diagnose_connection.py` for comprehensive connection testing.

---

## Solution Options

### Option 1: Build AirSim from Source (RECOMMENDED)

**Pros:**
- Guaranteed to have correct plugin files
- Can customize for specific needs
- Full control over configuration

**Cons:**
- Requires Unreal Engine 4.27 installation
- Long build time (30-60 minutes)
- More complex setup

**When to use:**
- If pre-built binaries don't work
- Need custom features
- Want latest improvements

### Option 2: Use Different Pre-built Binary

**Check:**
- AirSim community forks
- Older/newer AirSim releases
- Alternative download sources

### Option 3: Fix Plugin Copy (QUICKEST)

**If plugin copy is correct approach:**
- Verify all files copied correctly
- Check file permissions
- Ensure plugin descriptor (.uplugin) is correct
- Verify DLL dependencies

### Option 4: Use AirSimNH with Multirotor Mode (WORKAROUND)

**Pros:**
- AirSimNH has working plugin
- Can force Multirotor mode

**Cons:**
- Environment designed for cars
- May have visual inconsistencies
- Not ideal for drone research

**When to use:**
- As temporary workaround
- To test API connection
- If Blocks can't be fixed quickly

---

## Step-by-Step Fix Plan

### Phase 1: Comprehensive Diagnostics

1. **Check Log Files**
   - Navigate to log directory
   - Read Blocks.log for errors
   - Look for AirSim plugin messages

2. **Verify Plugin Structure**
   - Check all plugin files exist
   - Verify file sizes and dates
   - Compare with working AirSimNH plugin

3. **Test Plugin Loading**
   - Launch Blocks
   - Open Unreal Engine console
   - Check for plugin messages

4. **Network Diagnostics**
   - Check port status
   - Verify firewall rules
   - Test localhost connection

5. **Python Connection Test**
   - Run diagnostic script
   - Check error messages
   - Verify timeout vs connection refused

### Phase 2: Fix Attempts

#### Fix Attempt 1: Verify Plugin Copy

1. Stop Blocks completely
2. Delete plugin folder
3. Re-copy from AirSimNH with verification
4. Check file hashes match
5. Launch Blocks and test

#### Fix Attempt 2: Check Plugin Descriptor

1. Open AirSim.uplugin
2. Verify it's valid JSON
3. Check version numbers
4. Verify engine compatibility

#### Fix Attempt 3: Rebuild from Source

1. Download Unreal Engine 4.27
2. Clone AirSim source
3. Build plugin for Blocks
4. Copy to Blocks environment

#### Fix Attempt 4: Alternative Binary

1. Try different AirSim version
2. Check community builds
3. Use AirSimNH temporarily

### Phase 3: Validation

After each fix attempt:
1. Launch Blocks
2. Wait 3 minutes for initialization
3. Check port 41451
4. Run Python connection test
5. Verify drone can be controlled

---

## Success Criteria

### At Each Step:

**Plugin Loading:**
- [ ] Plugin DLLs exist in correct location
- [ ] File sizes > 0 and reasonable
- [ ] No permission errors

**Blocks Launch:**
- [ ] Blocks.exe starts without errors
- [ ] Environment loads visually
- [ ] No crash dialogs

**API Server:**
- [ ] Port 41451 is listening
- [ ] Logs show "API server ready"
- [ ] Process shows network connections

**Python Connection:**
- [ ] `client.confirmConnection()` succeeds
- [ ] Can get vehicle state
- [ ] Can capture images
- [ ] Can send control commands

---

## Rollback Procedures

**If something breaks:**

1. **Restore Original Blocks:**
   - Delete modified plugin folder
   - Re-extract Blocks.zip
   - Restore original settings.json

2. **Reset Settings:**
   - Delete settings.json
   - Let AirSim create default

3. **Clean Python:**
   - Recreate virtual environment
   - Reinstall packages

---

## Next Steps

1. Run diagnostic scripts
2. Check log files
3. Verify plugin structure
4. Follow fix attempts in order
5. Report findings at each step


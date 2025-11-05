# Quick Start - Run Diagnostics Now

## Step 1: Run Python Diagnostic (Most Important)

**Make sure Blocks.exe is running first!**

```powershell
cd E:\Drone
.\venv\Scripts\Activate.ps1
python diagnose_connection.py
```

**This will check:**
- Python version
- AirSim module import
- Port 41451 status
- Blocks process
- Plugin files
- Log files
- Settings configuration
- Connection test

**Save the output** - it tells you exactly what's wrong.

---

## Step 2: Check Log Files

```powershell
.\check_logs.ps1
```

**This will:**
- Find log files
- Search for AirSim messages
- Show plugin loading status
- Display errors

---

## Step 3: Verify Plugin Installation

```powershell
.\verify_plugin.ps1
```

**This will:**
- Check if plugin files exist
- Verify file sizes
- Compare with AirSimNH plugin
- Show what's missing

---

## Step 4: Check Port Status

```powershell
Get-NetTCPConnection -LocalPort 41451 -State Listen
```

**If nothing shows:**
- API server is not running
- Plugin may not be loaded

---

## Step 5: Read Results

Based on diagnostic results:

1. **If plugin files missing** → See `SYSTEMATIC_FIX_PLAN.md` Option 1
2. **If plugin not loading** → See `SYSTEMATIC_FIX_PLAN.md` Option 2
3. **If API server not starting** → See `SYSTEMATIC_FIX_PLAN.md` Option 3
4. **If all else fails** → See `SYSTEMATIC_FIX_PLAN.md` Option 4 (Build from source)

---

## Expected Output

### ✅ Working Setup:
```
[OK] Port 41451 is LISTENING
[OK] Blocks process found
[OK] Plugin files exist
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

## Next Steps

After running diagnostics, follow the fix plan in `SYSTEMATIC_FIX_PLAN.md` based on what the diagnostics reveal.


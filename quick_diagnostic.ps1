# AirSim API Connection Quick Diagnostic Script
# Run this script to quickly check the status of your AirSim installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AirSim API Connection Diagnostic" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$issuesFound = 0

# Check 1: Blocks Process
Write-Host "[1/6] Checking if Blocks.exe is running..." -ForegroundColor Yellow
$blocksProcess = Get-Process -Name "Blocks" -ErrorAction SilentlyContinue
if ($blocksProcess) {
    Write-Host "  ✓ Blocks.exe is running (PID: $($blocksProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "  ✗ Blocks.exe is NOT running" -ForegroundColor Red
    Write-Host "    → Start Blocks.exe first!" -ForegroundColor Yellow
    $issuesFound++
}

Write-Host ""

# Check 2: Port 41451
Write-Host "[2/6] Checking if port 41451 is listening..." -ForegroundColor Yellow
$portCheck = Get-NetTCPConnection -LocalPort 41451 -ErrorAction SilentlyContinue
if ($portCheck -and $portCheck.State -eq "Listen") {
    Write-Host "  ✓ Port 41451 is LISTENING" -ForegroundColor Green
} else {
    Write-Host "  ✗ Port 41451 is NOT listening" -ForegroundColor Red
    Write-Host "    → AirSim API server is not running" -ForegroundColor Yellow
    $issuesFound++
}

Write-Host ""

# Check 3: Plugin DLLs
Write-Host "[3/6] Checking for AirSim plugin DLLs..." -ForegroundColor Yellow
$pluginPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"
if (Test-Path $pluginPath) {
    $dlls = Get-ChildItem -Path $pluginPath -Filter "*.dll" -Recurse -ErrorAction SilentlyContinue
    if ($dlls) {
        Write-Host "  ✓ Found $($dlls.Count) DLL file(s) in plugin" -ForegroundColor Green
        $dlls | Select-Object -First 3 | ForEach-Object {
            $sizeKB = [math]::Round($_.Length / 1KB, 1)
            Write-Host "    - $($_.Name) ($sizeKB KB)" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ✗ No DLL files found in plugin directory" -ForegroundColor Red
        Write-Host "    → Plugin DLLs are missing - this is the root cause!" -ForegroundColor Yellow
        Write-Host "    → See API_CONNECTION_TROUBLESHOOTING.md for solution" -ForegroundColor Yellow
        $issuesFound++
    }
} else {
    Write-Host "  ✗ Plugin directory not found" -ForegroundColor Red
    $issuesFound++
}

Write-Host ""

# Check 4: Settings.json
Write-Host "[4/6] Checking settings.json..." -ForegroundColor Yellow
$settingsPath = "$env:USERPROFILE\Documents\AirSim\settings.json"
if (Test-Path $settingsPath) {
    try {
        $settings = Get-Content $settingsPath | ConvertFrom-Json
        Write-Host "  ✓ Settings file found" -ForegroundColor Green
        Write-Host "    SimMode: $($settings.SimMode)" -ForegroundColor Gray
        Write-Host "    ApiServerPort: $($settings.ApiServerPort)" -ForegroundColor Gray
        if ($settings.SimMode -ne "Multirotor") {
            Write-Host "    ⚠ Warning: SimMode is not 'Multirotor'" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠ Settings file exists but could not be read" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ Settings file not found at expected location" -ForegroundColor Yellow
    Write-Host "    Expected: $settingsPath" -ForegroundColor Gray
}

Write-Host ""

# Check 5: Python Environment
Write-Host "[5/6] Checking Python environment..." -ForegroundColor Yellow
$venvPython = "E:\Drone\venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "  ✓ Virtual environment found" -ForegroundColor Green
    try {
        $airsimCheck = & $venvPython -c "import airsim; print('OK')" 2>&1
        if ($airsimCheck -match "OK") {
            Write-Host "  ✓ AirSim Python API is importable" -ForegroundColor Green
        } else {
            Write-Host "  ✗ AirSim Python API import failed" -ForegroundColor Red
            $issuesFound++
        }
    } catch {
        Write-Host "  ⚠ Could not test AirSim import" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ Virtual environment not found" -ForegroundColor Yellow
}

Write-Host ""

# Check 6: Connection Test
Write-Host "[6/6] Testing TCP connection to localhost:41451..." -ForegroundColor Yellow
try {
    $tcpClient = New-Object System.Net.Sockets.TcpClient
    $result = $tcpClient.BeginConnect("127.0.0.1", 41451, $null, $null)
    $wait = $result.AsyncWaitHandle.WaitOne(2000, $false)
    if ($wait -and $tcpClient.Connected) {
        $tcpClient.EndConnect($result)
        Write-Host "  ✓ Successfully connected to port 41451" -ForegroundColor Green
        $tcpClient.Close()
    } else {
        Write-Host "  ✗ Connection refused or timeout" -ForegroundColor Red
        Write-Host "    → API server is not responding" -ForegroundColor Yellow
        $issuesFound++
    }
} catch {
    Write-Host "  ✗ Connection error: $($_.Exception.Message)" -ForegroundColor Red
    $issuesFound++
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($issuesFound -eq 0) {
    Write-Host "✓ All checks passed! AirSim should be working." -ForegroundColor Green
    Write-Host ""
    Write-Host "Try running: python test_airsim.py" -ForegroundColor Cyan
} else {
    Write-Host "✗ Found $issuesFound issue(s)" -ForegroundColor Red
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "1. Review the issues above" -ForegroundColor White
    Write-Host "2. See API_CONNECTION_TROUBLESHOOTING.md for detailed solutions" -ForegroundColor White
    Write-Host "3. Most likely fix: Build AirSim from source to get plugin DLLs" -ForegroundColor White
}
Write-Host "========================================" -ForegroundColor Cyan

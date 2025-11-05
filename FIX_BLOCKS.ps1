# Blocks Launch Fix Script
# This script attempts to diagnose and fix Blocks launch issues

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Blocks Launch Diagnostic & Fix" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$blocksExe = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor"

# Check if Blocks.exe exists
if (-not (Test-Path $blocksExe)) {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    Write-Host "Expected: $blocksExe" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Blocks.exe found" -ForegroundColor Green
Write-Host ""

# Check for Unreal Engine DLLs in Blocks directory
Write-Host "Checking for Unreal Engine DLLs..." -ForegroundColor Yellow
$ueDlls = @("UE4Editor*.dll", "Unreal*.dll", "AirSim*.dll")
$foundDlls = Get-ChildItem -Path $workingDir -Recurse -Include $ueDlls -ErrorAction SilentlyContinue | Select-Object -First 10 Name
if ($foundDlls) {
    Write-Host "[OK] Found Unreal Engine DLLs" -ForegroundColor Green
    $foundDlls | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor Gray }
} else {
    Write-Host "[WARNING] Few or no Unreal Engine DLLs found" -ForegroundColor Yellow
}

Write-Host ""

# Try to launch with different methods
Write-Host "Attempting to launch Blocks..." -ForegroundColor Cyan
Write-Host ""

# Method 1: Direct launch
Write-Host "[1/3] Trying direct launch..." -ForegroundColor Yellow
$proc1 = Start-Process -FilePath $blocksExe -WorkingDirectory $workingDir -WindowStyle Normal -PassThru -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
if (Get-Process -Id $proc1.Id -ErrorAction SilentlyContinue) {
    Write-Host "[SUCCESS] Blocks is running with direct launch! PID: $($proc1.Id)" -ForegroundColor Green
    Write-Host "Blocks window should appear shortly. Wait 2-3 minutes for full load." -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "[FAILED] Direct launch failed" -ForegroundColor Red
}

# Method 2: Launch via cmd.exe
Write-Host "[2/3] Trying launch via cmd.exe..." -ForegroundColor Yellow
$proc2 = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "cd /d `"$workingDir`" && `"$blocksExe`"" -WindowStyle Normal -PassThru -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5
if (Get-Process -Name "Blocks" -ErrorAction SilentlyContinue) {
    Write-Host "[SUCCESS] Blocks is running via cmd! PID: $((Get-Process -Name "Blocks").Id)" -ForegroundColor Green
    Write-Host "Blocks window should appear shortly." -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "[FAILED] Cmd launch failed" -ForegroundColor Red
}

# Method 3: Launch with explorer
Write-Host "[3/3] Opening Blocks directory for manual launch..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MANUAL LAUNCH REQUIRED" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Blocks needs to be launched manually. Opening directory..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Yellow
Write-Host "1. Find Blocks.exe in the opened window" -ForegroundColor White
Write-Host "2. Double-click Blocks.exe" -ForegroundColor White
Write-Host "3. Wait 2-5 minutes for it to load" -ForegroundColor White
Write-Host "4. If it asks for admin rights, click Yes" -ForegroundColor White
Write-Host "5. If you see any dialogs, accept them" -ForegroundColor White
Write-Host ""

Start-Process explorer.exe -ArgumentList $workingDir

Write-Host ""
Write-Host "After Blocks launches successfully, run:" -ForegroundColor Cyan
Write-Host "  cd E:\Drone" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  python test_airsim.py" -ForegroundColor Gray
Write-Host ""


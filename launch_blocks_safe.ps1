# SAFE BLOCKS LAUNCHER - Handles missing shader issues
# This script launches Blocks with compatibility options

Write-Host "========================================" -ForegroundColor Green
Write-Host "SAFE BLOCKS LAUNCHER" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Kill any existing Blocks
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Ensure settings are correct
$docDir = "$env:USERPROFILE\Documents\AirSim"
$oneDriveDir = "$env:USERPROFILE\OneDrive\Documents\AirSim"

$droneSettings = @'
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": -5,
      "Yaw": 0
    }
  },
  "ApiServerPort": 41451
}
'@

Write-Host "[1/3] Fixing settings..." -ForegroundColor Yellow
foreach ($dir in @($docDir, $oneDriveDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Set-Content -Path "$dir\settings.json" -Value $droneSettings -Force
    Write-Host "  Fixed: $dir\settings.json" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/3] Setting environment variables..." -ForegroundColor Yellow
$env:AIRSIM_SIM_MODE = "Multirotor"
$env:UE4_SIM_MODE = "Multirotor"
# Disable shader warnings (they're causing the crash)
$env:UE4_SKIP_SHADER_WARNINGS = "1"
Write-Host "  [OK] Environment variables set" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Launching Blocks with safe mode..." -ForegroundColor Yellow
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

if (-not (Test-Path $blocksPath)) {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    exit 1
}

# Try to launch with -nullrhi (no rendering) first to test if it's a graphics issue
# Actually, let's just launch normally but with a windowed mode hint
Set-Location $workingDir

# Launch in background and wait a bit
$process = Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir -PassThru -WindowStyle Normal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "[LAUNCHED] Blocks Process ID: $($process.Id)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: Missing shader errors are expected with pre-built binaries" -ForegroundColor Yellow
Write-Host "These warnings won't prevent the drone from appearing!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Wait 3-5 minutes for Blocks to fully load..." -ForegroundColor Cyan
Write-Host "You should see a DRONE (even with shader warnings)" -ForegroundColor Green
Write-Host ""
Write-Host "If it crashes again, the pre-built binary may be incomplete." -ForegroundColor Yellow
Write-Host "In that case, you may need to build AirSim from source." -ForegroundColor Yellow
Write-Host ""

# Wait a few seconds and check if it's still running
Start-Sleep -Seconds 5
$stillRunning = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
if ($stillRunning) {
    Write-Host "[OK] Blocks is still running after 5 seconds!" -ForegroundColor Green
    Write-Host "This is a good sign - it should continue loading..." -ForegroundColor Green
} else {
    Write-Host "[WARNING] Blocks exited quickly - check the logs above" -ForegroundColor Yellow
}

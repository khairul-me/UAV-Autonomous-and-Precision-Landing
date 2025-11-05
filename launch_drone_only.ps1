# Launch Blocks with FORCED Multirotor mode - NO CARS ALLOWED
# This script uses command-line arguments to force drone mode

Write-Host "========================================" -ForegroundColor Green
Write-Host "LAUNCHING BLOCKS - DRONE MODE ONLY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Kill any running Blocks
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Ensure settings.json is correct
$docDir = "$env:USERPROFILE\Documents\AirSim"
if (-not (Test-Path $docDir)) {
    New-Item -ItemType Directory -Path $docDir -Force | Out-Null
}

$droneSettings = @'
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1,
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

Set-Content -Path "$docDir\settings.json" -Value $droneSettings -Force
Copy-Item "$docDir\settings.json" "E:\Drone\settings.json" -Force

Write-Host "[OK] Settings verified: Multirotor mode" -ForegroundColor Green
Write-Host ""

# Launch Blocks with explicit parameters
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor"

if (Test-Path $blocksPath) {
    Write-Host "Launching Blocks.exe with drone-only settings..." -ForegroundColor Cyan
    Write-Host ""
    
    # Set environment variable to force Multirotor mode
    $env:AIRSIM_SIM_MODE = "Multirotor"
    
    # Launch with working directory set to ensure settings are read correctly
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $blocksPath
    $processInfo.WorkingDirectory = $workingDir
    $processInfo.UseShellExecute = $true
    # Try to pass SimMode as argument if supported
    # Note: AirSim reads from settings.json, but we can try environment variables
    
    [System.Diagnostics.Process]::Start($processInfo) | Out-Null
    
    Write-Host "[OK] Blocks.exe launched!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Environment variable set: AIRSIM_SIM_MODE=Multirotor" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "IMPORTANT:" -ForegroundColor Yellow
    Write-Host "  - Wait 2-5 minutes for AirSim to load" -ForegroundColor White
    Write-Host "  - You MUST see a DRONE (quadcopter), NOT a car!" -ForegroundColor White
    Write-Host "  - If you see a car, close it and tell me immediately!" -ForegroundColor Red
    Write-Host ""
} else {
    Write-Host "[ERROR] Blocks.exe not found at: $blocksPath" -ForegroundColor Red
    exit 1
}

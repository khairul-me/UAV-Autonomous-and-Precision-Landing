# FIX ONEDRIVE SETTINGS - The Real Problem!
# AirSim reads from OneDrive\Documents\AirSim\settings.json on some systems

Write-Host "========================================" -ForegroundColor Red
Write-Host "FIXING ONEDRIVE SETTINGS - DRONE ONLY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Kill Blocks first
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Settings locations (Documents AND OneDrive)
$settingsLocations = @(
    "$env:USERPROFILE\Documents\AirSim\settings.json",
    "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
)

# Drone-only settings
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

Write-Host "[1/2] Fixing settings in ALL locations..." -ForegroundColor Yellow
foreach ($loc in $settingsLocations) {
    $dir = Split-Path $loc
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created directory: $dir" -ForegroundColor Cyan
    }
    
    Set-Content -Path $loc -Value $droneSettings -Force
    Write-Host "  Fixed: $loc" -ForegroundColor Green
    
    # Verify
    $content = Get-Content $loc -Raw
    if ($content -match '"SimMode"\s*:\s*"Multirotor"' -and $content -notmatch '"SimMode"\s*:\s*"Car"') {
        Write-Host "    [OK] Verified: Multirotor mode" -ForegroundColor Green
    } else {
        Write-Host "    [ERROR] Verification failed!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[2/2] Launching Blocks..." -ForegroundColor Yellow
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

if (Test-Path $blocksPath) {
    Set-Location $workingDir
    Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] ALL SETTINGS FIXED!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Fixed settings in:" -ForegroundColor Cyan
    Write-Host "  - Documents\AirSim\settings.json" -ForegroundColor White
    Write-Host "  - OneDrive\Documents\AirSim\settings.json" -ForegroundColor White
    Write-Host ""
    Write-Host "WAIT 2-5 MINUTES FOR AIRSIM TO LOAD" -ForegroundColor Yellow
    Write-Host "YOU MUST SEE A DRONE NOW (NOT A CAR)!" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    exit 1
}

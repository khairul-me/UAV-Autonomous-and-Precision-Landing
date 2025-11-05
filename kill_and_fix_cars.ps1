# AGGRESSIVE FIX - Remove ALL cars, force drone-only mode
# This script does everything to ensure NO CARS appear

Write-Host "========================================" -ForegroundColor Red
Write-Host "AGGRESSIVE CAR REMOVAL - DRONE ONLY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Step 1: Kill ALL AirSim processes aggressively
Write-Host "[1/5] Killing ALL AirSim processes..." -ForegroundColor Yellow
$processes = @("Blocks", "AirSimNH", "UE4Editor", "UnrealEngine", "UE4*")
foreach ($proc in $processes) {
    Get-Process | Where-Object {$_.ProcessName -like $proc} | Stop-Process -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
Write-Host "[OK] All processes killed" -ForegroundColor Green
Write-Host ""

# Step 2: Delete any cached settings
Write-Host "[2/5] Removing cached settings..." -ForegroundColor Yellow
$cacheLocations = @(
    "$env:USERPROFILE\Documents\AirSim",
    "$env:APPDATA\AirSim",
    "$env:LOCALAPPDATA\AirSim"
)
foreach ($cache in $cacheLocations) {
    if (Test-Path "$cache\settings.json") {
        Remove-Item "$cache\settings.json" -Force -ErrorAction SilentlyContinue
        Write-Host "  Removed: $cache\settings.json" -ForegroundColor Cyan
    }
}
Write-Host "[OK] Cached settings removed" -ForegroundColor Green
Write-Host ""

# Step 3: Create clean drone-only settings
Write-Host "[3/5] Creating clean drone-only settings..." -ForegroundColor Yellow
$docDir = "$env:USERPROFILE\Documents\AirSim"
if (-not (Test-Path $docDir)) {
    New-Item -ItemType Directory -Path $docDir -Force | Out-Null
}

$cleanSettings = @"
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
"@

Set-Content -Path "$docDir\settings.json" -Value $cleanSettings -Force
Copy-Item "$docDir\settings.json" "E:\Drone\settings.json" -Force
Write-Host "[OK] Clean drone-only settings created" -ForegroundColor Green
Write-Host ""

# Step 4: Verify NO car references exist
Write-Host "[4/5] Verifying NO car references..." -ForegroundColor Yellow
$settingsContent = Get-Content "$docDir\settings.json" -Raw
if ($settingsContent -match '"SimMode"\s*:\s*"Car"') {
    Write-Host "[ERROR] CAR MODE FOUND! Removing..." -ForegroundColor Red
    $settingsContent = $settingsContent -replace '"SimMode"\s*:\s*"Car"', '"SimMode": "Multirotor"'
    Set-Content -Path "$docDir\settings.json" -Value $settingsContent -Force
    Write-Host "[OK] Removed car mode" -ForegroundColor Green
}
if ($settingsContent -match '"VehicleType"\s*:\s*"PhysXCar"') {
    Write-Host "[ERROR] CAR VEHICLE FOUND! Removing..." -ForegroundColor Red
    $settingsContent = $settingsContent -replace '"VehicleType"\s*:\s*"PhysXCar"', '"VehicleType": "SimpleFlight"'
    Set-Content -Path "$docDir\settings.json" -Value $settingsContent -Force
    Write-Host "[OK] Removed car vehicle" -ForegroundColor Green
}
Write-Host "[OK] Verified: NO car references" -ForegroundColor Green
Write-Host ""

# Step 5: Show final settings
Write-Host "[5/5] Final settings verification..." -ForegroundColor Yellow
$final = Get-Content "$docDir\settings.json" -Raw
Write-Host ""
Write-Host "Final settings.json:" -ForegroundColor Cyan
Write-Host $final
Write-Host ""

if ($final -match '"SimMode"\s*:\s*"Multirotor"' -and $final -notmatch '"SimMode"\s*:\s*"Car"') {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] DRONE MODE ONLY - NO CARS!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Settings verification failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Now launching Blocks with DRONE mode..." -ForegroundColor Cyan
Write-Host ""
Start-Sleep -Seconds 2

$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
if (Test-Path $blocksPath) {
    Start-Process -FilePath $blocksPath -WorkingDirectory (Split-Path $blocksPath)
    Write-Host "[OK] Blocks.exe launched with DRONE-ONLY settings!" -ForegroundColor Green
    Write-Host ""
    Write-Host "WAIT 2-5 MINUTES for AirSim to load" -ForegroundColor Yellow
    Write-Host "YOU MUST SEE A DRONE (quadcopter with 4 propellers)" -ForegroundColor Yellow
    Write-Host "IF YOU SEE A CAR, TELL ME IMMEDIATELY!" -ForegroundColor Red
} else {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    exit 1
}

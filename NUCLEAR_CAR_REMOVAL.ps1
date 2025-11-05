# NUCLEAR OPTION - Remove ALL traces of cars
# This script does EVERYTHING to ensure only drones appear

Write-Host "========================================" -ForegroundColor Red
Write-Host "NUCLEAR CAR REMOVAL - DRONE ONLY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Step 1: Kill everything
Write-Host "[1/6] Killing ALL AirSim/Unreal processes..." -ForegroundColor Yellow
$allProcesses = Get-Process | Where-Object {
    $_.ProcessName -like "*Blocks*" -or
    $_.ProcessName -like "*AirSim*" -or
    $_.ProcessName -like "*UE4*" -or
    $_.ProcessName -like "*Unreal*"
}
foreach ($proc in $allProcesses) {
    Write-Host "  Killing: $($proc.ProcessName)" -ForegroundColor Cyan
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 3
Write-Host "[OK] All processes killed" -ForegroundColor Green
Write-Host ""

# Step 2: Delete ALL possible settings files
Write-Host "[2/6] Deleting ALL settings files..." -ForegroundColor Yellow
$settingsLocations = @(
    "$env:USERPROFILE\Documents\AirSim\settings.json",
    "$env:APPDATA\AirSim\settings.json",
    "$env:LOCALAPPDATA\AirSim\settings.json",
    "E:\Drone\settings.json",
    "E:\Drone\AirSim\Blocks\WindowsNoEditor\settings.json"
)
foreach ($loc in $settingsLocations) {
    if (Test-Path $loc) {
        Remove-Item $loc -Force -ErrorAction SilentlyContinue
        Write-Host "  Deleted: $loc" -ForegroundColor Cyan
    }
}
Write-Host "[OK] All old settings deleted" -ForegroundColor Green
Write-Host ""

# Step 3: Create MINIMAL drone-only settings
Write-Host "[3/6] Creating minimal drone-only settings..." -ForegroundColor Yellow
$docDir = "$env:USERPROFILE\Documents\AirSim"
if (-not (Test-Path $docDir)) {
    New-Item -ItemType Directory -Path $docDir -Force | Out-Null
}

$minimalDroneSettings = @'
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight"
    }
  }
}
'@

Set-Content -Path "$docDir\settings.json" -Value $minimalDroneSettings -Force
Copy-Item "$docDir\settings.json" "E:\Drone\settings.json" -Force
Write-Host "[OK] Minimal drone settings created" -ForegroundColor Green
Write-Host ""

# Step 4: Verify settings content
Write-Host "[4/6] Verifying settings..." -ForegroundColor Yellow
$settingsContent = Get-Content "$docDir\settings.json" -Raw
if ($settingsContent -notmatch '"SimMode"\s*:\s*"Multirotor"') {
    Write-Host "[ERROR] SimMode is NOT Multirotor!" -ForegroundColor Red
    exit 1
}
if ($settingsContent -match '"SimMode"\s*:\s*"Car"') {
    Write-Host "[ERROR] CAR MODE STILL PRESENT!" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Settings verified: Multirotor only" -ForegroundColor Green
Write-Host ""

# Step 5: Display final settings
Write-Host "[5/6] Final settings content:" -ForegroundColor Yellow
Write-Host $settingsContent
Write-Host ""

# Step 6: Launch with environment variables
Write-Host "[6/6] Launching Blocks with forced Multirotor mode..." -ForegroundColor Yellow
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

if (-not (Test-Path $blocksPath)) {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    exit 1
}

# Set environment variables to force Multirotor
$env:AIRSIM_SIM_MODE = "Multirotor"
$env:UE4_SIM_MODE = "Multirotor"

# Change to working directory and launch
Set-Location $workingDir
Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "BLOCKS LAUNCHED - DRONE MODE FORCED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "  AIRSIM_SIM_MODE=Multirotor" -ForegroundColor White
Write-Host "  UE4_SIM_MODE=Multirotor" -ForegroundColor White
Write-Host ""
Write-Host "Settings file:" -ForegroundColor Cyan
Write-Host "  SimMode: Multirotor" -ForegroundColor White
Write-Host "  VehicleType: SimpleFlight" -ForegroundColor White
Write-Host ""
Write-Host "WAIT 2-5 MINUTES FOR AIRSIM TO LOAD" -ForegroundColor Yellow
Write-Host ""
Write-Host "YOU MUST SEE A DRONE (quadcopter with 4 propellers)" -ForegroundColor Green
Write-Host "IF YOU SEE A CAR, IT MEANS BLOCKS IS IGNORING SETTINGS!" -ForegroundColor Red
Write-Host ""

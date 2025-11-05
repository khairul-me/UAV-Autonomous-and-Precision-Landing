# COMPREHENSIVE LAUNCHER - Blocks + Keyboard Control
# This script ensures everything is ready before launching keyboard control

Write-Host "========================================" -ForegroundColor Green
Write-Host "COMPREHENSIVE LAUNCHER - DRONE CONTROL" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Step 1: Ensure settings are correct
Write-Host "[1/5] Verifying settings..." -ForegroundColor Yellow
$settingsPath = "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
$correctSettings = @'
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1,
  "DefaultVehicle": "Drone1",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0,
      "Y": 0,
      "Z": -5,
      "Yaw": 0
    }
  },
  "ApiServerPort": 41451,
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase"
}
'@

$docDir = Split-Path $settingsPath
if (-not (Test-Path $docDir)) {
    New-Item -ItemType Directory -Path $docDir -Force | Out-Null
}
Set-Content -Path $settingsPath -Value $correctSettings -Force
Set-Content -Path "$env:USERPROFILE\Documents\AirSim\settings.json" -Value $correctSettings -Force
Write-Host "  [OK] Settings verified" -ForegroundColor Green
Write-Host ""

# Step 2: Check if Blocks is running
Write-Host "[2/5] Checking Blocks status..." -ForegroundColor Yellow
$blocks = Get-Process -Name "Blocks" -ErrorAction SilentlyContinue
if ($blocks) {
    Write-Host "  [OK] Blocks is running (PID: $($blocks.Id))" -ForegroundColor Green
    $blocksRunning = $true
} else {
    Write-Host "  [INFO] Blocks is not running, launching it..." -ForegroundColor Cyan
    $blocksRunning = $false
    
    # Launch Blocks
    $blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
    $workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"
    
    if (Test-Path $blocksPath) {
        $env:AIRSIM_SIM_MODE = "Multirotor"
        Set-Location $workingDir
        Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir
        Write-Host "  [OK] Blocks launched!" -ForegroundColor Green
        Write-Host "  [WAIT] Waiting 60 seconds for Blocks to load..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    } else {
        Write-Host "  [ERROR] Blocks.exe not found!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 3: Wait for Blocks to be fully ready
Write-Host "[3/5] Waiting for Blocks to fully load..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$apiReady = $false

while ($attempt -lt $maxAttempts -and -not $apiReady) {
    $attempt++
    $attemptStr = "$attempt of $maxAttempts"
    Write-Host "  Attempt $attemptStr : Testing API connection..." -ForegroundColor Cyan
    
    $testResult = & "E:\Drone\venv\Scripts\python.exe" -c @"
import airsim
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print('READY')
except Exception as e:
    print('NOT_READY')
"@ 2>&1 | Select-String -Pattern "READY|NOT_READY"
    
    if ($testResult -match "READY") {
        Write-Host "  [OK] API is ready!" -ForegroundColor Green
        $apiReady = $true
    } else {
        Write-Host "  [WAIT] API not ready yet, waiting 5 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

if (-not $apiReady) {
    Write-Host "  [ERROR] API not ready after $maxAttempts attempts!" -ForegroundColor Red
    Write-Host "  Blocks may still be loading. Try again in a few minutes." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 4: Verify drone is spawned
Write-Host "[4/5] Verifying drone is spawned..." -ForegroundColor Yellow
$verifyResult = & "E:\Drone\venv\Scripts\python.exe" -c @"
import airsim
try:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    state = client.getMultirotorState()
    print('DRONE_OK')
except Exception as e:
    print(f'ERROR: {e}')
"@ 2>&1 | Select-String -Pattern "DRONE_OK|ERROR"

if ($verifyResult -match "DRONE_OK") {
    Write-Host "  [OK] Drone is ready!" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Drone verification had issues, but continuing..." -ForegroundColor Yellow
    Write-Host "  $verifyResult" -ForegroundColor Gray
}

Write-Host ""

# Step 5: Launch keyboard control
Write-Host "[5/5] Launching keyboard control..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "[READY] Starting keyboard control!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Cyan
Write-Host "  1. Press [C] to claim control" -ForegroundColor White
Write-Host "  2. Press [T] to take off" -ForegroundColor White
Write-Host "  3. Use WASD/Arrow keys to move" -ForegroundColor White
Write-Host "  4. Press [L] to land" -ForegroundColor White
Write-Host ""
Start-Sleep -Seconds 2

# Launch keyboard control
Set-Location "E:\Drone"
& ".\run_keyboard_control.bat"


# LAUNCH REALISTIC ENVIRONMENT (AirSimNH)
# This launches the Neighborhood environment - more realistic urban setting

Write-Host "========================================" -ForegroundColor Green
Write-Host "LAUNCHING REALISTIC ENVIRONMENT" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Kill any existing AirSim processes
Get-Process -Name "AirSimNH" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Ensure settings are correct for drone in realistic environment
$docDir = "$env:USERPROFILE\Documents\AirSim"
$oneDriveDir = "$env:USERPROFILE\OneDrive\Documents\AirSim"

$realisticSettings = @'
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
      "Z": -10,
      "Yaw": 0
    }
  },
  "ApiServerPort": 41451,
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase",
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 1920,
        "Height": 1080,
        "FOV_Degrees": 90
      }
    ]
  }
}
'@

Write-Host "[1/3] Configuring settings for realistic environment..." -ForegroundColor Yellow
foreach ($dir in @($docDir, $oneDriveDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Set-Content -Path "$dir\settings.json" -Value $realisticSettings -Force
    Write-Host "  Configured: $dir\settings.json" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/3] Setting environment variables..." -ForegroundColor Yellow
$env:AIRSIM_SIM_MODE = "Multirotor"
$env:UE4_SIM_MODE = "Multirotor"
$env:UE4_SKIP_SHADER_WARNINGS = "1"
Write-Host "  [OK] Environment variables set" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Launching AirSimNH (Neighborhood environment)..." -ForegroundColor Yellow
$nhPath = "E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64\AirSimNH.exe"
$workingDir = "E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64"

if (-not (Test-Path $nhPath)) {
    Write-Host "[ERROR] AirSimNH.exe not found!" -ForegroundColor Red
    Write-Host "Path: $nhPath" -ForegroundColor Yellow
    exit 1
}

Set-Location $workingDir
$process = Start-Process -FilePath $nhPath -WorkingDirectory $workingDir -PassThru -WindowStyle Normal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "[LAUNCHED] AirSimNH Process ID: $($process.Id)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "AirSimNH (Neighborhood Environment) is launching..." -ForegroundColor Cyan
Write-Host "This is a realistic urban environment with:" -ForegroundColor White
Write-Host "  - Houses and buildings" -ForegroundColor Gray
Write-Host "  - Streets and roads" -ForegroundColor Gray
Write-Host "  - Trees and vegetation" -ForegroundColor Gray
Write-Host "  - Realistic lighting" -ForegroundColor Gray
Write-Host ""
Write-Host "Wait 3-5 minutes for full load..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Once loaded, you can run:" -ForegroundColor Cyan
Write-Host "  .\run_realistic_flight.bat" -ForegroundColor White
Write-Host ""

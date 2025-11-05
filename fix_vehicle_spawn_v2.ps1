# FIX VEHICLE SPAWN V2 - Alternative configuration
# Some Blocks versions need DefaultVehicle setting

Write-Host "========================================" -ForegroundColor Red
Write-Host "FIXING VEHICLE SPAWN - VERSION 2" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Kill Blocks
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Alternative settings with DefaultVehicle
$altSettings = @'
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
      "Yaw": 0,
      "PawnPath": ""
    }
  },
  "ApiServerPort": 41451,
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase"
}
'@

Write-Host "[1/2] Writing alternative settings..." -ForegroundColor Yellow
$settingsLocations = @(
    "$env:USERPROFILE\Documents\AirSim\settings.json",
    "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
)

foreach ($loc in $settingsLocations) {
    Set-Content -Path $loc -Value $altSettings -Force
    Write-Host "  Updated: $loc" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/2] Launching Blocks..." -ForegroundColor Yellow
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

$env:AIRSIM_SIM_MODE = "Multirotor"
Set-Location $workingDir
Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "LAUNCHED WITH DEFAULT VEHICLE SETTING" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "New settings include:" -ForegroundColor Cyan
Write-Host "  - DefaultVehicle: Drone1" -ForegroundColor White
Write-Host "  - This tells AirSim which vehicle to spawn by default" -ForegroundColor White
Write-Host ""
Write-Host "Wait 2-5 minutes and check console for '1 Vehicles spawned'" -ForegroundColor Yellow
Write-Host ""

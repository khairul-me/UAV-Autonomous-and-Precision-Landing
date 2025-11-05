# FIX VEHICLE SPAWN - Correct settings format for AirSim
# The issue: Settings load but 0 vehicles spawned
# Solution: Use correct AirSim settings format with proper vehicle configuration

Write-Host "========================================" -ForegroundColor Red
Write-Host "FIXING VEHICLE SPAWN ISSUE" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Kill Blocks first
Get-Process -Name "Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# CORRECT AirSim settings format for Multirotor with vehicle spawn
$correctSettings = @'
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
  "ApiServerPort": 41451,
  "RpcEnabled": true,
  "ViewMode": "SpringArmChase"
}
'@

Write-Host "[1/3] Updating settings in ALL locations..." -ForegroundColor Yellow
$settingsLocations = @(
    "$env:USERPROFILE\Documents\AirSim\settings.json",
    "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json",
    "E:\Drone\settings.json"
)

foreach ($loc in $settingsLocations) {
    $dir = Split-Path $loc
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    
    Set-Content -Path $loc -Value $correctSettings -Force
    Write-Host "  Fixed: $loc" -ForegroundColor Green
    
    # Verify
    $content = Get-Content $loc -Raw
    if ($content -match '"SimMode"\s*:\s*"Multirotor"' -and $content -match '"Vehicles"') {
        Write-Host "    [OK] Verified: Multirotor mode + Vehicles section" -ForegroundColor Green
    } else {
        Write-Host "    [ERROR] Verification failed!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[2/3] Verifying settings format..." -ForegroundColor Yellow
$testContent = Get-Content "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json" -Raw
try {
    $json = $testContent | ConvertFrom-Json
    Write-Host "  [OK] JSON is valid" -ForegroundColor Green
    Write-Host "  SimMode: $($json.SimMode)" -ForegroundColor Cyan
    if ($json.Vehicles) {
        Write-Host "  Vehicles found: $($json.Vehicles.PSObject.Properties.Name -join ', ')" -ForegroundColor Cyan
        foreach ($vehicleName in $json.Vehicles.PSObject.Properties.Name) {
            $vehicle = $json.Vehicles.$vehicleName
            Write-Host "    - $vehicleName : Type=$($vehicle.VehicleType), Pos=($($vehicle.X), $($vehicle.Y), $($vehicle.Z))" -ForegroundColor White
        }
    } else {
        Write-Host "  [ERROR] No Vehicles section!" -ForegroundColor Red
    }
} catch {
    Write-Host "  [ERROR] JSON parse failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "[3/3] Launching Blocks..." -ForegroundColor Yellow
$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

if (Test-Path $blocksPath) {
    Set-Location $workingDir
    
    # Set environment variables
    $env:AIRSIM_SIM_MODE = "Multirotor"
    $env:UE4_SIM_MODE = "Multirotor"
    
    # Launch Blocks
    Start-Process -FilePath $blocksPath -WorkingDirectory $workingDir
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] SETTINGS FIXED AND LAUNCHED!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Settings now include:" -ForegroundColor Cyan
    Write-Host "  - SimMode: Multirotor" -ForegroundColor White
    Write-Host "  - Vehicles: Drone1 (SimpleFlight)" -ForegroundColor White
    Write-Host "  - Position: (0, 0, -5)" -ForegroundColor White
    Write-Host "  - ApiServerPort: 41451" -ForegroundColor White
    Write-Host ""
    Write-Host "WAIT 2-5 MINUTES FOR AIRSIM TO LOAD" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You should now see:" -ForegroundColor Green
    Write-Host "  - '1 Vehicles spawned' in the console" -ForegroundColor Green
    Write-Host "  - A DRONE (quadcopter) in the 3D view" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    exit 1
}

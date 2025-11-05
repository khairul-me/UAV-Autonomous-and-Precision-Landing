# Fix Drone Mode - Ensure NO CARS, ONLY DRONES
# This script kills any running AirSim and ensures drone-only mode

Write-Host "========================================" -ForegroundColor Red
Write-Host "FIXING DRONE MODE - REMOVING CARS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

# Step 1: Kill all AirSim processes
Write-Host "[1/4] Killing any running AirSim processes..." -ForegroundColor Yellow
$processes = @("Blocks", "AirSimNH", "UE4Editor", "UnrealEngine")
foreach ($proc in $processes) {
    $running = Get-Process -Name $proc -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "  Found $proc - Killing..." -ForegroundColor Cyan
        Stop-Process -Name $proc -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}
Write-Host "[OK] All AirSim processes stopped" -ForegroundColor Green
Write-Host ""

# Step 2: Verify settings.json is correct
Write-Host "[2/4] Verifying settings.json..." -ForegroundColor Yellow
$workspaceSettings = "E:\Drone\settings.json"
$docSettings = "$env:USERPROFILE\Documents\AirSim\settings.json"

# Ensure Documents directory exists
$docDir = Split-Path $docSettings
if (-not (Test-Path $docDir)) {
    New-Item -ItemType Directory -Path $docDir -Force | Out-Null
}

# Copy workspace settings to Documents (where AirSim actually reads from)
if (Test-Path $workspaceSettings) {
    Copy-Item $workspaceSettings $docSettings -Force
    Write-Host "[OK] Copied drone-only settings to Documents\AirSim\settings.json" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Workspace settings.json not found!" -ForegroundColor Red
    exit 1
}

# Verify SimMode is Multirotor
$settingsContent = Get-Content $docSettings -Raw
if ($settingsContent -match '"SimMode"\s*:\s*"Multirotor"') {
    Write-Host "[OK] SimMode is set to Multirotor (NO CARS)" -ForegroundColor Green
} else {
    Write-Host "[ERROR] SimMode is NOT Multirotor! Fixing..." -ForegroundColor Red
    # Fix it
    $settingsContent = $settingsContent -replace '"SimMode"\s*:\s*"[^"]*"', '"SimMode": "Multirotor"'
    Set-Content -Path $docSettings -Value $settingsContent -NoNewline
    Write-Host "[OK] Fixed SimMode to Multirotor" -ForegroundColor Green
}
Write-Host ""

# Step 3: Verify VehicleType is SimpleFlight (drone)
Write-Host "[3/4] Verifying VehicleType..." -ForegroundColor Yellow
if ($settingsContent -match '"VehicleType"\s*:\s*"SimpleFlight"') {
    Write-Host "[OK] VehicleType is SimpleFlight (DRONE)" -ForegroundColor Green
} else {
    Write-Host "[WARNING] VehicleType may not be SimpleFlight" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Wait a bit then launch
Write-Host "[4/4] Waiting 5 seconds before launch..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "SETTINGS VERIFIED - DRONE MODE ONLY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Now launching AirSim Blocks with DRONE mode..." -ForegroundColor Cyan
Write-Host ""

$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
if (Test-Path $blocksPath) {
    Start-Process -FilePath $blocksPath -WorkingDirectory (Split-Path $blocksPath)
    Write-Host "[OK] Blocks.exe launched!" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: Wait 2-5 minutes for AirSim to fully load" -ForegroundColor Yellow
    Write-Host "You should see a DRONE (quadcopter), NOT a car!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After it loads, run:" -ForegroundColor Cyan
    Write-Host "  .\run_keyboard_control.bat" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "[ERROR] Blocks.exe not found at: $blocksPath" -ForegroundColor Red
    exit 1
}

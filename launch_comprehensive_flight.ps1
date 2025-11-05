# Comprehensive Autonomous Flight Launcher
# Sets up sensor configuration and launches comprehensive flight script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "COMPREHENSIVE AUTONOMOUS FLIGHT LAUNCHER" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Kill existing AirSim processes
Write-Host "[1/4] Stopping existing AirSim processes..." -ForegroundColor Yellow
Get-Process -Name "AirSimNH","Blocks" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Setup comprehensive settings.json
Write-Host "[2/4] Setting up comprehensive sensor configuration..." -ForegroundColor Yellow
$settingsPath = "settings_comprehensive.json"
$targetLocations = @(
    "$PWD\settings.json",
    "$env:USERPROFILE\Documents\AirSim\settings.json",
    "$env:USERPROFILE\OneDrive\Documents\AirSim\settings.json"
)

if (Test-Path $settingsPath) {
    foreach ($target in $targetLocations) {
        $targetDir = Split-Path $target -Parent
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        Copy-Item -Path $settingsPath -Destination $target -Force
        Write-Host "  [OK] Settings copied to: $target" -ForegroundColor Green
    }
} else {
    Write-Host "  [WARNING] $settingsPath not found, using defaults" -ForegroundColor Yellow
}

# Launch AirSimNH
Write-Host "[3/4] Launching AirSimNH environment..." -ForegroundColor Yellow
$airsimPath = "AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64\AirSimNH.exe"

if (Test-Path $airsimPath) {
    $env:UE4_SKIP_SHADER_WARNINGS = "1"
    Start-Process -FilePath $airsimPath -WorkingDirectory (Split-Path $airsimPath)
    Write-Host "  [OK] AirSimNH launched!" -ForegroundColor Green
    Write-Host "  [INFO] Please wait 3-5 minutes for AirSimNH to fully load" -ForegroundColor Cyan
} else {
    Write-Host "  [ERROR] AirSimNH.exe not found at: $airsimPath" -ForegroundColor Red
    Write-Host "  [INFO] Please launch AirSimNH manually and continue" -ForegroundColor Yellow
}

# Wait and then launch flight script
Write-Host "[4/4] Ready to launch comprehensive flight script" -ForegroundColor Yellow
Write-Host ""
Write-Host "Wait for AirSimNH to fully load (3-5 minutes), then:" -ForegroundColor Cyan
Write-Host "  1. Run: venv\Scripts\python.exe autonomous_flight_comprehensive.py" -ForegroundColor White
Write-Host "  OR" -ForegroundColor Gray
Write-Host "  2. Run: .\run_comprehensive_flight.bat" -ForegroundColor White
Write-Host ""

# Launch Blocks Environment for Drone Simulation
# Blocks supports Multirotor (drone) mode by default

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Launching Blocks Environment for Drones" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$blocksExe = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64\Blocks.exe"
$workingDir = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Binaries\Win64"

if (-not (Test-Path $blocksExe)) {
    Write-Host "[ERROR] Blocks.exe not found!" -ForegroundColor Red
    Write-Host "Path: $blocksExe" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Blocks environment needs to be extracted properly." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Found Blocks.exe for drone simulation" -ForegroundColor Green
Write-Host ""
Write-Host "Launching Blocks (drone environment)..." -ForegroundColor Cyan

$proc = Start-Process -FilePath $blocksExe -WorkingDirectory $workingDir -WindowStyle Normal -PassThru

Write-Host "[OK] Launched! PID: $($proc.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "Blocks supports Multirotor (drone) mode by default." -ForegroundColor Cyan
Write-Host "Wait 2-5 minutes for it to fully load." -ForegroundColor Yellow
Write-Host ""
Write-Host "After Blocks loads, test with:" -ForegroundColor Cyan
Write-Host "  cd E:\Drone" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  python test_drone.py" -ForegroundColor Gray
Write-Host ""


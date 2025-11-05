# Simple AirSim Launcher
# This will launch the AirSimNH environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Launching AirSim Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$exePath = "E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Binaries\Win64\AirSimNH.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] AirSimNH.exe not found!" -ForegroundColor Red
    Write-Host "Path: $exePath" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Found AirSimNH.exe" -ForegroundColor Green
Write-Host ""
Write-Host "Launching..." -ForegroundColor Yellow

Start-Process -FilePath $exePath -WorkingDirectory (Split-Path $exePath)

Write-Host "[OK] Launch command sent!" -ForegroundColor Green
Write-Host ""
Write-Host "Please check your screen for the AirSim window." -ForegroundColor Cyan
Write-Host "Wait 2-5 minutes for it to fully load." -ForegroundColor Yellow
Write-Host ""
Write-Host "After it loads, run this to test:" -ForegroundColor Cyan
Write-Host "  cd E:\Drone" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  python test_airsim.py" -ForegroundColor Gray
Write-Host ""


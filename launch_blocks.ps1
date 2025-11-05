# Quick launch script for Blocks environment

Write-Host "Launching AirSim Blocks Environment..." -ForegroundColor Cyan
Write-Host ""

$blocksPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks.exe"

if (Test-Path $blocksPath) {
    Write-Host "[OK] Found Blocks.exe" -ForegroundColor Green
    Write-Host "Starting Blocks environment..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "NOTE: Keep this window open while using AirSim!" -ForegroundColor Yellow
    Write-Host "First launch may take 2-5 minutes to compile shaders." -ForegroundColor Gray
    Write-Host ""
    
    Start-Process -FilePath $blocksPath -WorkingDirectory "E:\Drone\AirSim\Blocks\WindowsNoEditor"
    
    Write-Host "Blocks is launching..." -ForegroundColor Green
    Write-Host ""
    Write-Host "After Blocks loads, run test in another terminal:" -ForegroundColor Cyan
    Write-Host "  cd E:\Drone" -ForegroundColor Gray
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host "  python test_airsim.py" -ForegroundColor Gray
} else {
    Write-Host "[ERROR] Blocks.exe not found at: $blocksPath" -ForegroundColor Red
    Write-Host "Please ensure Blocks.zip was extracted correctly." -ForegroundColor Yellow
}


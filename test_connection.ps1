# Quick connection test script
# Run this after Blocks.exe has fully loaded

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AirSim Connection Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Blocks is running
$blocksProcess = Get-Process -Name "Blocks" -ErrorAction SilentlyContinue
if (-not $blocksProcess) {
    Write-Host "[WARNING] Blocks.exe is not running!" -ForegroundColor Yellow
    Write-Host "Please launch Blocks first:" -ForegroundColor White
    Write-Host "  .\launch_blocks.ps1" -ForegroundColor Gray
    Write-Host "  OR" -ForegroundColor Gray
    Write-Host "  cd E:\Drone\AirSim\Blocks\WindowsNoEditor; .\Blocks.exe" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Write-Host "[OK] Blocks.exe is running" -ForegroundColor Green
Write-Host ""

# Activate virtual environment and test
Write-Host "Activating Python environment..." -ForegroundColor Yellow
cd E:\Drone
& ".\venv\Scripts\Activate.ps1"

Write-Host "Waiting 10 seconds for Blocks to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Testing AirSim connection..." -ForegroundColor Yellow
Write-Host ""

python test_airsim.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! AirSim is working correctly!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[WARNING] Connection test had issues." -ForegroundColor Yellow
    Write-Host "Make sure Blocks.exe has fully loaded (wait 1-2 minutes)." -ForegroundColor Yellow
}


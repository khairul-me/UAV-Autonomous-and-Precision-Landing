# Complete startup procedure for Windows PowerShell
# Pre-training system verification and demo flight

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "AIRSIM DRONE NAVIGATION STARTUP" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Check if AirSim is running
Write-Host ""
Write-Host "Checking AirSim connection..." -ForegroundColor Yellow

try {
    python -c "import airsim; c = airsim.MultirotorClient(); c.confirmConnection(); print('[OK] AirSim is running')" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] AirSim is not running!" -ForegroundColor Red
        Write-Host "Please start Unreal Engine with AirSim plugin first" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "[FAIL] AirSim is not running!" -ForegroundColor Red
    Write-Host "Please start Unreal Engine with AirSim plugin first" -ForegroundColor Yellow
    exit 1
}

# Run preflight checks
Write-Host ""
Write-Host "Running system verification..." -ForegroundColor Yellow
python preflight_check.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] System verification failed" -ForegroundColor Red
    exit 1
}

# Ask to run demo
Write-Host ""
$response = Read-Host "Run 2-minute demo flight? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    python demo_flight.py
}

# Ready for training
Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "SYSTEM READY FOR TRAINING" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  python quick_test.py" -ForegroundColor White
Write-Host "  python train_complete.py --mode baseline" -ForegroundColor White
Write-Host "  python train_complete.py --mode robust --enable-all-defenses" -ForegroundColor White
Write-Host "==================================" -ForegroundColor Green


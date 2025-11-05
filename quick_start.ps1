# AirSim Quick Start Script
# This script helps you get started with AirSim installation verification

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AirSim Quick Start Guide" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will guide you through:" -ForegroundColor Yellow
Write-Host "1. Checking prerequisites" -ForegroundColor White
Write-Host "2. Setting up Python environment" -ForegroundColor White
Write-Host "3. Verifying AirSim installation" -ForegroundColor White
Write-Host ""

# Check if AirSim directories exist
Write-Host "[CHECK] Verifying AirSim installation..." -ForegroundColor Yellow
$airsimPath = "E:\Drone\AirSim"
$blocksPath = "E:\Drone\AirSim\Blocks"

if (Test-Path $airsimPath) {
    Write-Host "✓ AirSim directory found: $airsimPath" -ForegroundColor Green
} else {
    Write-Host "✗ AirSim directory not found: $airsimPath" -ForegroundColor Red
    Write-Host "  Download from: https://github.com/microsoft/AirSim/releases" -ForegroundColor Gray
}

if (Test-Path $blocksPath) {
    if (Test-Path "$blocksPath\Blocks.exe") {
        Write-Host "✓ Blocks.exe found" -ForegroundColor Green
    } else {
        Write-Host "⚠ Blocks directory exists but Blocks.exe not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Blocks directory not found: $blocksPath" -ForegroundColor Red
    Write-Host "  Download Blocks.zip from AirSim releases page" -ForegroundColor Gray
}

Write-Host ""

# Check Python
Write-Host "[CHECK] Verifying Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Install Python 3.8+ from python.org" -ForegroundColor Red
}

Write-Host ""

# Check virtual environment
Write-Host "[CHECK] Verifying Python environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
    
    # Try to activate and check packages
    & ".\venv\Scripts\Activate.ps1"
    $packages = @("airsim", "torch", "cv2", "numpy")
    foreach ($pkg in $packages) {
        $result = python -c "import $pkg; print('OK')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $pkg installed" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $pkg NOT installed" -ForegroundColor Red
        }
    }
} else {
    Write-Host "⚠ Virtual environment not found" -ForegroundColor Yellow
    Write-Host "  Run: python -m venv venv" -ForegroundColor Gray
    Write-Host "  Then: .\setup_airsim.ps1" -ForegroundColor Gray
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. If Blocks.exe not found:" -ForegroundColor White
Write-Host "   Download from: https://github.com/microsoft/AirSim/releases" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Launch Blocks environment:" -ForegroundColor White
Write-Host "   cd E:\Drone\AirSim\Blocks" -ForegroundColor Gray
Write-Host "   .\Blocks.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "3. In a NEW terminal, test connection:" -ForegroundColor White
Write-Host "   cd E:\Drone" -ForegroundColor Gray
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "   python test_airsim.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. For detailed installation guide:" -ForegroundColor White
Write-Host "   See: INSTALLATION_WALKTHROUGH.md" -ForegroundColor Gray
Write-Host ""


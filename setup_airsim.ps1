# AirSim Setup Script for Windows PowerShell
# Run this script to automate AirSim setup (partial automation)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AirSim Installation Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
    
    # Check Python version
    $version = python -c "import sys; print(sys.version_info[1])" 2>&1
    if ([int]$version -lt 8) {
        Write-Host "✗ Python 3.8+ required. Found: Python 3.$version" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check CUDA
Write-Host ""
Write-Host "[2/4] Checking CUDA installation..." -ForegroundColor Yellow
try {
    $cudaVersion = nvcc --version 2>&1
    Write-Host "✓ CUDA found" -ForegroundColor Green
} catch {
    Write-Host "⚠ CUDA not found in PATH (optional, but recommended for GPU acceleration)" -ForegroundColor Yellow
}

# Create virtual environment
Write-Host ""
Write-Host "[3/4] Setting up Python virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host ""
Write-Host "[4/4] Installing Python dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Download AirSim pre-built binaries from:" -ForegroundColor White
Write-Host "   https://github.com/microsoft/AirSim/releases" -ForegroundColor Gray
Write-Host "2. Extract to: E:\Drone\AirSim" -ForegroundColor White
Write-Host "3. Download Blocks environment and extract to: E:\Drone\AirSim\Blocks" -ForegroundColor White
Write-Host "4. Launch Blocks.exe" -ForegroundColor White
Write-Host "5. Run: python test_airsim.py" -ForegroundColor White
Write-Host ""



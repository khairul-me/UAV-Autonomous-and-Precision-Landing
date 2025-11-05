# Automated AirSim Installation Script
# This script automates the installation process as much as possible

param(
    [switch]$SkipDownload,
    [switch]$SkipPython
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Automated AirSim Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# Configuration
$projectRoot = "E:\Drone"
$airsimPath = "$projectRoot\AirSim"
$blocksPath = "$projectRoot\AirSim\Blocks"

# Step 1: Check and setup directories
Write-Host "[1/7] Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $projectRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path $airsimPath -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Force -Path $blocksPath -ErrorAction SilentlyContinue | Out-Null
Write-Host "[OK] Directories created" -ForegroundColor Green

# Step 2: Check Python
Write-Host ""
Write-Host "[2/7] Checking Python installation..." -ForegroundColor Yellow
$pythonFound = $false
$pythonCmd = $null

# Check various Python installations
$pythonPaths = @("python", "python3", "py")

foreach ($cmd in $pythonPaths) {
    try {
        $null = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $version = & $cmd --version 2>&1
            if ($version -notmatch "Windows Store") {
                $pythonCmd = $cmd
                $pythonFound = $true
                Write-Host "[OK] Python found: $version" -ForegroundColor Green
                break
            }
        }
    } catch {
        continue
    }
}

# Try py launcher separately
if (-not $pythonFound) {
    try {
        $version = & py --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $version -notmatch "Windows Store") {
            $pythonCmd = "py"
            $pythonFound = $true
            Write-Host "[OK] Python found via py launcher: $version" -ForegroundColor Green
        }
    } catch {
        # Ignore
    }
}

if (-not $pythonFound) {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    if (-not $SkipPython) {
        Write-Host ""
        Write-Host "Opening Python download page..." -ForegroundColor Cyan
        Start-Process "https://www.python.org/downloads/"
    }
    Write-Host ""
    Write-Host "Installation cannot continue without Python." -ForegroundColor Red
    exit 1
}

# Step 3: Create virtual environment
Write-Host ""
Write-Host "[3/7] Setting up Python virtual environment..." -ForegroundColor Yellow
$venvPath = "$projectRoot\venv"
if (Test-Path $venvPath) {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
} else {
    try {
        if ($pythonCmd -eq "py") {
            & py -3 -m venv venv
        } else {
            & $pythonCmd -m venv venv
        }
        if (Test-Path $venvPath) {
            Write-Host "[OK] Virtual environment created" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "[ERROR] Failed to create virtual environment: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Activate and upgrade pip
Write-Host ""
Write-Host "[4/7] Installing/upgrading pip..." -ForegroundColor Yellow
try {
    & "$venvPath\Scripts\python.exe" -m pip install --upgrade pip --quiet 2>&1 | Out-Null
    Write-Host "[OK] Pip upgraded" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Pip upgrade had issues: $_" -ForegroundColor Yellow
}

# Step 5: Install Python dependencies
Write-Host ""
Write-Host "[5/7] Installing Python dependencies..." -ForegroundColor Yellow
try {
    $requirementsPath = "$projectRoot\requirements.txt"
    if (Test-Path $requirementsPath) {
        & "$venvPath\Scripts\python.exe" -m pip install -r $requirementsPath --quiet 2>&1 | Out-Null
        Write-Host "[OK] Requirements installed" -ForegroundColor Green
    } else {
        Write-Host "[INFO] requirements.txt not found, installing core packages..." -ForegroundColor Yellow
        & "$venvPath\Scripts\python.exe" -m pip install msgpack-rpc-python airsim torch torchvision opencv-python numpy --quiet 2>&1 | Out-Null
        Write-Host "[OK] Core packages installed" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] Package installation had issues: $_" -ForegroundColor Yellow
}

# Step 6: Download AirSim binaries
if (-not $SkipDownload) {
    Write-Host ""
    Write-Host "[6/7] Downloading AirSim binaries..." -ForegroundColor Yellow
    Write-Host "  Note: This requires manual download from GitHub releases" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Please download:" -ForegroundColor White
    Write-Host "  1. AirSim.zip from: https://github.com/microsoft/AirSim/releases/latest" -ForegroundColor Cyan
    Write-Host "  2. Blocks.zip from: https://github.com/microsoft/AirSim/releases/latest" -ForegroundColor Cyan
    Write-Host ""
    
    try {
        Write-Host "  Opening GitHub releases page in browser..." -ForegroundColor Cyan
        Start-Process "https://github.com/microsoft/AirSim/releases/latest"
    } catch {
        Write-Host "  Please visit: https://github.com/microsoft/AirSim/releases/latest" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "  After downloading, extract:" -ForegroundColor White
    Write-Host "  - AirSim.zip to: $airsimPath" -ForegroundColor Gray
    Write-Host "  - Blocks.zip to: $blocksPath" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  IMPORTANT: Extract the downloaded ZIP files!" -ForegroundColor Yellow
    Write-Host "  Then run this script again to verify installation." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "[6/7] Skipping download (user requested)" -ForegroundColor Yellow
}

# Step 7: Verify installation
Write-Host ""
Write-Host "[7/7] Verifying installation..." -ForegroundColor Yellow

$allGood = $true

# Check AirSim directory
if (Test-Path $airsimPath) {
    Write-Host "[OK] AirSim directory exists" -ForegroundColor Green
} else {
    Write-Host "[ERROR] AirSim directory not found" -ForegroundColor Red
    $allGood = $false
}

# Check Blocks executable
if (Test-Path "$blocksPath\Blocks.exe") {
    Write-Host "[OK] Blocks.exe found" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Blocks.exe not found (download and extract Blocks.zip)" -ForegroundColor Yellow
}

# Check Python packages
try {
    $airsimCheck = & "$venvPath\Scripts\python.exe" -c "import airsim; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] AirSim Python API installed" -ForegroundColor Green
    } else {
        Write-Host "[INFO] AirSim Python API not installed, installing now..." -ForegroundColor Yellow
        & "$venvPath\Scripts\python.exe" -m pip install msgpack-rpc-python airsim 2>&1 | Out-Null
        Write-Host "[OK] AirSim Python API installed" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] Could not verify AirSim Python API" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allGood) {
    Write-Host "[SUCCESS] Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. If Blocks.zip not extracted, extract it to: $blocksPath" -ForegroundColor White
    Write-Host "2. Launch Blocks: cd $blocksPath; .\Blocks.exe" -ForegroundColor White
    Write-Host "3. Test connection: cd $projectRoot; .\venv\Scripts\Activate.ps1; python test_airsim.py" -ForegroundColor White
} else {
    Write-Host "[INFO] Setup partially complete" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Remaining tasks:" -ForegroundColor Yellow
    Write-Host "1. Download AirSim binaries from GitHub releases" -ForegroundColor White
    Write-Host "2. Extract to appropriate directories" -ForegroundColor White
    Write-Host "3. Run this script again to verify" -ForegroundColor White
}

Write-Host ""

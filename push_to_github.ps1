# Push AirSim Drone Project to GitHub as New Branch
# Repository: https://github.com/khairul-me/UAV-Autonomous-and-Precision-Landing

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PUSHING TO GITHUB - AIRSIM BRANCH" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version 2>&1
    Write-Host "[OK] Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed or not in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "After installation, restart PowerShell and run this script again." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Branch name
$branchName = "airsim-comprehensive-flight"
Write-Host "Branch name: $branchName" -ForegroundColor Cyan
Write-Host ""

# Step 1: Initialize git if needed
if (-not (Test-Path ".git")) {
    Write-Host "[1/7] Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "[OK] Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "[1/7] Git repository already exists" -ForegroundColor Green
}

# Step 2: Add remote
Write-Host ""
Write-Host "[2/7] Setting up remote repository..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/khairul-me/UAV-Autonomous-and-Precision-Landing.git"

$existingRemote = git remote get-url origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[INFO] Remote 'origin' already exists: $existingRemote" -ForegroundColor Gray
    $updateRemote = Read-Host "Update remote URL? (y/n)"
    if ($updateRemote -eq "y" -or $updateRemote -eq "Y") {
        git remote set-url origin $remoteUrl
        Write-Host "[OK] Remote URL updated" -ForegroundColor Green
    }
} else {
    git remote add origin $remoteUrl
    Write-Host "[OK] Remote 'origin' added" -ForegroundColor Green
}

# Step 3: Create or switch to branch
Write-Host ""
Write-Host "[3/7] Creating/checking out branch: $branchName..." -ForegroundColor Yellow
$currentBranch = git branch --show-current 2>&1
if ($LASTEXITCODE -eq 0 -and $currentBranch -eq $branchName) {
    Write-Host "[OK] Already on branch: $branchName" -ForegroundColor Green
} else {
    git checkout -b $branchName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Switched to branch: $branchName" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Branch creation returned: $LASTEXITCODE" -ForegroundColor Yellow
        # Try to switch if branch exists
        git checkout $branchName 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Switched to existing branch: $branchName" -ForegroundColor Green
        }
    }
}

# Step 4: Add all files
Write-Host ""
Write-Host "[4/7] Adding files to git..." -ForegroundColor Yellow
git add .
Write-Host "[OK] Files staged" -ForegroundColor Green

# Show status
Write-Host ""
Write-Host "Files to be committed:" -ForegroundColor Cyan
git status --short

# Step 5: Commit
Write-Host ""
Write-Host "[5/7] Creating commit..." -ForegroundColor Yellow
$commitMessage = "Add AirSim comprehensive autonomous flight system - Phase 0 complete

- Complete Phase 0 implementation (Tasks 0.1, 0.2, 0.3)
- Multi-sensor capture (RGB, Depth, Segmentation, IMU, GPS, Magnetometer, Barometer)
- Comprehensive data logging system
- Organized directory structure for flight recordings
- Configurable capture rates (1-60 Hz)
- Urban exploration flight patterns
- Real-time video recording with overlays
- Complete documentation and launcher scripts

Features:
- autonomous_flight_comprehensive.py: Main flight script with all sensors
- settings_comprehensive.json: Full sensor configuration
- Data logger with synchronized timestamps
- Flight recordings with video and complete sensor data"

git commit -m $commitMessage
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Commit created" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Commit may have failed or no changes to commit" -ForegroundColor Yellow
}

# Step 6: Fetch latest from remote
Write-Host ""
Write-Host "[6/7] Fetching latest from remote..." -ForegroundColor Yellow
git fetch origin 2>&1 | Out-Null
Write-Host "[OK] Remote fetched" -ForegroundColor Green

# Step 7: Push to GitHub
Write-Host ""
Write-Host "[7/7] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "Pushing branch '$branchName' to origin..." -ForegroundColor Cyan

git push -u origin $branchName

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! Pushed to GitHub" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Branch URL:" -ForegroundColor Cyan
    Write-Host "https://github.com/khairul-me/UAV-Autonomous-and-Precision-Landing/tree/$branchName" -ForegroundColor White
    Write-Host ""
    Write-Host "You can now:" -ForegroundColor Cyan
    Write-Host "1. View the branch on GitHub" -ForegroundColor White
    Write-Host "2. Create a Pull Request to merge into main" -ForegroundColor White
    Write-Host "3. Continue development on this branch" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Push failed. Common issues:" -ForegroundColor Red
    Write-Host "1. Authentication required - GitHub may prompt for credentials" -ForegroundColor Yellow
    Write-Host "2. Branch may already exist - try: git push -u origin $branchName --force" -ForegroundColor Yellow
    Write-Host "3. Check your internet connection" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can manually push using:" -ForegroundColor Cyan
    Write-Host "  git push -u origin $branchName" -ForegroundColor White
    Write-Host ""
}

# AirSim Download Helper Script
# This script helps download AirSim binaries from GitHub releases

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AirSim Download Helper" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = "E:\Drone"
$airsimPath = "$projectRoot\AirSim"
$blocksPath = "$projectRoot\AirSim\Blocks"
$downloadDir = "$env:USERPROFILE\Downloads"

Write-Host "This script will help you download AirSim binaries." -ForegroundColor Yellow
Write-Host ""

# Check if directories exist
if (-not (Test-Path $airsimPath)) {
    Write-Host "Creating directories..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $airsimPath | Out-Null
    New-Item -ItemType Directory -Force -Path $blocksPath | Out-Null
}

Write-Host "AirSim download locations:" -ForegroundColor White
Write-Host "  Main: https://github.com/microsoft/AirSim/releases/latest" -ForegroundColor Cyan
Write-Host ""

Write-Host "Files to download:" -ForegroundColor White
Write-Host "  1. AirSim.zip (or AirSim-Windows.zip)" -ForegroundColor Yellow
Write-Host "  2. Blocks.zip" -ForegroundColor Yellow
Write-Host ""

# Try to get latest release info
Write-Host "Fetching latest release information..." -ForegroundColor Gray
try {
    $releaseUrl = "https://github.com/microsoft/AirSim/releases/latest"
    $response = Invoke-WebRequest -Uri $releaseUrl -UseBasicParsing -MaximumRedirection 0 -ErrorAction SilentlyContinue
    
    if ($response.StatusCode -eq 302 -or $response.StatusCode -eq 301) {
        $actualReleaseUrl = $response.Headers.Location
        Write-Host "[OK] Latest release found" -ForegroundColor Green
        Write-Host "  Release URL: $actualReleaseUrl" -ForegroundColor Gray
    }
} catch {
    Write-Host "[INFO] Could not fetch release info automatically" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Opening GitHub releases page in your browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 1
Start-Process "https://github.com/microsoft/AirSim/releases/latest"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Download Instructions" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "On the GitHub page:" -ForegroundColor White
Write-Host "  1. Scroll down to 'Assets' section" -ForegroundColor Gray
Write-Host "  2. Download 'AirSim.zip' (or 'AirSim-Windows.zip')" -ForegroundColor Gray
Write-Host "  3. Download 'Blocks.zip'" -ForegroundColor Gray
Write-Host ""
Write-Host "After downloading:" -ForegroundColor White
Write-Host "  Extract AirSim.zip to: $airsimPath" -ForegroundColor Yellow
Write-Host "  Extract Blocks.zip to: $blocksPath" -ForegroundColor Yellow
Write-Host ""

# Check Downloads folder for existing files
Write-Host "Checking Downloads folder for existing files..." -ForegroundColor Gray
$downloadedFiles = Get-ChildItem -Path $downloadDir -Filter "*AirSim*.zip" -ErrorAction SilentlyContinue
$blockFiles = Get-ChildItem -Path $downloadDir -Filter "*Blocks*.zip" -ErrorAction SilentlyContinue

if ($downloadedFiles) {
    Write-Host "[FOUND] AirSim ZIP files in Downloads:" -ForegroundColor Green
    foreach ($file in $downloadedFiles) {
        Write-Host "  - $($file.Name) ($([math]::Round($file.Length/1MB, 2)) MB)" -ForegroundColor Gray
    }
}

if ($blockFiles) {
    Write-Host "[FOUND] Blocks ZIP files in Downloads:" -ForegroundColor Green
    foreach ($file in $blockFiles) {
        Write-Host "  - $file.Name ($([math]::Round($file.Length/1MB, 2)) MB)" -ForegroundColor Gray
    }
}

if ($downloadedFiles -or $blockFiles) {
    Write-Host ""
    $extract = Read-Host "Do you want to extract these files now? (Y/N)"
    if ($extract -eq 'Y' -or $extract -eq 'y') {
        Write-Host ""
        Write-Host "Extracting files..." -ForegroundColor Yellow
        
        # Extract AirSim.zip
        foreach ($file in $downloadedFiles) {
            if ($file.Name -match "Blocks") {
                Write-Host "  Extracting $($file.Name) to $blocksPath..." -ForegroundColor Gray
                Expand-Archive -Path $file.FullName -DestinationPath $blocksPath -Force -ErrorAction SilentlyContinue
            } else {
                Write-Host "  Extracting $($file.Name) to $airsimPath..." -ForegroundColor Gray
                Expand-Archive -Path $file.FullName -DestinationPath $airsimPath -Force -ErrorAction SilentlyContinue
            }
        }
        
        # Extract Blocks.zip
        foreach ($file in $blockFiles) {
            Write-Host "  Extracting $($file.Name) to $blocksPath..." -ForegroundColor Gray
            Expand-Archive -Path $file.FullName -DestinationPath $blocksPath -Force -ErrorAction SilentlyContinue
        }
        
        Write-Host "[OK] Extraction complete!" -ForegroundColor Green
        
        # Verify
        Write-Host ""
        Write-Host "Verifying extraction..." -ForegroundColor Yellow
        if (Test-Path "$blocksPath\Blocks.exe") {
            Write-Host "[OK] Blocks.exe found!" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Blocks.exe not found. Check extraction." -ForegroundColor Yellow
        }
        
        if (Test-Path "$airsimPath\PythonClient") {
            Write-Host "[OK] AirSim PythonClient found!" -ForegroundColor Green
        } else {
            Write-Host "[INFO] AirSim structure may vary. Check $airsimPath" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "After extraction, run:" -ForegroundColor White
Write-Host "  .\install_airsim.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then launch Blocks:" -ForegroundColor White
Write-Host "  cd E:\Drone\AirSim\Blocks" -ForegroundColor Gray
Write-Host "  .\Blocks.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "And test:" -ForegroundColor White
Write-Host "  cd E:\Drone" -ForegroundColor Gray
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  python test_airsim.py" -ForegroundColor Gray
Write-Host ""


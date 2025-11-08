# AirSim Plugin Verification Script
# Checks if AirSim plugin is correctly installed in Blocks

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AIRSIM PLUGIN VERIFICATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$blocksPluginPath = "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim"
$airsimnhPluginPath = "E:\Drone\AirSim\AirSimNH\AirSimNH\WindowsNoEditor\AirSimNH\Plugins\AirSim"

Write-Host "[1] Checking Blocks plugin directory..." -ForegroundColor Yellow

if (Test-Path $blocksPluginPath) {
    Write-Host "[OK] Plugin directory exists" -ForegroundColor Green
    Write-Host "  Path: $blocksPluginPath" -ForegroundColor Gray
    
    # Check critical files
    $criticalFiles = @(
        @{Path = "AirSim.uplugin"; Name = "Plugin Descriptor"},
        @{Path = "Binaries\Win64\AirSim.dll"; Name = "Main Plugin DLL"},
        @{Path = "Binaries\Win64\AirSim.lib"; Name = "Plugin Library"}
    )
    
    Write-Host ""
    Write-Host "Checking critical files..." -ForegroundColor Cyan
    $allPresent = $true
    
    foreach ($file in $criticalFiles) {
        $fullPath = Join-Path $blocksPluginPath $file.Path
        if (Test-Path $fullPath) {
            $fileInfo = Get-Item $fullPath
            $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
            Write-Host "  [OK] $($file.Name)" -ForegroundColor Green
            Write-Host "    Path: $($file.Path)" -ForegroundColor Gray
            Write-Host "    Size: $sizeMB MB" -ForegroundColor Gray
            
            if ($fileInfo.Length -eq 0) {
                Write-Host "    [ERROR] File is empty!" -ForegroundColor Red
                $allPresent = $false
            }
        } else {
            Write-Host "  [ERROR] $($file.Name) MISSING" -ForegroundColor Red
            Write-Host "    Expected: $fullPath" -ForegroundColor Gray
            $allPresent = $false
        }
    }
    
    if (-not $allPresent) {
        Write-Host ""
        Write-Host "[WARNING] Some plugin files are missing!" -ForegroundColor Yellow
    }
    
} else {
    Write-Host "[ERROR] Plugin directory does not exist!" -ForegroundColor Red
    Write-Host "  Expected: $blocksPluginPath" -ForegroundColor Gray
    Write-Host ""
    Write-Host "The AirSim plugin is not installed in Blocks." -ForegroundColor Yellow
    Write-Host "This explains why the API server isn't starting." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[2] Checking AirSimNH plugin (source for comparison)..." -ForegroundColor Yellow

if (Test-Path $airsimnhPluginPath) {
    Write-Host "[OK] AirSimNH plugin directory exists" -ForegroundColor Green
    
    # Compare DLLs
    $blocksDll = Join-Path $blocksPluginPath "Binaries\Win64\AirSim.dll"
    $airsimnhDll = Join-Path $airsimnhPluginPath "Binaries\Win64\AirSim.dll"
    
    if ((Test-Path $blocksDll) -and (Test-Path $airsimnhDll)) {
        $blocksDllInfo = Get-Item $blocksDll
        $airsimnhDllInfo = Get-Item $airsimnhDll
        
        Write-Host ""
        Write-Host "Comparing DLLs:" -ForegroundColor Cyan
        Write-Host "  Blocks DLL: $([math]::Round($blocksDllInfo.Length / 1MB, 2)) MB" -ForegroundColor Gray
        Write-Host "  AirSimNH DLL: $([math]::Round($airsimnhDllInfo.Length / 1MB, 2)) MB" -ForegroundColor Gray
        
        if ($blocksDllInfo.Length -eq $airsimnhDllInfo.Length) {
            Write-Host "  [OK] DLLs are the same size" -ForegroundColor Green
            
            # Check if they're the same file
            $blocksHash = (Get-FileHash $blocksDll -Algorithm MD5).Hash
            $airsimnhHash = (Get-FileHash $airsimnhDll -Algorithm MD5).Hash
            
            if ($blocksHash -eq $airsimnhHash) {
                Write-Host "  [OK] DLLs are identical (same hash)" -ForegroundColor Green
            } else {
                Write-Host "  [INFO] DLLs have different hashes (may be different versions)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  [WARNING] DLLs have different sizes" -ForegroundColor Yellow
            Write-Host "  This may indicate a version mismatch" -ForegroundColor Yellow
        }
    }
    
    # Check plugin descriptor
    $blocksUplugin = Join-Path $blocksPluginPath "AirSim.uplugin"
    $airsimnhUplugin = Join-Path $airsimnhPluginPath "AirSim.uplugin"
    
    if ((Test-Path $blocksUplugin) -and (Test-Path $airsimnhUplugin)) {
        Write-Host ""
        Write-Host "Comparing plugin descriptors..." -ForegroundColor Cyan
        try {
            $blocksJson = Get-Content $blocksUplugin -Raw | ConvertFrom-Json
            $airsimnhJson = Get-Content $airsimnhUplugin -Raw | ConvertFrom-Json
            
            Write-Host "  Blocks EngineVersion: $($blocksJson.EngineVersion)" -ForegroundColor Gray
            Write-Host "  AirSimNH EngineVersion: $($airsimnhJson.EngineVersion)" -ForegroundColor Gray
            
            if ($blocksJson.EngineVersion -eq $airsimnhJson.EngineVersion) {
                Write-Host "  [OK] Engine versions match" -ForegroundColor Green
            } else {
                Write-Host "  [WARNING] Engine versions differ" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  [WARNING] Could not parse plugin descriptors" -ForegroundColor Yellow
        }
    }
    
} else {
    Write-Host "[WARNING] AirSimNH plugin directory not found" -ForegroundColor Yellow
    Write-Host "  Cannot compare with source" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[3] Checking plugin structure..." -ForegroundColor Yellow

$expectedStructure = @(
    "Binaries",
    "Binaries\Win64",
    "Content"
)

$structureOK = $true
foreach ($dir in $expectedStructure) {
    $fullPath = Join-Path $blocksPluginPath $dir
    if (Test-Path $fullPath) {
        Write-Host "  [OK] $dir\" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] $dir\ missing" -ForegroundColor Yellow
        $structureOK = $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path $blocksPluginPath) {
    $dllPath = Join-Path $blocksPluginPath "Binaries\Win64\AirSim.dll"
    if (Test-Path $dllPath) {
        Write-Host "[OK] AirSim plugin appears to be installed" -ForegroundColor Green
        Write-Host ""
        Write-Host "If API server still doesn't start:" -ForegroundColor Yellow
        Write-Host "1. Verify Blocks is using this plugin" -ForegroundColor Gray
        Write-Host "2. Check Unreal Engine logs for plugin loading errors" -ForegroundColor Gray
        Write-Host "3. Try restarting Blocks" -ForegroundColor Gray
        Write-Host "4. Consider rebuilding plugin from source" -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Plugin DLL is missing!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Action required:" -ForegroundColor Yellow
        Write-Host "1. Copy plugin from AirSimNH to Blocks" -ForegroundColor Gray
        Write-Host "2. Or rebuild AirSim from source" -ForegroundColor Gray
    }
} else {
    Write-Host "[ERROR] Plugin directory does not exist!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Action required:" -ForegroundColor Yellow
    Write-Host "1. Copy plugin from AirSimNH:" -ForegroundColor Gray
    Write-Host "   From: $airsimnhPluginPath" -ForegroundColor Cyan
    Write-Host "   To: $blocksPluginPath" -ForegroundColor Cyan
}

Write-Host ""


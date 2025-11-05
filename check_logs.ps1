# AirSim Log File Diagnostic Script
# Checks Unreal Engine logs for AirSim plugin loading messages

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AIRSIM LOG FILE DIAGNOSTIC" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check multiple possible log locations
$logPaths = @(
    "$env:LOCALAPPDATA\AirSim\Blocks\Saved\Logs",
    "E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Saved\Logs",
    "$env:LOCALAPPDATA\AirSim\Saved\Logs"
)

$logFound = $false

foreach ($logPath in $logPaths) {
    if (Test-Path $logPath) {
        Write-Host "[OK] Log directory found: $logPath" -ForegroundColor Green
        $logFiles = Get-ChildItem -Path $logPath -Filter "*.log" | Sort-Object LastWriteTime -Descending
        
        if ($logFiles.Count -gt 0) {
            Write-Host "  Found $($logFiles.Count) log file(s)" -ForegroundColor Gray
            Write-Host ""
            
            # Check most recent log
            $latestLog = $logFiles[0]
            Write-Host "Latest log: $($latestLog.Name)" -ForegroundColor Yellow
            Write-Host "  Size: $([math]::Round($latestLog.Length / 1KB, 2)) KB" -ForegroundColor Gray
            Write-Host "  Modified: $($latestLog.LastWriteTime)" -ForegroundColor Gray
            Write-Host ""
            
            # Read log and search for AirSim messages
            Write-Host "Searching for AirSim messages..." -ForegroundColor Cyan
            $content = Get-Content -Path $latestLog.FullName -ErrorAction SilentlyContinue | Where-Object { $_ -match "AirSim" }
            
            if ($content) {
                Write-Host "[OK] Found AirSim messages in log" -ForegroundColor Green
                Write-Host ""
                Write-Host "Recent AirSim messages:" -ForegroundColor Yellow
                $content[-10..-1] | ForEach-Object {
                    if ($_ -match "error|Error|ERROR|fail|Fail|FAIL") {
                        Write-Host "  $_" -ForegroundColor Red
                    } elseif ($_ -match "AirSim.*ready|API.*server|plugin.*load") {
                        Write-Host "  $_" -ForegroundColor Green
                    } else {
                        Write-Host "  $_" -ForegroundColor Gray
                    }
                }
            } else {
                Write-Host "[WARNING] No AirSim messages found in log" -ForegroundColor Yellow
                Write-Host "  This suggests the AirSim plugin may not be loading"
            }
            
            Write-Host ""
            Write-Host "Searching for plugin loading messages..." -ForegroundColor Cyan
            $pluginContent = Get-Content -Path $latestLog.FullName -ErrorAction SilentlyContinue | Where-Object { 
                $_ -match "plugin|Plugin|PLUGIN|AirSim|module|Module" 
            }
            
            if ($pluginContent) {
                Write-Host "Plugin-related messages:" -ForegroundColor Yellow
                $pluginContent[-10..-1] | ForEach-Object {
                    if ($_ -match "error|Error|ERROR|fail|Fail|FAIL|missing|Missing|MISSING") {
                        Write-Host "  $_" -ForegroundColor Red
                    } else {
                        Write-Host "  $_" -ForegroundColor Gray
                    }
                }
            }
            
            Write-Host ""
            Write-Host "Searching for network/port messages..." -ForegroundColor Cyan
            $networkContent = Get-Content -Path $latestLog.FullName -ErrorAction SilentlyContinue | Where-Object { 
                $_ -match "41451|port|Port|PORT|network|Network|socket|Socket" 
            }
            
            if ($networkContent) {
                Write-Host "Network-related messages:" -ForegroundColor Yellow
                $networkContent[-10..-1] | ForEach-Object {
                    Write-Host "  $_" -ForegroundColor Gray
                }
            } else {
                Write-Host "[WARNING] No network/port messages found" -ForegroundColor Yellow
            }
            
            $logFound = $true
            break
        } else {
            Write-Host "[WARNING] No log files found in directory" -ForegroundColor Yellow
        }
    }
}

if (-not $logFound) {
    Write-Host "[ERROR] No log directories found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible reasons:" -ForegroundColor Yellow
    Write-Host "1. Blocks hasn't been run yet" -ForegroundColor Gray
    Write-Host "2. Logs are in a different location" -ForegroundColor Gray
    Write-Host "3. Logging is disabled" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Try running Blocks.exe first, then check again." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "To view full log file:" -ForegroundColor Cyan
if ($logFound) {
    Write-Host "  notepad `"$($latestLog.FullName)`"" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan


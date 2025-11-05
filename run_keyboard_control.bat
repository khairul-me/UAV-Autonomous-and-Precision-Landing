@echo off
REM Simple batch file to run keyboard_control.py
REM This bypasses PowerShell execution policy issues

echo ========================================
echo Starting Keyboard Control for Drone
echo ========================================
echo.

REM Use Python from venv directly (no activation needed)
cd /d E:\Drone
"venv\Scripts\python.exe" keyboard_control.py

pause

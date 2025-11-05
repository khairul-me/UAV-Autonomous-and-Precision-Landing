@echo off
echo ========================================
echo QUICK START - DRONE KEYBOARD CONTROL
echo ========================================
echo.
echo Make sure Blocks.exe is running!
echo.
echo This script will:
echo   1. Connect to AirSim
echo   2. Wait for you to press [C] to claim control
echo   3. Then you can fly with keyboard
echo.
pause
cd /d E:\Drone
"venv\Scripts\python.exe" keyboard_control.py
pause

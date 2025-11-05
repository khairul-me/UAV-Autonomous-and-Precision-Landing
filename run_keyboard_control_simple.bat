@echo off
echo ========================================
echo DRONE KEYBOARD CONTROL
echo ========================================
echo.
echo Make sure Blocks.exe is running first!
echo Wait 2-5 minutes after launching Blocks.
echo.
echo This will connect to AirSim and start keyboard control.
echo.
pause
cd /d E:\Drone
"venv\Scripts\python.exe" keyboard_control.py
pause

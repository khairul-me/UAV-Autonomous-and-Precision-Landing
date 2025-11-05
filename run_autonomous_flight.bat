@echo off
echo ========================================
echo AUTONOMOUS DRONE FLIGHT
echo ========================================
echo.
echo This will:
echo   1. Connect to AirSim
echo   2. Fly autonomously in a pattern
echo   3. Record the flight
echo   4. Save video and data
echo.
echo Make sure Blocks.exe is running first!
echo.
pause
cd /d E:\Drone
"venv\Scripts\python.exe" autonomous_flight.py
pause

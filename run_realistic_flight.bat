@echo off
echo ========================================
echo AUTONOMOUS FLIGHT - REALISTIC ENVIRONMENT
echo ========================================
echo.
echo This will:
echo   1. Connect to AirSimNH
echo   2. Fly autonomously in urban environment
echo   3. Record the flight
echo   4. Save video and data
echo.
echo IMPORTANT: Make sure AirSimNH.exe is running first!
echo Wait 2-5 minutes after launching for it to fully load.
echo.
echo EXIT: Press Ctrl+C at any time to exit safely
echo.
pause
cd /d E:\Drone
"venv\Scripts\python.exe" autonomous_flight_realistic.py
pause

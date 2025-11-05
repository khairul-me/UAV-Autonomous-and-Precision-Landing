@echo off
echo ========================================
echo COMPREHENSIVE AUTONOMOUS FLIGHT
echo ========================================
echo.
echo This script implements ALL Phase 0 requirements:
echo   - Task 0.1: Basic flight commands
echo   - Task 0.2: Multi-sensor capture
echo   - Task 0.3: Comprehensive data logging
echo.
echo FEATURES:
echo   - RGB, Depth, Segmentation images
echo   - IMU, GPS, Magnetometer, Barometer
echo   - Organized data directory structure
echo   - Synchronized timestamps
echo.
echo IMPORTANT: Make sure AirSimNH.exe is running first!
echo Wait 3-5 minutes after launching for it to fully load.
echo.
echo EXIT: Press Ctrl+C at any time to exit safely
echo.
pause
cd /d E:\Drone
"venv\Scripts\python.exe" autonomous_flight_comprehensive.py
pause

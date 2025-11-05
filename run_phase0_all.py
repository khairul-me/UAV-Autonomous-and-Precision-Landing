"""
Phase 0: AirSim Foundation Setup - Complete Test Suite
Runs all Phase 0 tasks sequentially:
- Task 0.1: Fix Current AirSim Issues (MAKE_IT_FLY.py)
- Task 0.2: Environment Preparation
- Task 0.3: Data Pipeline Setup

This script verifies that all Phase 0 success criteria are met.
"""

import subprocess
import sys
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_status(message, status="INFO"):
    """Print formatted status"""
    symbols = {"OK": "[OK]", "ERROR": "[ERROR]", "INFO": "[INFO]", "SUCCESS": "[SUCCESS]", "WARNING": "[WARNING]"}
    print(f"{symbols.get(status, '[INFO]')} {message}")

def check_blocks_running():
    """Check if Blocks.exe or AirSimNH.exe is running"""
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Blocks.exe'], 
                              capture_output=True, text=True, timeout=5)
        if 'Blocks.exe' in result.stdout:
            return True
        
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq AirSimNH.exe'], 
                              capture_output=True, text=True, timeout=5)
        if 'AirSimNH.exe' in result.stdout:
            return True
        
        return False
    except:
        return False

def check_port_listening():
    """Check if port 41451 is listening"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 41451))
        sock.close()
        return result == 0
    except:
        return False

def run_task(script_name, task_name):
    """
    Run a Phase 0 task script
    
    Args:
        script_name: Name of the Python script to run
        task_name: Human-readable task name
    
    Returns:
        True if successful, False otherwise
    """
    print_header(f"Running {task_name}")
    
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print_status(f"Script not found: {script_name}", "ERROR")
        return False
    
    print_status(f"Executing: {script_name}", "INFO")
    print_status(f"Path: {script_path}", "INFO")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print_status(f"{task_name} completed successfully", "SUCCESS")
            return True
        else:
            print_status(f"{task_name} failed with exit code {result.returncode}", "ERROR")
            return False
            
    except subprocess.TimeoutExpired:
        print_status(f"{task_name} timed out after 10 minutes", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error running {task_name}: {e}", "ERROR")
        return False

def main():
    """Run all Phase 0 tasks"""
    print("=" * 70)
    print("  PHASE 0: AIRSIM FOUNDATION SETUP - COMPLETE TEST SUITE")
    print("=" * 70)
    print("\nThis script will run all Phase 0 tasks sequentially:")
    print("  1. Task 0.1: Fix Current AirSim Issues (MAKE_IT_FLY.py)")
    print("  2. Task 0.2: Environment Preparation")
    print("  3. Task 0.3: Data Pipeline Setup")
    print()
    
    # Pre-flight checks
    print_header("Pre-Flight Checks")
    
    print_status("Checking if AirSim environment is running...", "INFO")
    if check_blocks_running():
        print_status("Blocks.exe or AirSimNH.exe is running", "OK")
    else:
        print_status("Blocks.exe or AirSimNH.exe is NOT running", "WARNING")
        print_status("Please start Blocks.exe or AirSimNH.exe before running this script", "WARNING")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print_status("Aborted by user", "INFO")
            return False
    
    print_status("Checking if API server is listening on port 41451...", "INFO")
    if check_port_listening():
        print_status("Port 41451 is listening", "OK")
    else:
        print_status("Port 41451 is NOT listening", "WARNING")
        print_status("API server may not be ready. Please ensure AirSim is fully loaded", "WARNING")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print_status("Aborted by user", "INFO")
            return False
    
    # Run tasks
    tasks = [
        ("MAKE_IT_FLY.py", "Task 0.1: Make It Fly"),
        ("phase0_task02_environment_setup.py", "Task 0.2: Environment Preparation"),
        ("phase0_task03_data_pipeline.py", "Task 0.3: Data Pipeline Setup")
    ]
    
    results = {}
    
    for script_name, task_name in tasks:
        success = run_task(script_name, task_name)
        results[task_name] = success
        
        if not success:
            print_status(f"{task_name} failed. Review the output above for details.", "ERROR")
            response = input(f"\nContinue with remaining tasks? (y/n): ")
            if response.lower() != 'y':
                break
        
        # Brief pause between tasks
        if task_name != tasks[-1][1]:
            print("\n" + "-" * 70)
            print("Pausing 3 seconds before next task...")
            time.sleep(3)
    
    # Summary
    print_header("Phase 0 Test Summary")
    
    print("Task Results:")
    for task_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"  {symbol} {task_name}: {status}")
    
    all_success = all(results.values())
    
    print("\n" + "=" * 70)
    if all_success:
        print_status("PHASE 0: ALL TASKS COMPLETED SUCCESSFULLY!", "SUCCESS")
        print("=" * 70)
        print("\nSuccess Criteria Met:")
        print("  ✓ Task 0.1: MAKE_IT_FLY.py successfully executes full flight sequence")
        print("  ✓ Task 0.2: Can capture synchronized RGB+Depth+IMU+GPS data at 30Hz")
        print("  ✓ Task 0.3: Can record and replay entire flight sessions with all sensor data")
        print("\n" + "=" * 70)
        return True
    else:
        print_status("PHASE 0: SOME TASKS FAILED", "ERROR")
        print("=" * 70)
        print("\nPlease review the output above and fix any issues before proceeding.")
        print("\n" + "=" * 70)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

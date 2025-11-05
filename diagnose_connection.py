"""
Comprehensive AirSim API Connection Diagnostic Script
Run this to systematically diagnose connection issues
"""

import airsim
import sys
import socket
import subprocess
import os
import time
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_python_version():
    print_section("1. PYTHON VERSION CHECK")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("[OK] Python version is compatible with AirSim 1.8.1")
        return True
    else:
        print("[WARNING] AirSim 1.8.1 requires Python 3.8+")
        return False

def check_airsim_import():
    print_section("2. AIRSIM MODULE CHECK")
    try:
        import airsim
        print("[OK] AirSim module imported successfully")
        print(f"    AirSim version: {airsim.__version__ if hasattr(airsim, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"[ERROR] Cannot import AirSim: {e}")
        print("    Install with: pip install airsim")
        return False

def check_port_availability(port=41451):
    print_section("3. PORT 41451 AVAILABILITY CHECK")
    try:
        # Check if port is listening
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"[OK] Port {port} is LISTENING and accepting connections")
            return True
        else:
            print(f"[ERROR] Port {port} is NOT listening")
            print("    This means AirSim API server is not running")
            print("    Possible causes:")
            print("    - Blocks hasn't fully initialized (wait 2-3 minutes)")
            print("    - AirSim plugin not loaded")
            print("    - Plugin failed to start API server")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking port: {e}")
        return False

def check_blocks_process():
    print_section("4. BLOCKS PROCESS CHECK")
    try:
        # Check if Blocks.exe is running
        try:
            import psutil
            use_psutil = True
        except ImportError:
            use_psutil = False
            print("[SKIP] psutil not installed (install with: pip install psutil)")
            print("    Checking with basic methods...")
        
        if use_psutil:
            blocks_found = False
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    if proc.info['exe'] and 'Blocks' in proc.info['exe']:
                        blocks_found = True
                        print(f"[OK] Blocks process found")
                        print(f"    PID: {proc.info['pid']}")
                        print(f"    Path: {proc.info['exe']}")
                        
                        # Check if it's the correct Blocks.exe
                        if 'AirSim\\Blocks' in proc.info['exe']:
                            print("    [OK] Correct Blocks.exe (drone environment)")
                        else:
                            print("    [WARNING] May not be the correct Blocks.exe")
                        
                        # Check network connections
                        try:
                            connections = proc.connections()
                            port_41451_conn = [c for c in connections if c.laddr.port == 41451]
                            if port_41451_conn:
                                print(f"    [OK] Process has connection on port 41451")
                            else:
                                print(f"    [WARNING] Process has no connection on port 41451")
                        except:
                            pass
                        
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not blocks_found:
                print("[ERROR] Blocks.exe is NOT running")
                print("    Launch Blocks.exe before running this diagnostic")
                return False
            
            return True
        else:
            # Fallback: use basic process check
            try:
                result = subprocess.run(
                    ["powershell", "-Command", "Get-Process | Where-Object { $_.Path -like '*Block*' } | Select-Object -First 1"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.stdout and 'Blocks' in result.stdout:
                    print("[OK] Blocks process appears to be running")
                    return True
                else:
                    print("[ERROR] Blocks.exe is NOT running")
                    print("    Launch Blocks.exe before running this diagnostic")
                    return False
            except:
                print("[SKIP] Cannot check process (install psutil for better diagnostics)")
                return None
    except ImportError:
        print("[SKIP] psutil not installed (install with: pip install psutil)")
        return None
    except Exception as e:
        print(f"[ERROR] Error checking process: {e}")
        return False

def check_plugin_files():
    print_section("5. PLUGIN FILES CHECK")
    plugin_path = Path("E:/Drone/AirSim/Blocks/WindowsNoEditor/Blocks/Plugins/AirSim")
    
    if not plugin_path.exists():
        print(f"[ERROR] Plugin directory not found: {plugin_path}")
        print("    The AirSim plugin may not be installed")
        return False
    
    print(f"[OK] Plugin directory exists: {plugin_path}")
    
    # Check key files
    critical_files = [
        "AirSim.uplugin",
        "Binaries/Win64/AirSim.dll",
    ]
    
    all_present = True
    for file_rel in critical_files:
        file_path = plugin_path / file_rel
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"    [OK] {file_rel} ({size:,} bytes)")
            if size == 0:
                print(f"        [WARNING] File is empty!")
        else:
            print(f"    [ERROR] {file_rel} MISSING")
            all_present = False
    
    # Check DLL dependencies
    dll_path = plugin_path / "Binaries/Win64/AirSim.dll"
    if dll_path.exists():
        print("\n    Checking DLL dependencies...")
        try:
            # Use PowerShell to check DLL dependencies
            result = subprocess.run(
                ["powershell", "-Command", 
                 f"Get-Item '{dll_path}' | Select-Object -ExpandProperty VersionInfo"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("    [OK] DLL file appears valid")
        except:
            pass
    
    return all_present

def check_log_files():
    print_section("6. LOG FILES CHECK")
    log_paths = [
        Path(os.environ.get('LOCALAPPDATA', '')) / "AirSim" / "Blocks" / "Saved" / "Logs",
        Path("E:/Drone/AirSim/Blocks/WindowsNoEditor/Blocks/Saved/Logs"),
    ]
    
    log_found = False
    for log_dir in log_paths:
        if log_dir.exists():
            print(f"[OK] Log directory found: {log_dir}")
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                print(f"    Found {len(log_files)} log file(s)")
                # Check most recent log
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                print(f"    Latest: {latest_log.name}")
                print(f"    Size: {latest_log.stat().st_size:,} bytes")
                print(f"    Modified: {time.ctime(latest_log.stat().st_mtime)}")
                
                # Check for AirSim messages
                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'AirSim' in content:
                            print("    [OK] Log contains AirSim messages")
                            # Show last few AirSim lines
                            lines = [l for l in content.split('\n') if 'AirSim' in l]
                            if lines:
                                print("\n    Recent AirSim messages:")
                                for line in lines[-5:]:
                                    print(f"        {line[:100]}")
                        else:
                            print("    [WARNING] Log does not contain AirSim messages")
                            print("    This suggests plugin may not be loading")
                except Exception as e:
                    print(f"    [ERROR] Could not read log: {e}")
                
                log_found = True
                break
    
    if not log_found:
        print("[WARNING] No log directory found")
        print("    Logs may not be created until Blocks runs")
    
    return log_found

def test_connection():
    print_section("7. CONNECTION TEST")
    print("Attempting to connect to AirSim...")
    print("(This will timeout if API server is not running)")
    
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[SUCCESS] Connected to AirSim API!")
        
        # Get API version if available
        try:
            version = client.getApiVersion()
            print(f"    API Version: {version}")
        except:
            pass
        
        # Get vehicle state
        try:
            state = client.getMultirotorState()
            print("[OK] Vehicle state retrieved")
            print(f"    Position: {state.kinematics_estimated.position}")
        except Exception as e:
            print(f"    [WARNING] Could not get vehicle state: {e}")
        
        # Test camera
        try:
            camera_info = client.simGetCameraInfo("front_center")
            print("[OK] Camera access working")
        except Exception as e:
            print(f"    [WARNING] Camera check failed: {e}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Connection failed: {error_msg}")
        
        if "Connection refused" in error_msg or "actively refused" in error_msg:
            print("\n    DIAGNOSIS: API server is not running")
            print("    Possible causes:")
            print("    1. Blocks hasn't fully initialized (wait 2-3 minutes)")
            print("    2. AirSim plugin not loaded in Blocks")
            print("    3. Plugin failed to start API server")
            print("    4. Wrong Blocks executable (using AirSimNH instead)")
        elif "Connection timed out" in error_msg:
            print("\n    DIAGNOSIS: Connection timeout")
            print("    Possible causes:")
            print("    1. Blocks is still initializing")
            print("    2. Firewall blocking connection")
            print("    3. Network adapter issues")
        elif "No connection could be made" in error_msg:
            print("\n    DIAGNOSIS: Cannot reach API server")
            print("    This usually means port 41451 is not listening")
        
        return False

def check_settings():
    print_section("8. SETTINGS FILE CHECK")
    settings_paths = [
        Path(os.environ.get('USERPROFILE', '')) / "Documents" / "AirSim" / "settings.json",
        Path("E:/Drone/AirSim/Blocks/WindowsNoEditor/settings.json"),
        Path("E:/Drone/settings.json"),
    ]
    
    settings_found = False
    for settings_path in settings_paths:
        if settings_path.exists():
            print(f"[OK] Settings file found: {settings_path}")
            try:
                import json
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                
                sim_mode = settings.get('SimMode', 'Unknown')
                api_port = settings.get('ApiServerPort', 41451)
                
                print(f"    SimMode: {sim_mode}")
                print(f"    ApiServerPort: {api_port}")
                
                if sim_mode == 'Multirotor':
                    print("    [OK] SimMode is set to Multirotor (drone)")
                else:
                    print(f"    [WARNING] SimMode is '{sim_mode}', should be 'Multirotor'")
                
                if api_port == 41451:
                    print("    [OK] ApiServerPort is 41451")
                else:
                    print(f"    [WARNING] ApiServerPort is {api_port}, expected 41451")
                
                settings_found = True
                break
            except Exception as e:
                print(f"    [ERROR] Could not read settings: {e}")
    
    if not settings_found:
        print("[WARNING] No settings.json found")
        print("    AirSim will use defaults")
    
    return settings_found

def main():
    print("\n" + "=" * 70)
    print(" AIRSIM API CONNECTION DIAGNOSTIC TOOL")
    print("=" * 70)
    print("\nThis script will check your AirSim setup systematically.")
    print("Make sure Blocks.exe is running before proceeding.\n")
    
    input("Press Enter to start diagnostics...")
    
    results = {}
    
    results['python'] = check_python_version()
    results['airsim'] = check_airsim_import()
    results['settings'] = check_settings()
    results['plugin'] = check_plugin_files()
    results['process'] = check_blocks_process()
    results['logs'] = check_log_files()
    results['port'] = check_port_availability()
    results['connection'] = test_connection()
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)
    
    print(f"Tests passed: {passed}/{total}")
    print()
    
    if results['connection']:
        print("[SUCCESS] AirSim API connection is working!")
        print("You can proceed with flight scripts.")
    else:
        print("[ISSUE] AirSim API connection is not working.")
        print("\nRecommended next steps:")
        
        if not results['process']:
            print("1. Launch Blocks.exe first")
        elif not results['plugin']:
            print("1. Fix plugin installation (copy from AirSimNH or rebuild)")
        elif not results['port']:
            print("1. Wait longer for Blocks to initialize (3+ minutes)")
            print("2. Check log files for plugin loading errors")
            print("3. Verify plugin DLLs are correct version")
        else:
            print("1. Check log files for detailed error messages")
            print("2. Try restarting Blocks")
            print("3. Consider building AirSim from source")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


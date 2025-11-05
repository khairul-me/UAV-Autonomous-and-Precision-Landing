# AirSim API Connection Diagnostic Script
import sys
import socket
import subprocess
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("AirSim API Connection Diagnostic")
print("=" * 60)
print()

# Check 1: Is Blocks process running?
print("[1] Checking if Blocks.exe is running...")
try:
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq Blocks.exe'], 
                          capture_output=True, text=True, timeout=5)
    if 'Blocks.exe' in result.stdout:
        print("  [OK] Blocks.exe is running")
        # Extract PID
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Blocks.exe' in line and 'INFO' not in line:
                parts = line.split()
                if len(parts) > 1:
                    print(f"  PID: {parts[1]}")
    else:
        print("  [WARNING] Blocks.exe is NOT running")
        print("  -> Start Blocks.exe first to test connection")
except Exception as e:
    print(f"  [ERROR] Error checking process: {e}")

print()

# Check 2: Is port 41451 listening?
print("[2] Checking if port 41451 is listening...")
try:
    result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, timeout=5)
    if '41451' in result.stdout and 'LISTENING' in result.stdout:
        print("  [OK] Port 41451 is LISTENING")
        # Extract listening line
        for line in result.stdout.split('\n'):
            if '41451' in line and 'LISTENING' in line:
                print(f"  {line.strip()}")
    else:
        print("  [ERROR] Port 41451 is NOT listening")
        print("  -> AirSim API server is not running")
except Exception as e:
    print(f"  [ERROR] Error checking port: {e}")

print()

# Check 3: Can we connect to the port?
print("[3] Testing TCP connection to localhost:41451...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('127.0.0.1', 41451))
    sock.close()
    if result == 0:
        print("  [OK] Successfully connected to port 41451")
    else:
        print(f"  [ERROR] Connection refused (error code: {result})")
        print("  -> API server is not responding")
except Exception as e:
    print(f"  [ERROR] Connection error: {e}")

print()

# Check 4: Check plugin DLLs
print("[4] Checking for AirSim plugin DLLs...")
blocks_plugin = Path(r"E:\Drone\AirSim\Blocks\WindowsNoEditor\Blocks\Plugins\AirSim")
if blocks_plugin.exists():
    dlls = list(blocks_plugin.rglob("*.dll"))
    if dlls:
        print(f"  [OK] Found {len(dlls)} DLL file(s)")
        for dll in dlls[:5]:
            size_kb = dll.stat().st_size / 1024
            print(f"    - {dll.name} ({size_kb:.1f} KB)")
    else:
        print("  [ERROR] No DLL files found in plugin directory")
        print("  -> Plugin may be incomplete or not compiled")
        print("  -> This is the ROOT CAUSE of the API connection issue")
else:
    print(f"  [ERROR] Plugin directory not found: {blocks_plugin}")

print()

# Check 5: Check settings.json
print("[5] Checking settings.json...")
settings_paths = [
    Path(os.path.expanduser("~")) / "Documents" / "AirSim" / "settings.json",
    Path(r"E:\Drone\AirSim\Blocks\WindowsNoEditor\settings.json"),
    Path(r"E:\Drone\settings.json")
]

found_settings = False
for settings_path in settings_paths:
    if settings_path.exists():
        found_settings = True
        print(f"  [OK] Found: {settings_path}")
        try:
            import json
            with open(settings_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print("    [WARNING] Settings file is empty")
                else:
                    settings = json.loads(content)
                    print(f"    SimMode: {settings.get('SimMode', 'NOT SET')}")
                    print(f"    ApiServerPort: {settings.get('ApiServerPort', 'NOT SET')}")
        except json.JSONDecodeError as e:
            print(f"    [WARNING] JSON decode error: {e}")
        except Exception as e:
            print(f"    [WARNING] Error reading settings: {e}")
        break

if not found_settings:
    print("  [WARNING] settings.json not found in expected locations")

print()
print("=" * 60)
print("Diagnostic Complete")
print("=" * 60)
print()
print("SUMMARY:")
print("  - If Blocks.exe is not running: Start it first")
print("  - If port 41451 is not listening: Plugin DLLs are missing")
print("  - Solution: Build AirSim from source (see QUICK_FIX_GUIDE.md)")
print("=" * 60)

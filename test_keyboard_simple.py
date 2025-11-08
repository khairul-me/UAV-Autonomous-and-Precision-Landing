"""Simple keyboard test - verify input works"""
import msvcrt
import sys

print("="*70)
print("KEYBOARD INPUT TEST")
print("="*70)
print("\nPress keys to test. Press ESC to exit.\n")

try:
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                print("\nESC pressed - exiting")
                break
            elif key == b'\x00' or key == b'\xe0':
                key2 = msvcrt.getch()
                if key2 == b'H':
                    print("UP ARROW")
                elif key2 == b'P':
                    print("DOWN ARROW")
                elif key2 == b'K':
                    print("LEFT ARROW")
                elif key2 == b'M':
                    print("RIGHT ARROW")
            else:
                try:
                    decoded = key.decode('utf-8')
                    print(f"Key: '{decoded}' (code: {key})")
                except:
                    print(f"Key code: {key}")
except KeyboardInterrupt:
    print("\nExiting...")

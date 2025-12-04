import os
import sys
import subprocess
import time

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Checking dependencies...")
required = ['anthropic', 'pyautogui', 'Pillow', 'python-dotenv']
for req in required:
    try:
        __import__(req)
        print(f"Found {req}")
    except ImportError:
        print(f"Installing {req}...")
        install(req)

import pyautogui
from PIL import Image

print("\nTesting Permissions...")
print("You may see popups asking for Screen Recording or Accessibility access.")
print("Please Allow/Grant these permissions.")

try:
    input("Press ENTER to test Screen Recording...")
    subprocess.run(["screencapture", "-x", "test_permission.png"])
    if os.path.exists("test_permission.png"):
        img = Image.open("test_permission.png")
        print(f"Success: Screenshot captured ({img.size})")
        os.remove("test_permission.png")
    else:
        print("Fail: Screenshot file not created.")
except Exception as e:
    print(f"Fail: {e}")
    print("Go to System Settings > Privacy > Screen Recording")

try:
    input("Press ENTER to test Mouse Control (Mouse will jiggle)...")
    current_x, current_y = pyautogui.position()
    pyautogui.moveRel(10, 0)
    pyautogui.moveRel(-10, 0)
    print("Success: Mouse moved")
except Exception as e:
    print(f"Fail: {e}")
    print("Go to System Settings > Privacy > Accessibility")

print("\nConfig Setup...")
api_key = input("Enter Anthropic API Key: ").strip()
with open(".env", "w") as f:
    f.write(f"ANTHROPIC_API_KEY={api_key}\n")
    print("Created .env file")

print("\nSetup Complete.")
import os
import time
import base64
import io
import pyautogui
import tkinter as tk
import subprocess
from dotenv import load_dotenv
from anthropic import Anthropic
from PIL import Image

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

REAL_CENTER = (0, 0)

def start_calibration_window():
    global REAL_CENTER
    
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.configure(bg='white')
    
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    
    center_x = screen_w // 2
    center_y = screen_h // 2
    
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg='white', highlightthickness=0)
    canvas.pack()
    
    r = 5
    canvas.create_oval(center_x - r, center_y - r, center_x + r, center_y + r, fill='red', outline='red')
    
    root.update()
    
    REAL_CENTER = (center_x, center_y)
    print(f"Target Location: {REAL_CENTER}")
    
    time.sleep(1) 
    subprocess.run(["screencapture", "-x", "-m", "-C", "calib_temp.png"])
    
    root.after(2000, root.destroy)
    root.mainloop()

def analyze_calibration():
    start_calibration_window()
    
    print("Processing screenshot...")
    
    if not os.path.exists("calib_temp.png"):
        print("Error: Screenshot failed.")
        return

    img_data = ""
    with Image.open("calib_temp.png") as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        w, h = pyautogui.size()
        img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img_resized.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode("utf-8")
        
        img_dims = (w, h)

    print("Asking AI to click the dot...")
    
    tools = [{
        "type": "computer_20250124", 
        "name": "computer",
        "display_width_px": img_dims[0],
        "display_height_px": img_dims[1],
        "display_number": 1
    }]
    
    response = client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=tools,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Click the exact center of the red dot."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}}
            ]
        }],
        betas=["computer-use-2025-01-24"]
    )
    
    ai_x, ai_y = None, None
    for block in response.content:
        if block.type == "tool_use":
            coords = block.input.get("coordinate")
            if coords:
                ai_x, ai_y = coords[0], coords[1]
                
    if ai_x is None:
        print("Error: AI did not click.")
        return
        
    print(f"AI Clicked: ({ai_x}, {ai_y})")
    
    ratio_x = REAL_CENTER[0] / ai_x
    ratio_y = REAL_CENTER[1] / ai_y
    
    print("\nRESULTS")
    print(f"CALIB_X = {ratio_x:.4f}")
    print(f"CALIB_Y = {ratio_y:.4f}")
    
    if os.path.exists("calib_temp.png"):
        os.remove("calib_temp.png")

    with open(".env", "a") as f:
        f.write(f"CALIB_X={ratio_x:.4f}\n")
        f.write(f"CALIB_Y = {ratio_y:.4f}\n")
    print("Created .env file")

if __name__ == "__main__":
    analyze_calibration()

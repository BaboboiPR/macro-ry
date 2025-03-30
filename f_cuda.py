import torch
import pyautogui
import numpy as np
import keyboard
import time
from ultralytics import YOLO
import pydirectinput

# Detect if CUDA is available, else fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device.upper()}")

# Load YOLO model and move to GPU if available
model = YOLO(r"C:\Disk D\foldar nou\best.pt").to(device)

is_paused = False

def capture_screen():
    """Captures the screen and returns a NumPy array (HWC format)."""
    screenshot = pyautogui.screenshot()
    return np.array(screenshot.convert("RGB"))

def process_frame(frame):
    """Processes a frame using YOLO and returns detected object coordinates."""
    
    # Convert NumPy array (HWC) to PyTorch tensor (BCHW) and move to CUDA
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # Resize image to (640, 640), ensuring it's a multiple of 32
    frame_tensor = torch.nn.functional.interpolate(frame_tensor, size=(640, 640), mode="bilinear", align_corners=False)

    # Run YOLO object detection on GPU
    results = model(frame_tensor)

    detected_notes = []
    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                if conf > 0.4:  # Confidence threshold
                    detected_notes.append((x1, y1))

    return detected_notes

def toggle_pause():
    """Toggles pause state."""
    global is_paused
    is_paused = not is_paused
    print("⏸️ Paused" if is_paused else "▶️ Resumed")

def instant_mouse_move(x, y):
    """Moves the mouse instantly to the given coordinates."""
    pydirectinput.moveTo(x, y)

def main():
    """Main loop: Captures screen, detects objects, and moves the mouse."""
    global is_paused
    last_f1_press = 0

    while True:
        # Toggle pause with F1
        if keyboard.is_pressed("F1") and time.time() - last_f1_press > 0.5:
            toggle_pause()
            last_f1_press = time.time()

        if is_paused:
            time.sleep(0.1)
            continue

        frame = capture_screen()
        detected_notes = process_frame(frame)

        if detected_notes:
            for note_x, note_y in detected_notes:
                instant_mouse_move(note_x, note_y)
                print(f"🎯 Hovering over note at X:{note_x}, Y:{note_y}")
        else:
            print("❌ No notes found", end="\r")

        # Quit the program with "Q"
        if keyboard.is_pressed("q"):
            print("\n🔴 Exiting...")
            break

if __name__ == "__main__":
    main()

import torch
import pyautogui
import numpy as np
import keyboard
import time
import sys
from ultralytics import YOLO
import pydirectinput

model = YOLO(r"C:\Disk D\foldar nou\best.pt")

is_paused = False

def capture_screen():
    screenshot = pyautogui.screenshot()
    return np.array(screenshot.convert("RGB"))

def process_frame(frame):
    results = model(frame, imgsz=512)
    detected_notes = []

    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                if conf > 0.4:
                    detected_notes.append((x1, y1))

    return detected_notes

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    print("⏸️ Paused" if is_paused else "▶️ Resumed")

def instant_mouse_move(x, y):
    pydirectinput.moveTo(x, y)

def main():
    global is_paused
    last_f1_press = 0

    while True:
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

        if keyboard.is_pressed("q"):
            print("\n🔴 Exiting...")
            break

if __name__ == "__main__":
    main()

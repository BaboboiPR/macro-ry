import cv2
import torch
import pyautogui
import numpy as np
import keyboard
from ultralytics import YOLO
import pydirectinput  # Correct import

# Load YOLOv8 model
model = YOLO(r"C:\sound_space\pythonProject2\best.pt")

is_paused = False

def capture_screen():
    screenshot = pyautogui.screenshot()
    return np.array(screenshot)

def process_frame(frame):
    results = model(frame, imgsz=544)
    detected_notes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()

            if conf > 0.4:  # Confidence threshold
                # Directly use the coordinates of the detected box
                detected_notes.append((x1, y1))

    return detected_notes

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    print("Paused" if is_paused else "Resumed")

keyboard.add_hotkey("F1", toggle_pause)
keyboard.add_hotkey("F2", lambda: exit())

def instant_mouse_move(x, y):
    # Move instantly using pydirectinput, no duration, no delay
    pydirectinput.moveTo(x, y, duration=0)  # Instant movement (no delay)

def main():
    while True:
        if is_paused:
            continue
        frame = capture_screen()
        detected_notes = process_frame(frame)
        if detected_notes:
            for note_x, note_y in detected_notes:
                instant_mouse_move(note_x, note_y)  # Move instantly to detected note
                pydirectinput.click()  # Perform the click
                print(f"Hitting note at X:{note_x}, Y:{note_y}")
        else:
            print("No notes found")
        if keyboard.is_pressed("q"):
            break

if __name__ == "__main__":
    main()

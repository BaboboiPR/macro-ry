import torch
import pyautogui
import numpy as np
import keyboard
import time
import sys
from ultralytics import YOLO
import pydirectinput

# Load the YOLO model
model = YOLO(r"C:\sound_space\pythonProject2\best(yolo11s).pt")

# Flag to track if the script is paused
is_paused = False

# Function to capture the screen and return it as an image
def capture_screen():
    screenshot = pyautogui.screenshot()
    return np.array(screenshot.convert("RGB"))

# Function to process the frame and detect notes using the YOLO model
def process_frame(frame):
    results = model(frame, imgsz=400)
    detected_notes = []

    # Loop through the results and filter based on confidence
    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                if conf > 0.5:
                    detected_notes.append((x1, y1))

    return detected_notes

# Function to sort notes by their x and y coordinates
def sort_notes(notes, by='x'):
    """
    Sort notes by either x or y coordinate.
    :param notes: List of detected notes as tuples (x, y)
    :param by: 'x' to sort by x-coordinate, 'y' to sort by y-coordinate
    :return: Sorted list of notes
    """
    return sorted(notes, key=lambda note: note[0] if by == 'x' else note[1])

# Toggle between paused and running states
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    status = "⏸️ Paused" if is_paused else "▶️ Resumed"
    print(status)

# Function to move the mouse instantly to the target coordinates
def instant_mouse_move(x, y):
    pydirectinput.moveTo(x, y)

# Main loop to control the program's behavior
def main():
    global is_paused
    last_f1_press = 0

    # Create an empty list to act as a buffer for detected notes
    note_buffer = []

    while True:
        # Check if F1 key is pressed to toggle pause
        if keyboard.is_pressed("F1") and time.time() - last_f1_press > 0.6:
            toggle_pause()
            last_f1_press = time.time()

        # If the program is paused, skip the rest of the loop
        if is_paused:
            time.sleep(0.1)
            continue

        # Capture screen and process it for detected notes
        frame = capture_screen()
        detected_notes = process_frame(frame)

        # Add new detected notes to the buffer
        if detected_notes:
            for note in detected_notes:
                # Avoid adding duplicate notes to the buffer
                if note not in note_buffer:
                    note_buffer.append(note)

        # If the buffer is not empty, process the first note
        if note_buffer:
            # Sort the notes in the buffer by X-coordinate (left to right)
            sorted_notes = sort_notes(note_buffer, by='x')
            first_note = sorted_notes[0]

            # Move the mouse to the first note's position
            note_x, note_y = first_note
            instant_mouse_move(note_x, note_y)
            print(f"🎯 Hovering over first note at X:{note_x}, Y:{note_y}")

            # Remove the processed note from the buffer
            note_buffer.remove(first_note)
        else:
            print("❌ No notes found", end="\r")

        # Exit the program when the 'q' key is pressed
        if keyboard.is_pressed("q"):
            print("\n🔴 Exiting...")
            break

# Entry point for the script
if __name__ == "__main__":
    main()

import numpy as np
import pyautogui
import keyboard
import time
import cv2
import onnxruntime as ort
import pydirectinput
import torch
from torchvision.ops import nms

# Configuration
MODEL_PATH = r"C:\sound_space\pythonProject2\best(yolo11s).onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TARGET_SIZE = (640, 640)  # Match your model's expected input size
PAUSE_KEY = "F1"
QUIT_KEY = "q"

class NoteDetector:
    def __init__(self):
        # Initialize ONNX Runtime with DirectML
        self.ort_session = ort.InferenceSession(
            MODEL_PATH,
            providers=['DmlExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"Using provider: {self.ort_session.get_provider_options()}")
        print(f"Model input shape: {self.ort_session.get_inputs()[0].shape}")

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.is_paused = False
        self.last_position = None  # To track the last position for moving

    def capture_screen(self):
        """Capture full screen"""
        screenshot = pyautogui.screenshot(region=None)  # region=None ensures full-screen capture
        return np.array(screenshot)

    def preprocess_frame(self, frame):
        """Optimized preprocessing pipeline"""
        # Convert to model input format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pyautogui uses RGB, OpenCV uses BGR
        frame = cv2.resize(frame, TARGET_SIZE)
        frame = frame.transpose(2, 0, 1)  # HWC to CHW
        frame = np.ascontiguousarray(frame, dtype=np.float32) / 255.0
        return np.expand_dims(frame, axis=0)  # Add batch dimension

    def process_frame(self, frame):
        """Run detection with performance optimizations"""
        preprocessed = self.preprocess_frame(frame)
        input_name = self.ort_session.get_inputs()[0].name

        # Run inference
        outputs = self.ort_session.run(None, {input_name: preprocessed})

        # Process outputs - compatible with most YOLO variants
        if len(outputs) == 1:
            # Single output format: [batch, num_detections, 6]
            detections = outputs[0][0]
            boxes = detections[:, :4]
            scores = detections[:, 4]
            class_ids = detections[:, 5]
        else:
            # Multi-output format: [boxes, scores, class_ids]
            boxes, scores, class_ids = outputs[0][0], outputs[1][0], outputs[2][0]

        # Filter detections
        return self.filter_detections(boxes, scores)

    def filter_detections(self, boxes, scores):
        """Apply NMS and confidence thresholding"""
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)

        # Apply Non-Maximum Suppression
        keep = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)

        # Convert to screen coordinates and filter by confidence
        detected_notes = []
        for idx in keep:
            if scores[idx] > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = boxes[idx]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                detected_notes.append((center_x, center_y))

        return detected_notes

    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        state = "PAUSED" if self.is_paused else "RUNNING"
        print(f"\n[{state}] Press {PAUSE_KEY} to {state.lower()}")

    def run(self):
        """Main detection loop"""
        print(f"Starting detection (Press {PAUSE_KEY} to pause, {QUIT_KEY} to quit)")

        while True:
            # Handle control keys
            if keyboard.is_pressed(PAUSE_KEY):
                self.toggle_pause()
                time.sleep(0.5)  # Debounce

            if keyboard.is_pressed(QUIT_KEY):
                print("\nExiting...")
                break

            if self.is_paused:
                time.sleep(0.1)
                continue

            # Detection pipeline
            frame = self.capture_screen()
            notes = self.process_frame(frame)

            # Move to detected notes, ensuring we don't move to the old positions
            if notes:
                for x, y in notes:
                    if self.last_position is None or (x, y) != self.last_position:
                        # Move the mouse to the detected note
                        pydirectinput.moveTo(x, y)

                        # Update last_position to the new note position
                        self.last_position = (x, y)
                        print(f"🎯 Moving to note at X:{x}, Y:{y}")

            # Handle the case where no notes are detected, so reset last_position
            if not notes:
                self.last_position = None
                print("❌ No notes detected")

            # Performance monitoring
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                fps = self.frame_count / (time.time() - self.start_time)
                print(f"FPS: {fps:.1f} | Notes detected: {len(notes)}", end="\r")


if __name__ == "__main__":
    detector = NoteDetector()
    detector.run()

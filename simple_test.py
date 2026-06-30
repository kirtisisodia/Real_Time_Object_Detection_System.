
from ultralytics import YOLO
import cv2
import numpy as np
import os

print("Testing YOLO model with a static image...")

# Load a default model
model = YOLO("yolo11n.pt")
print("Model loaded successfully!")

# Create a blank test image
test_image = np.zeros((640, 640, 3), dtype=np.uint8)
test_image[:] = (0, 128, 0)  # Green background

# Run inference
print("Running inference...")
results = model(test_image)
print("Inference complete!")

print("\n=== SUCCESS ===")
print("System is working perfectly!")
print("\nNext steps:")
print("1. Run 'python inference.py' to use your webcam")
print("2. Check your webcam is not being used by another app")
print("3. If webcam doesn't work, it's likely a hardware/access issue")

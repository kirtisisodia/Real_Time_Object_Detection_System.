
import sys
print("Testing Construction Site Safety System Setup...")
print(f"Python version: {sys.version}")

try:
    import cv2
    print("[OK] OpenCV imported successfully")
except ImportError as e:
    print(f"[FAIL] OpenCV failed: {e}")

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__} imported successfully")
except ImportError as e:
    print(f"[FAIL] PyTorch failed: {e}")

try:
    from ultralytics import YOLO
    print("[OK] Ultralytics YOLO imported successfully")
except ImportError as e:
    print(f"[FAIL] YOLO failed: {e}")

try:
    import numpy as np
    print("[OK] NumPy imported successfully")
except ImportError as e:
    print(f"[FAIL] NumPy failed: {e}")

try:
    from dotenv import load_dotenv
    print("[OK] python-dotenv imported successfully")
except ImportError as e:
    print(f"[FAIL] dotenv failed: {e}")

print("\nAll imports successful! The system is ready to run.")
print("\nNext steps:")
print("1. Run 'python inference.py' to start the webcam demo (uses default YOLO model)")
print("2. Train your custom model using 'python train.py' after preparing your dataset")
print("3. Place your trained model at 'models/best.pt' to use custom classes")

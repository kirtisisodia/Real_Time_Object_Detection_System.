# Construction Site Safety Surveillance System

A comprehensive, multi-task edge AI surveillance system using YOLOv8/YOLOv11 and OpenCV for real-time construction site monitoring.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CCTV / RTSP Stream                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  YOLO Object Detection Model                    │
│  (Helmet, Vest, Boots, Fire, Smoke, Persons, Vehicles, Animals) │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ PPE Compliance│    │ Fire/Smoke Check │    │ Intrusion Detection│
│   Logic       │    │  (30 Frame Rule) │    │   (Geofencing)     │
└───────┬───────┘    └────────┬─────────┘    └─────────┬─────────┘
        │                     │                        │
        └─────────────────────┼────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Alert Generation │
                    │   & Logging      │
                    └──────────────────┘
```

## Dataset Setup Guide

### Dataset Structure
```
data/
├── data.yaml
└── images/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### Class Definitions
| ID | Class Name          | Description                                  |
|----|---------------------|----------------------------------------------|
| 0  | helmet              | Hardhat/Safety Helmet                        |
| 1  | safety_boots        | Safety Boots                                 |
| 2  | safety_vest         | High-Visibility Safety Vest                  |
| 3  | fire                | Fire/Flame                                   |
| 4  | smoke               | Smoke                                        |
| 5  | person_authorized   | Worker with Badge                            |
| 6  | person_unknown      | Intruder/No Badge                            |
| 7  | car                 | Car/Sedan                                    |
| 8  | truck               | Truck/Heavy Vehicle                          |
| 9  | bike                | Bicycle/Motorcycle                           |
| 10 | animal              | Animal (Dog, Cat, etc.)                      |

### Roboflow Workflow
1. Create a project on Roboflow with these 11 classes
2. Upload construction site images (diverse lighting, angles, weather)
3. Label all objects consistently
4. Split into train/val/test (70/20/10)
5. Apply augmentations: flip, rotate, brightness, contrast, blur
6. Export in YOLO format

### Model Strategy
**Recommendation: Train one unified model**
- More efficient (single inference pass)
- Better feature sharing between related classes
- Easier deployment and maintenance
- Use YOLO11n (nano) or YOLO11s (small) for edge deployment

## Installation & Setup

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure `.env` file with your settings

## Training

1. Place your labeled dataset in `data/images/` (train/val/test splits)
2. Run training:
```bash
python train.py --epochs 100 --batch 16 --imgsz 640
```

## Inference

1. Ensure trained model is in `models/best.pt`
2. Configure RTSP_URL in `.env` or use webcam (0)
3. Run inference:
```bash
python inference.py
```

### Controls
- **Left Click**: Add points to draw danger zone polygon
- **Right Click / 'c' Key**: Clear danger zone
- **'q' Key**: Quit application

## Key Features

### 1. PPE Compliance Check
- Detects when a person is missing helmet or vest
- Uses bounding box containment logic
- Red alerts and logs violations

### 2. Fire/Smoke Detection
- Persistence check: Only alerts after 30 consecutive frames
- Reduces false positives from dust, lights, etc.

### 3. Intrusion Detection (Geofencing)
- Draw custom danger zones with mouse
- Detects unauthorized persons/animals/vehicles
- Only active during after hours (configurable)

## Edge Optimization

### 1. ONNX Conversion
```python
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="onnx", imgsz=640)
```

### 2. TensorRT Conversion (NVIDIA GPUs/Jetson)
```python
model.export(format="engine", imgsz=640, half=True)
```

### 3. OpenVINO Conversion (Intel CPUs/GPUs)
```python
model.export(format="openvino", imgsz=640)
```

### 4. Additional Optimizations
- Use smaller models (YOLO11n, YOLO8n)
- Reduce image size (e.g., 416x416)
- Adjust confidence threshold
- Use FP16/INT8 quantization
- Frame skipping for high FPS streams

from ultralytics import YOLO
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script for Construction Site Safety")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model to use (yolo11n.pt, yolo8n.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device to use (0 for GPU, cpu for CPU, auto-detect if not specified)")
    args = parser.parse_args()

    # Auto-detect device if not specified
    device_info = "Unknown"
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "0"
            device_info = "CUDA GPU"
        elif torch.backends.mps.is_available():
            args.device = "mps"
            device_info = "Apple Silicon MPS"
        else:
            args.device = "cpu"
            device_info = "CPU"
    elif args.device == "0" and not torch.cuda.is_available():
        args.device = "cpu"
        device_info = "CPU (CUDA not available)"
    elif args.device == "cpu":
        device_info = "CPU"

    print("=== Construction Site Safety YOLO Training ===")
    print(f"Data path: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device} ({device_info})")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=50,
        save=True,
        plots=True,
        val=True,
        project="runs/train",
        name="construction_safety"
    )

    print("\n=== Training Complete ===")
    print(f"Best model saved at: {results.save_dir / 'weights/best.pt'}")

    val_results = model.val()
    print("\n=== Validation Results ===")
    print(val_results)

if __name__ == "__main__":
    main()

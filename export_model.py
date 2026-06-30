from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to edge-optimized formats")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model")
    parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, engine, openvino, torchscript)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", help="Use FP16 half-precision")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    print(f"=== Exporting Model to {args.format.upper()} ===")
    print(f"Model: {args.model}")
    print(f"Image size: {args.imgsz}")
    print(f"FP16: {args.half}")
    print(f"INT8: {args.int8}")

    model = YOLO(args.model)

    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        simplify=True
    )

    print(f"\n=== Export Complete ===")
    print(f"Model exported to: {export_path}")

if __name__ == "__main__":
    main()

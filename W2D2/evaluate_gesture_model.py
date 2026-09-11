from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/gesture_detect/weights/best.pt"
DATA_PATH = BASE_DIR / "data.yaml"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_PATH, split="test", imgsz=320, device=device)
    print(f"运行设备: {device}")
    print(f"precision: {metrics.box.mp:.6f}")
    print(f"recall: {metrics.box.mr:.6f}")
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()

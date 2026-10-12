from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.yaml"
MODEL_NAME = "yolo11n.pt"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"数据配置文件不存在: {DATA_PATH}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATA_PATH,
        epochs=100,
        batch=64,
        imgsz=320,
        device=device,
        project=BASE_DIR / "runs",
        name="gesture_detect",
        exist_ok=True,
    )
    print(f"训练设备: {device}")


if __name__ == "__main__":
    main()

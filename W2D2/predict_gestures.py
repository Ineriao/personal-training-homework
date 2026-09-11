import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用手势模型进行预测")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "runs/gesture_detect/weights/best.pt")
    parser.add_argument("--source", type=Path, default=BASE_DIR / "raw/images")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"预测来源不存在: {args.source}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=320,
        device=device,
        save=True,
        project=BASE_DIR / "results",
        name="gesture_predict",
        exist_ok=True,
    )
    print(f"预测图片数量: {len(results)}")
    print(f"运行设备: {device}")
    print(f"结果目录: {BASE_DIR / 'results/gesture_predict'}")


if __name__ == "__main__":
    main()

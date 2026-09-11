import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集手势识别图片")
    parser.add_argument("--class-name", required=True, choices=["fist", "like", "ok", "palm", "peace"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("raw/images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    saved_count = 0
    try:
        while saved_count < args.count:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("无法读取摄像头画面")

            display = frame.copy()
            text = f"{args.class_name}  {saved_count}/{args.count}  Space: save  Q: quit"
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gesture Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                image_path = output_dir / f"{args.class_name}_{saved_count:04d}.jpg"
                if cv2.imwrite(str(image_path), frame):
                    saved_count += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"已保存 {saved_count} 张 {args.class_name} 图片到 {output_dir}")


if __name__ == "__main__":
    main()

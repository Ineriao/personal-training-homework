from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def extract_frames(video_path, output_root, interval_seconds=1.0):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path.name}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    step = max(1, round(fps * interval_seconds)) if fps > 0 else 1

    output_dir = output_root / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % step == 0:
            timestamp = frame_index / fps if fps > 0 else 0
            output_path = output_dir / f"frame_{frame_index:06d}_{timestamp:08.2f}s.jpg"
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError(f"无法保存图片：{output_path}")
            output_path.write_bytes(encoded.tobytes())
            saved_count += 1
        frame_index += 1

    capture.release()
    return {
        "video": video_path.name,
        "fps": fps,
        "frames": frame_count,
        "duration": duration,
        "saved": saved_count,
    }


def main():
    video_dir = Path(__file__).parent / "video"
    output_root = Path(__file__).parent / "frames"
    videos = sorted(
        path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise FileNotFoundError(f"视频目录为空：{video_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "抽帧清单.txt"
    results = [extract_frames(path, output_root) for path in videos]

    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("抽帧间隔：1 秒 1 帧\n\n")
        for result in results:
            summary.write(
                f"{result['video']}\t"
                f"{result['fps']:.2f} FPS\t"
                f"{result['frames']} 帧\t"
                f"{result['duration']:.2f} 秒\t"
                f"抽取 {result['saved']} 张\n"
            )
            print(
                f"{result['video']}: {result['duration']:.2f}s, "
                f"抽取 {result['saved']} 张"
            )


if __name__ == "__main__":
    main()

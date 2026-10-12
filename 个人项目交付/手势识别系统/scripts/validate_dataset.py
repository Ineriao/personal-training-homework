import argparse
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["fist", "like", "ok", "palm", "peace"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 YOLO 手势数据集")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    return parser.parse_args()


def validate_label(label_path: Path, class_counts: Counter) -> list[str]:
    errors = []
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [f"空标注文件: {label_path}"]

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"格式错误: {label_path}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"数值错误: {label_path}:{line_number}")
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"类别编号越界: {label_path}:{line_number}")
        elif not all(0 <= value <= 1 for value in values):
            errors.append(f"坐标越界: {label_path}:{line_number}")
        elif values[2] <= 0 or values[3] <= 0:
            errors.append(f"标注框宽高无效: {label_path}:{line_number}")
        else:
            class_counts[CLASS_NAMES[class_id]] += 1
    return errors


def main() -> None:
    args = parse_args()
    all_errors = []

    for split_name in ("train", "val"):
        image_dir = args.dataset / "images" / split_name
        label_dir = args.dataset / "labels" / split_name
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        } if image_dir.is_dir() else {}
        labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.is_dir() else {}

        for stem in sorted(images.keys() - labels.keys()):
            all_errors.append(f"图片缺少标签: {images[stem]}")
        for stem in sorted(labels.keys() - images.keys()):
            all_errors.append(f"标签缺少图片: {labels[stem]}")

        class_counts = Counter()
        for stem in sorted(images.keys() & labels.keys()):
            all_errors.extend(validate_label(labels[stem], class_counts))

        print(f"{split_name}: {len(images)} 张图片，{len(labels)} 个标签文件")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {class_counts[class_name]} 个标注")

    if all_errors:
        print("检查失败:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("数据集检查通过")


if __name__ == "__main__":
    main()

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["fist", "like", "ok", "palm", "peace"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将标注样本转换并划分为 YOLO 数据集")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--save-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def convert_xml(xml_path: Path, image_width: int, image_height: int) -> list[str]:
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "").strip()
        if label not in CLASS_NAMES:
            raise ValueError(f"{xml_path} 包含未定义类别: {label}")
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        center_x = ((xmin + xmax) / 2) / image_width
        center_y = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        values = [center_x, center_y, width, height]
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"{xml_path} 的标注框超出图片范围")
        lines.append(f"{CLASS_NAMES.index(label)} " + " ".join(f"{value:.6f}" for value in values))
    return lines


def find_annotation(image_path: Path, label_dir: Path) -> Path:
    candidates = [
        label_dir / f"{image_path.stem}.xml",
        label_dir / f"{image_path.stem}.txt",
        image_path.with_suffix(".xml"),
        image_path.with_suffix(".txt"),
    ]
    for annotation_path in candidates:
        if annotation_path.is_file():
            return annotation_path
    raise FileNotFoundError(f"缺少标注文件: {image_path.name}")


def annotation_lines(image_path: Path, annotation_path: Path) -> list[str]:
    if annotation_path.suffix.lower() == ".txt":
        lines = [line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"标注文件为空: {annotation_path}")
        return lines

    from PIL import Image

    with Image.open(image_path) as image:
        return convert_xml(annotation_path, image.width, image.height)


def split_samples(
    samples: list[tuple[str, Path, Path]], val_ratio: float, seed: int
) -> dict[str, list[tuple[str, Path, Path]]]:
    grouped_samples = {class_name: [] for class_name in CLASS_NAMES}
    for sample in samples:
        grouped_samples[sample[0]].append(sample)

    result = {"train": [], "val": []}
    random_generator = random.Random(seed)
    for class_name, class_samples in grouped_samples.items():
        if len(class_samples) < 2:
            raise ValueError(f"{class_name} 至少需要 2 张已标注图片才能划分数据集")
        random_generator.shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * val_ratio))
        val_count = min(val_count, len(class_samples) - 1)
        result["val"].extend(class_samples[:val_count])
        result["train"].extend(class_samples[val_count:])

    random_generator.shuffle(result["train"])
    random_generator.shuffle(result["val"])
    return result


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 0 和 1 之间")

    samples = []
    for class_name in CLASS_NAMES:
        image_dir = args.raw_dir / "images" / class_name
        label_dir = args.raw_dir / "labels" / class_name
        image_paths = sorted(image_dir.iterdir()) if image_dir.is_dir() else []
        for image_path in image_paths:
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            annotation_path = find_annotation(image_path, label_dir)
            samples.append((class_name, image_path, annotation_path))

    if not samples:
        raise FileNotFoundError("未找到已标注样本，请先采集图片并使用 LabelImg 保存 XML 标注")

    splits = split_samples(samples, args.val_ratio, args.seed)
    existing_files = [path for path in args.save_dir.rglob("*") if path.is_file()]
    if existing_files:
        raise FileExistsError(f"输出目录中已有文件，请更换 --save-dir: {args.save_dir}")

    for split_name, split_samples in splits.items():
        image_dir = args.save_dir / "images" / split_name
        label_dir = args.save_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for _, image_path, annotation_path in split_samples:
            target_image = image_dir / image_path.name
            target_label = label_dir / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(annotation_lines(image_path, annotation_path)) + "\n", encoding="utf-8")

    class_counts = Counter(sample[0] for sample in samples)
    print(f"总样本: {len(samples)}")
    for class_name in CLASS_NAMES:
        print(f"{class_name}: {class_counts[class_name]}")
    print(f"训练集: {len(splits['train'])}")
    print(f"验证集: {len(splits['val'])}")
    print(f"输出目录: {args.save_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path


TARGET_CLASSES = ["fist", "like", "ok", "palm", "peace"]
IMAGE_ROOT = "hagrid-sample-30k-384p/hagrid_30k"
ANNOTATION_ROOT = "hagrid-sample-30k-384p/ann_train_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选 HaGRID 类别并转换为 YOLO 检测数据集")
    parser.add_argument("--archive", type=Path, default=Path("raw/archive.zip"))
    parser.add_argument("--save-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--limit-per-class", type=int, default=600)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def load_annotations(archive: zipfile.ZipFile) -> dict[str, tuple[str, dict]]:
    annotations = {}
    for class_name in TARGET_CLASSES:
        member = f"{ANNOTATION_ROOT}/{class_name}.json"
        for image_id, annotation in json.loads(archive.read(member)).items():
            annotations.setdefault(image_id, (class_name, annotation))
    return annotations


def selected_samples(
    annotations: dict[str, tuple[str, dict]],
    available_ids: set[str],
    limit_per_class: int,
    seed: int,
) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    by_class = {class_name: [] for class_name in TARGET_CLASSES}
    for image_id, (_, annotation) in annotations.items():
        if image_id not in available_ids:
            continue
        if any(label in TARGET_CLASSES for label in annotation.get("labels", [])):
            for class_name in TARGET_CLASSES:
                if class_name in annotation.get("labels", []):
                    by_class[class_name].append((image_id, annotation))

    selected_ids = set()
    for class_name, candidates in by_class.items():
        rng.shuffle(candidates)
        selected_ids.update(image_id for image_id, _ in candidates[:limit_per_class])

    return [
        (image_id, annotations[image_id][1])
        for image_id in sorted(selected_ids)
    ]


def yolo_lines(annotation: dict) -> list[str]:
    lines = []
    for bbox, label in zip(annotation.get("bboxes", []), annotation.get("labels", [])):
        if label not in TARGET_CLASSES:
            continue
        x, y, width, height = bbox
        center_x = x + width / 2
        center_y = y + height / 2
        if not all(0 <= value <= 1 for value in (center_x, center_y, width, height)):
            raise ValueError(f"标注坐标超出范围: {bbox}")
        lines.append(
            f"{TARGET_CLASSES.index(label)} "
            f"{center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def split_samples(
    samples: list[tuple[str, dict]], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[tuple[str, dict]]]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    test_count = round(len(samples) * test_ratio)
    val_count = round(len(samples) * val_ratio)
    if test_count == 0 or val_count == 0 or len(samples) <= test_count + val_count:
        raise ValueError("样本数量不足以划分训练集、验证集和测试集")
    return {
        "test": samples[:test_count],
        "val": samples[test_count:test_count + val_count],
        "train": samples[test_count + val_count:],
    }


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"找不到压缩包: {args.archive}")
    if args.limit_per_class <= 0:
        raise ValueError("--limit-per-class 必须大于 0")
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("验证集和测试集比例设置无效")
    if any(path.is_file() for path in args.save_dir.rglob("*")):
        raise FileExistsError(f"输出目录已有内容，请使用空目录: {args.save_dir}")

    with zipfile.ZipFile(args.archive) as archive:
        annotations = load_annotations(archive)
        image_members = {
            Path(name).stem: name
            for name in archive.namelist()
            if name.startswith(f"{IMAGE_ROOT}/") and name.endswith(".jpg")
        }
        samples = selected_samples(
            annotations, set(image_members), args.limit_per_class, args.seed
        )
        splits = split_samples(samples, args.val_ratio, args.test_ratio, args.seed)

        for split_name, split_samples_list in splits.items():
            image_dir = args.save_dir / "images" / split_name
            label_dir = args.save_dir / "labels" / split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_id, annotation in split_samples_list:
                member = image_members[image_id]
                with archive.open(member) as source, (image_dir / f"{image_id}.jpg").open("wb") as target:
                    shutil.copyfileobj(source, target)
                lines = yolo_lines(annotation)
                if not lines:
                    raise ValueError(f"没有目标类别标注: {image_id}")
                (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"保留样本: {len(samples)}")
    for split_name, split_samples_list in splits.items():
        print(f"{split_name}: {len(split_samples_list)}")
    print(f"类别: {', '.join(TARGET_CLASSES)}")
    print(f"输出目录: {args.save_dir}")


if __name__ == "__main__":
    main()

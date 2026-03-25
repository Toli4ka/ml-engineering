import kagglehub
from pathlib import Path
import re
import random
import shutil
from PIL import Image
import yaml



def get_dataset_dir():
    data_dir = Path(kagglehub.dataset_download("psvishnu/pennfudan-database-for-pedestrian-detection-zip"))
    data_dir = data_dir / "PennFudanPed"
    return data_dir


def check_dataset(dataset_path, min_relative_area=0.01, create_yolo_labels=False):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_path}")
    
    annotation_dir_path = dataset_path / "Annotation"
    if not annotation_dir_path.exists() or not annotation_dir_path.is_dir():
        raise ValueError(f"Annotation directory does not exist: {annotation_dir_path}")

    pattern_img_size = re.compile(r"Image size \(X x Y x C\)\s*:\s*(\d+)\s*x\s*(\d+)\s*x\s*\d+")
    pattern_img_path = re.compile(r'Image filename\s*:\s*"([^"]+)"')
    pattern_obj_count = re.compile(r"Objects with ground truth\s*:\s*(\d+)")
    pattern_bbox = re.compile(
        r"Bounding box for object\s+(\d+).*\(Xmin, Ymin\) - \(Xmax, Ymax\)\s*:\s*"
        r"\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)"
    )

    errors = []
    small_boxes = []
    checked_annotations = 0
    total_boxes = 0

    for annotation_file in sorted(annotation_dir_path.iterdir()):
        if not annotation_file.is_file():
            continue

        text = annotation_file.read_text(encoding="utf-8", errors="ignore")
        match_path = pattern_img_path.search(text)
        match_size = pattern_img_size.search(text)
        match_obj_count = pattern_obj_count.search(text)
        bbox_matches = list(pattern_bbox.finditer(text))

        if not match_path or not match_size:
            errors.append(
                {
                    "annotation": str(annotation_file),
                    "error": "missing_image_path_or_size",
                }
            )
            continue

        if not match_obj_count:
            errors.append(
                {
                    "annotation": str(annotation_file),
                    "error": "missing_objects_count",
                }
            )
            continue

        # Image filename example: "PennFudanPed/PNGImages/FudanPed00001.png"
        img_path = Path(dataset_path.parent / Path(match_path.group(1)))
        ann_width, ann_height = map(int, match_size.groups())
        expected_objects = int(match_obj_count.group(1))
        found_objects = len(bbox_matches)
        checked_annotations += 1

        if not img_path.exists():
            errors.append(
                {
                    "annotation": str(annotation_file),
                    "image_path": str(img_path),
                    "error": "image_not_found",
                }
            )
            continue

        with Image.open(img_path) as img:
            true_width, true_height = img.size

        if (true_width, true_height) != (ann_width, ann_height):
            errors.append(
                {
                    "annotation": str(annotation_file),
                    "image_path": str(img_path),
                    "annotation_size": (ann_width, ann_height),
                    "actual_size": (true_width, true_height),
                    "error": "image_size_mismatch",
                }
            )
            continue
        
        if expected_objects != found_objects:
            errors.append(
                {
                    "annotation": str(annotation_file),
                    "expected_objects": expected_objects,
                    "found_boxes": found_objects,
                    "error": "objects_count_mismatch",
                }
            )
            continue

        img_area = ann_width * ann_height
        bb_list = []
        for bbox_match in bbox_matches:
            object_id, xmin, ymin, xmax, ymax = map(int, bbox_match.groups())
            total_boxes += 1

            if xmin < 1 or ymin < 1 or xmax < 1 or ymax < 1:
                errors.append(
                    {
                        "annotation": str(annotation_file),
                        "object_id": object_id,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "error": "bbox_coordinate_below_1",
                    }
                )
                continue

            if xmin >= xmax or ymin >= ymax:
                errors.append(
                    {
                        "annotation": str(annotation_file),
                        "object_id": object_id,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "error": "bbox_invalid_min_max",
                    }
                )
                continue

            if xmax > ann_width or ymax > ann_height:
                errors.append(
                    {
                        "annotation": str(annotation_file),
                        "object_id": object_id,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "image_size": (ann_width, ann_height),
                        "error": "bbox_out_of_image_bounds",
                    }
                )
                continue

            bbox_area = (xmax - xmin) * (ymax - ymin)
            relative_area = bbox_area / img_area
            if relative_area < min_relative_area:
                small_boxes.append(
                    {
                        "annotation": str(annotation_file),
                        "object_id": object_id,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "relative_area": relative_area,
                        "threshold": min_relative_area,
                    }
                )       

            # add xmin, ymin, xmax, ymax to the bb_list
            bb_list.append((xmin, ymin, xmax, ymax))

        if create_yolo_labels:
            create_yolo_label(annotation_file, bb_list, ann_width, ann_height)

    return {
        "is_valid": len(errors) == 0,
        "checked_annotations": checked_annotations,
        "total_boxes": total_boxes,
        "small_boxes_count": len(small_boxes),
        "errors": errors,
        "small_boxes": small_boxes,
    }



def create_yolo_label(annotation_file, bb_list, ann_width, ann_height):
    annotation_file = Path(annotation_file)
    labels_dir = annotation_file.parent.parent / "labels_yolo"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_file = labels_dir / f"{annotation_file.stem}.txt"

    lines = []
    for xmin, ymin, xmax, ymax in bb_list:
        box_width = xmax - xmin
        box_height = ymax - ymin
        x_center = xmin + box_width / 2
        y_center = ymin + box_height / 2

        # YOLO expects all values normalized by image width/height.
        x_center_norm = x_center / ann_width
        y_center_norm = y_center / ann_height
        width_norm = box_width / ann_width
        height_norm = box_height / ann_height

        lines.append(
            f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        )

    label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_file


def create_yolo_dataset(dataset_dir, val_size):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    labels_src_dir = dataset_dir / "labels_yolo"
    images_src_dir = dataset_dir / "PNGImages"
    if not labels_src_dir.exists() or not labels_src_dir.is_dir():
        raise ValueError(f"labels_yolo directory does not exist: {labels_src_dir}")
    if not images_src_dir.exists() or not images_src_dir.is_dir():
        raise ValueError(f"PNGImages directory does not exist: {images_src_dir}")

    label_files = sorted(labels_src_dir.glob("*.txt"))
    if not label_files:
        raise ValueError(f"No label files found in: {labels_src_dir}")

    pairs = []
    missing_images = []
    for label_path in label_files:
        stem = label_path.stem
        img_candidates = [
            images_src_dir / f"{stem}.png",
            images_src_dir / f"{stem}.jpg",
            images_src_dir / f"{stem}.jpeg",
        ]
        img_path = next((candidate for candidate in img_candidates if candidate.exists()), None)
        if img_path is None:
            missing_images.append(str(label_path))
            continue
        pairs.append((img_path, label_path))

    if not pairs:
        raise ValueError("No image/label pairs available to build YOLO dataset.")

    if isinstance(val_size, float):
        if not 0 < val_size < 1:
            raise ValueError("val_size as float must be in range (0, 1).")
        val_count = int(round(len(pairs) * val_size))
    elif isinstance(val_size, int):
        if val_size < 1 or val_size >= len(pairs):
            raise ValueError("val_size as int must be in range [1, len(pairs)-1].")
        val_count = val_size
    else:
        raise TypeError("val_size must be float (ratio) or int (count).")

    random.Random(42).shuffle(pairs)
    train_pairs = pairs[:-val_count]
    val_pairs = pairs[-val_count:]

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    train_images_dir = data_dir / "images" / "train"
    val_images_dir = data_dir / "images" / "val"
    train_labels_dir = data_dir / "labels" / "train"
    val_labels_dir = data_dir / "labels" / "val"

    for target_dir in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        target_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in train_pairs:
        shutil.copy2(img_path, train_images_dir / img_path.name)
        shutil.copy2(label_path, train_labels_dir / label_path.name)

    for img_path, label_path in val_pairs:
        shutil.copy2(img_path, val_images_dir / img_path.name)
        shutil.copy2(label_path, val_labels_dir / label_path.name)

    return {
        "data_dir": str(data_dir),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "missing_images_for_labels": missing_images,
    }


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_data_config(data_config_path):
    if not data_config_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_config_path}")

    with data_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Unexpected dataset config format: {data_config_path}")

    return config


def resolve_dataset_root(data_config_path):
    config = load_data_config(data_config_path)
    dataset_root = Path(config.get("path", data_config_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_config_path.parent / dataset_root).resolve()
    return dataset_root


def get_dataset_split_dir(split, data_config_path):
    config = load_data_config(data_config_path)
    dataset_root = resolve_dataset_root(data_config_path)
    split_value = config.get(split)
    if not split_value:
        raise ValueError(f"Split '{split}' is missing in {data_config_path}")
    if isinstance(split_value, list):
        raise ValueError(f"Split '{split}' must resolve to a single directory for this GUI draft.")

    split_dir = Path(split_value)
    if not split_dir.is_absolute():
        split_dir = dataset_root / split_dir
    split_dir = split_dir.resolve()

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    return split_dir


def list_dataset_images(data_config_path, splits=("train", "val")):
    image_paths = []
    for split in splits:
        split_dir = get_dataset_split_dir(split, data_config_path)
        split_images = sorted(
            path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths.extend(split_images)
    return image_paths


def get_label_path_for_image(image_path, data_config_path):
    image_path = Path(image_path)
    dataset_root = resolve_dataset_root(data_config_path)

    try:
        relative_path = image_path.resolve().relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Image path must be inside dataset root: {image_path}") from exc

    if len(relative_path.parts) < 3 or relative_path.parts[0] != "images":
        raise ValueError(f"Expected image path inside images/<split>: {image_path}")

    split = relative_path.parts[1]
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def read_yolo_labels(image_path, data_config_path=None):
    image_path = Path(image_path)
    label_path = get_label_path_for_image(image_path, data_config_path)
    if not label_path.exists():
        return []

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        x_min = (x_center - box_width / 2) * image_width
        y_min = (y_center - box_height / 2) * image_height
        x_max = (x_center + box_width / 2) * image_width
        y_max = (y_center + box_height / 2) * image_height

        labels.append(
            {
                "class_id": class_id,
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
            }
        )

    return labels


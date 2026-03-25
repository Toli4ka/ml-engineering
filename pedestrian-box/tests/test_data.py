from pathlib import Path

import pytest
import yaml
from PIL import Image

from pedestrian_box import data


def make_image(path: Path, size=(100, 200), color=(255, 255, 255)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_annotation(
    path: Path,
    image_reference: str,
    size=(100, 200),
    boxes=((1, 10, 20, 50, 120),),
    include_size=True,
    include_objects_count=True,
):
    lines = [f'Image filename : "{image_reference}"']
    if include_size:
        lines.append(f"Image size (X x Y x C) : {size[0]} x {size[1]} x 3")
    if include_objects_count:
        lines.append(f"Objects with ground truth : {len(boxes)}")

    for object_id, xmin, ymin, xmax, ymax in boxes:
        lines.append(
            "Bounding box for object "
            f"{object_id} \"PASperson\" (Xmin, Ymin) - (Xmax, Ymax) : "
            f"({xmin}, {ymin}) - ({xmax}, {ymax})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_dataset_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "PennFudanPed"
    (dataset_root / "Annotation").mkdir(parents=True)
    (dataset_root / "PNGImages").mkdir(parents=True)
    return dataset_root


def write_data_config(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_data_config_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "data.yml"

    with pytest.raises(FileNotFoundError):
        data.load_data_config(missing_path)


def test_resolve_dataset_root_supports_relative_path(tmp_path):
    config_path = tmp_path / "config" / "data.yml"
    write_data_config(config_path, {"path": "../dataset", "train": "images/train", "val": "images/val"})

    resolved = data.resolve_dataset_root(config_path)

    assert resolved == (config_path.parent / "../dataset").resolve()


def test_get_dataset_split_dir_raises_for_missing_split(tmp_path):
    config_path = tmp_path / "data.yml"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    write_data_config(config_path, {"path": str(dataset_root), "train": "images/train"})

    with pytest.raises(ValueError, match="Split 'val' is missing"):
        data.get_dataset_split_dir("val", config_path)


def test_list_dataset_images_collects_supported_extensions(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    make_image(train_dir / "a.png")
    make_image(train_dir / "ignore.gif")
    make_image(val_dir / "b.jpg")

    config_path = tmp_path / "data.yml"
    write_data_config(
        config_path,
        {"path": str(dataset_root), "train": "images/train", "val": "images/val"},
    )

    image_names = [path.name for path in data.list_dataset_images(config_path)]

    assert image_names == ["a.png", "b.jpg"]


def test_get_label_path_for_image_maps_to_matching_split(tmp_path):
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "train" / "sample.png"
    make_image(image_path)

    config_path = tmp_path / "data.yml"
    write_data_config(
        config_path,
        {"path": str(dataset_root), "train": "images/train", "val": "images/val"},
    )

    label_path = data.get_label_path_for_image(image_path, config_path)

    assert label_path == dataset_root / "labels" / "train" / "sample.txt"


def test_get_label_path_for_image_rejects_paths_outside_dataset_root(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside_image = tmp_path / "outside.png"
    make_image(outside_image)

    config_path = tmp_path / "data.yml"
    write_data_config(
        config_path,
        {"path": str(dataset_root), "train": "images/train", "val": "images/val"},
    )

    with pytest.raises(ValueError, match="inside dataset root"):
        data.get_label_path_for_image(outside_image, config_path)


def test_read_yolo_labels_converts_normalized_boxes_and_skips_bad_lines(tmp_path):
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "train" / "sample.png"
    make_image(image_path, size=(100, 200))

    label_path = dataset_root / "labels" / "train" / "sample.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "0 0.300000 0.350000 0.400000 0.500000\ninvalid line\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "data.yml"
    write_data_config(
        config_path,
        {"path": str(dataset_root), "train": "images/train", "val": "images/val"},
    )

    labels = data.read_yolo_labels(image_path, config_path)

    assert len(labels) == 1
    assert labels[0]["class_id"] == 0
    assert labels[0]["bbox_xyxy"] == pytest.approx([10.0, 20.0, 50.0, 120.0])


def test_create_yolo_label_writes_expected_normalized_values(tmp_path):
    annotation_file = tmp_path / "PennFudanPed" / "Annotation" / "FudanPed00001.txt"
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_file.write_text("", encoding="utf-8")

    label_file = data.create_yolo_label(annotation_file, [(10, 20, 50, 120)], 100, 200)

    assert label_file.read_text(encoding="utf-8") == "0 0.300000 0.350000 0.400000 0.500000\n"


def test_check_dataset_returns_valid_summary_and_creates_yolo_labels(tmp_path):
    dataset_root = make_dataset_root(tmp_path)
    make_image(dataset_root / "PNGImages" / "FudanPed00001.png", size=(100, 200))
    make_annotation(
        dataset_root / "Annotation" / "FudanPed00001.txt",
        "PennFudanPed/PNGImages/FudanPed00001.png",
        size=(100, 200),
        boxes=((1, 10, 20, 50, 120),),
    )

    result = data.check_dataset(dataset_root, create_yolo_labels=True)

    assert result["is_valid"] is True
    assert result["checked_annotations"] == 1
    assert result["total_boxes"] == 1
    assert result["small_boxes_count"] == 0
    assert result["errors"] == []
    assert (dataset_root / "labels_yolo" / "FudanPed00001.txt").exists()


def test_check_dataset_reports_small_boxes_without_failing_validation(tmp_path):
    dataset_root = make_dataset_root(tmp_path)
    make_image(dataset_root / "PNGImages" / "FudanPed00001.png", size=(100, 100))
    make_annotation(
        dataset_root / "Annotation" / "FudanPed00001.txt",
        "PennFudanPed/PNGImages/FudanPed00001.png",
        size=(100, 100),
        boxes=((1, 10, 10, 12, 12),),
    )

    result = data.check_dataset(dataset_root, min_relative_area=0.01)

    assert result["is_valid"] is True
    assert result["small_boxes_count"] == 1
    assert result["small_boxes"][0]["bbox"] == (10, 10, 12, 12)


def test_check_dataset_reports_image_size_mismatch(tmp_path):
    dataset_root = make_dataset_root(tmp_path)
    make_image(dataset_root / "PNGImages" / "FudanPed00001.png", size=(80, 200))
    make_annotation(
        dataset_root / "Annotation" / "FudanPed00001.txt",
        "PennFudanPed/PNGImages/FudanPed00001.png",
        size=(100, 200),
        boxes=((1, 10, 20, 50, 120),),
    )

    result = data.check_dataset(dataset_root)

    assert result["is_valid"] is False
    assert result["errors"][0]["error"] == "image_size_mismatch"


def test_check_dataset_reports_invalid_bbox_coordinates(tmp_path):
    dataset_root = make_dataset_root(tmp_path)
    make_image(dataset_root / "PNGImages" / "FudanPed00001.png", size=(100, 200))
    make_annotation(
        dataset_root / "Annotation" / "FudanPed00001.txt",
        "PennFudanPed/PNGImages/FudanPed00001.png",
        size=(100, 200),
        boxes=((1, 0, 20, 50, 120),),
    )

    result = data.check_dataset(dataset_root)

    assert result["is_valid"] is False
    assert result["errors"][0]["error"] == "bbox_coordinate_below_1"


def test_create_yolo_dataset_builds_train_val_split_and_reports_missing_images(tmp_path, monkeypatch):
    dataset_root = make_dataset_root(tmp_path)
    labels_dir = dataset_root / "labels_yolo"
    labels_dir.mkdir()

    for idx in range(4):
        stem = f"FudanPed{idx:05d}"
        make_image(dataset_root / "PNGImages" / f"{stem}.png")
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")

    (labels_dir / "FudanPed99999.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")

    fake_project_root = tmp_path / "project"
    fake_module_path = fake_project_root / "src" / "pedestrian_box" / "data.py"
    fake_module_path.parent.mkdir(parents=True, exist_ok=True)
    fake_module_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(data, "__file__", str(fake_module_path))

    result = data.create_yolo_dataset(dataset_root, val_size=0.25)

    train_images = sorted(path.name for path in (fake_project_root / "data" / "images" / "train").iterdir())
    val_images = sorted(path.name for path in (fake_project_root / "data" / "images" / "val").iterdir())

    assert result["train_count"] == 3
    assert result["val_count"] == 1
    assert len(result["missing_images_for_labels"]) == 1
    assert train_images == ["FudanPed00001.png", "FudanPed00002.png", "FudanPed00003.png"]
    assert val_images == ["FudanPed00000.png"]


@pytest.mark.parametrize("val_size", [0, 1.0, "bad"])
def test_create_yolo_dataset_rejects_invalid_val_size(tmp_path, val_size):
    dataset_root = make_dataset_root(tmp_path)
    labels_dir = dataset_root / "labels_yolo"
    labels_dir.mkdir()

    for idx in range(2):
        stem = f"FudanPed{idx:05d}"
        make_image(dataset_root / "PNGImages" / f"{stem}.png")
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")

    expected_exception = TypeError if isinstance(val_size, str) else ValueError
    with pytest.raises(expected_exception):
        data.create_yolo_dataset(dataset_root, val_size=val_size)

# -*- coding: utf-8 -*-
"""학습용 데이터셋 폴더를 검사하고 train/val/test 구조로 배치합니다."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetPair:
    """같은 이름의 이미지와 라벨 파일 쌍을 나타냅니다."""

    image_path: Path
    label_path: Path


@dataclass
class DatasetBuildResult:
    """데이터셋 분할 결과 요약입니다."""

    project_dir: Path
    train_count: int
    val_count: int
    test_count: int


def _load_class_names(classes_path: Path) -> list[str]:
    """classes.txt에서 비어 있지 않은 클래스명만 읽어옵니다."""
    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError("classes.txt에 유효한 클래스명이 없습니다.")
    return class_names


def _is_valid_label_file(label_path: Path) -> bool:
    """라벨 파일이 비어 있지 않고 읽을 수 있는지 간단히 검사합니다."""
    try:
        if not label_path.is_file():
            return False
        _text = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    return True


def scan_dataset_pairs(dataset_dir: Path) -> tuple[Path, list[DatasetPair]]:
    """classes.txt와 이미지/라벨 쌍을 검사해 유효한 목록만 수집합니다."""
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError("선택한 데이터셋 폴더를 찾을 수 없습니다.")

    classes_path = dataset_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError("선택한 폴더에 classes.txt가 없습니다.")

    pairs: list[DatasetPair] = []
    for image_path in sorted(dataset_dir.iterdir(), key=lambda item: item.name.lower()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        if not _is_valid_label_file(label_path):
            continue
        pairs.append(DatasetPair(image_path=image_path, label_path=label_path))

    if not pairs:
        raise FileNotFoundError("같은 이름의 이미지/라벨 쌍을 찾지 못했습니다.")

    return classes_path, pairs


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """입력된 비율이 모두 0 이상이며 합계가 100인지 확인합니다."""
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Train, Val, Test 비율은 0 이상이어야 합니다.")

    total_ratio = round(sum(ratios), 6)
    if total_ratio != 100.0:
        raise ValueError("Train, Val, Test 비율의 합계는 100이어야 합니다.")


def _split_pairs(
    pairs: list[DatasetPair],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle_enabled: bool,
) -> tuple[list[DatasetPair], list[DatasetPair], list[DatasetPair]]:
    """비율과 셔플 여부에 따라 데이터셋 쌍을 train/val/test로 나눕니다."""
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    ordered_pairs = list(pairs)
    if shuffle_enabled:
        # 재현 가능한 고정 시드 대신 실행마다 새로 섞어 실제 셔플 동작을 보장합니다.
        random.shuffle(ordered_pairs)

    total_count = len(ordered_pairs)
    train_count = int(total_count * train_ratio / 100.0)
    val_count = int(total_count * val_ratio / 100.0)
    test_count = total_count - train_count - val_count

    train_pairs = ordered_pairs[:train_count]
    val_pairs = ordered_pairs[train_count : train_count + val_count]
    test_pairs = ordered_pairs[train_count + val_count : train_count + val_count + test_count]
    return train_pairs, val_pairs, test_pairs


def _prepare_project_directories(project_dir: Path) -> None:
    """프로젝트용 images/labels/result 디렉터리 구조를 생성합니다."""
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError("같은 이름의 학습 프로젝트 폴더가 이미 존재합니다.")

    for split_name in ("train", "val", "test"):
        (project_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (project_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    (project_dir / "result").mkdir(parents=True, exist_ok=True)


def _write_data_yaml(project_dir: Path, class_names: list[str]) -> None:
    """Ultralytics가 항상 올바른 학습 폴더를 찾도록 data.yaml을 생성합니다."""
    # Ultralytics는 실행 위치에 따라 상대 경로를 다르게 해석할 수 있으므로
    # 프로젝트 폴더 절대 경로를 path에 기록해 데이터셋 경로가 흔들리지 않게 합니다.
    yaml_lines = [
        f"path: {project_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for index, class_name in enumerate(class_names):
        yaml_lines.append(f"  {index}: {class_name}")
    (project_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def _transfer_file(source_path: Path, target_path: Path, move_files: bool) -> None:
    """복사 또는 이동 모드에 따라 파일을 목적지로 옮깁니다."""
    if move_files:
        shutil.move(str(source_path), str(target_path))
    else:
        shutil.copy2(source_path, target_path)


def build_training_project(
    dataset_dir: Path,
    workspace_dir: Path,
    project_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle_enabled: bool,
    move_files: bool,
) -> DatasetBuildResult:
    """원본 데이터셋을 학습 프로젝트용 train/val/test 구조로 배치합니다."""
    classes_path, pairs = scan_dataset_pairs(dataset_dir)
    class_names = _load_class_names(classes_path)

    if not project_name.strip():
        raise ValueError("학습 프로젝트명을 입력해주세요.")

    project_dir = workspace_dir / project_name.strip()
    _prepare_project_directories(project_dir)

    train_pairs, val_pairs, test_pairs = _split_pairs(
        pairs=pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle_enabled=shuffle_enabled,
    )

    split_map = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    for split_name, split_pairs in split_map.items():
        image_dir = project_dir / "images" / split_name
        label_dir = project_dir / "labels" / split_name
        for pair in split_pairs:
            _transfer_file(pair.image_path, image_dir / pair.image_path.name, move_files)
            _transfer_file(pair.label_path, label_dir / pair.label_path.name, move_files)

    target_classes_path = project_dir / "classes.txt"
    _transfer_file(classes_path, target_classes_path, move_files)
    _write_data_yaml(project_dir, class_names)

    return DatasetBuildResult(
        project_dir=project_dir,
        train_count=len(train_pairs),
        val_count=len(val_pairs),
        test_count=len(test_pairs),
    )

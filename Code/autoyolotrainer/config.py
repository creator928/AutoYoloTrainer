# -*- coding: utf-8 -*-
"""프로그램 경로와 settings.json 로딩/저장을 담당합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .constants import DEFAULT_CONFIG
from .models import AppConfig, AppPaths


def get_app_root() -> Path:
    """실행 환경에 맞는 루트 경로를 계산합니다."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_runtime_code_dir(root_dir: Path) -> Path:
    """외부 Python으로 실행할 러너 스크립트 위치를 반환합니다."""
    if getattr(sys, "frozen", False):
        # PyInstaller onefile 실행 시 add-data로 포함한 러너는 임시 해제 폴더에 위치합니다.
        return Path(getattr(sys, "_MEIPASS", root_dir)).resolve()
    return root_dir / "Code"


def build_paths() -> AppPaths:
    """프로젝트 고정 디렉터리 구조를 경로 객체로 구성합니다."""
    root_dir = get_app_root()
    data_dir = root_dir / "Data"
    return AppPaths(
        root_dir=root_dir,
        code_dir=get_runtime_code_dir(root_dir),
        data_dir=data_dir,
        document_dir=root_dir / "Document",
        work_dir=root_dir / "Work",
        model_dir=data_dir / "models",
        ultralytics_dir=data_dir / "ultralytics",
        settings_path=data_dir / "settings.json",
    )


def _default_settings_payload(_paths: AppPaths) -> dict[str, object]:
    """기본 설정값 사본을 반환합니다."""
    return dict(DEFAULT_CONFIG)


def _normalize_model_path(paths: AppPaths, model_path: str, selected_model_name: str) -> str:
    """모델 경로를 실행 위치와 무관한 절대 경로로 정규화합니다."""
    if not model_path and not selected_model_name:
        return ""

    raw_path = Path(model_path) if model_path else paths.model_dir / selected_model_name
    if model_path and not raw_path.is_absolute():
        raw_path = paths.model_dir / raw_path.name
    try:
        raw_path.relative_to(paths.data_dir)
        if selected_model_name:
            return str((paths.model_dir / selected_model_name).resolve())
        return str(raw_path.resolve())
    except ValueError:
        pass

    if model_path:
        return str(raw_path.resolve())
    return str((paths.model_dir / selected_model_name).resolve())


def ensure_app_directories() -> AppPaths:
    """필수 디렉터리와 기본 settings.json 파일을 생성합니다."""
    paths = build_paths()
    for directory in (
        paths.root_dir,
        paths.data_dir,
        paths.document_dir,
        paths.work_dir,
        paths.model_dir,
        paths.ultralytics_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not paths.settings_path.exists():
        paths.settings_path.write_text(
            json.dumps(_default_settings_payload(paths), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return paths


def load_app_config() -> AppConfig:
    """settings.json을 읽고 누락된 값은 기본값으로 보정합니다."""
    paths = ensure_app_directories()
    raw = json.loads(paths.settings_path.read_text(encoding="utf-8"))
    defaults = _default_settings_payload(paths)
    merged = {**defaults, **raw}

    selected_model_name = str(merged.get("selected_model_name", defaults["selected_model_name"]))
    current_model_path = _normalize_model_path(
        paths,
        str(merged.get("model_path", defaults["model_path"])),
        selected_model_name,
    )

    return AppConfig(
        theme_mode=str(merged.get("theme_mode", defaults["theme_mode"])),
        use_gpu=bool(merged.get("use_gpu", defaults["use_gpu"])),
        training_mode=str(merged.get("training_mode", defaults["training_mode"])),
        selected_model_name=selected_model_name,
        selected_model_number=str(merged.get("selected_model_number", defaults["selected_model_number"])),
        selected_model_size=str(merged.get("selected_model_size", defaults["selected_model_size"])),
        model_path=current_model_path,
        train_epochs=str(merged.get("train_epochs", defaults["train_epochs"])),
        train_batch=str(merged.get("train_batch", defaults["train_batch"])),
        train_imgsz=str(merged.get("train_imgsz", defaults["train_imgsz"])),
        train_workers=str(merged.get("train_workers", defaults["train_workers"])),
        train_patience=str(merged.get("train_patience", defaults["train_patience"])),
        paths=paths,
    )


def save_app_config(config: AppConfig) -> None:
    """메모리의 설정 객체를 settings.json 파일로 저장합니다."""
    if config.paths is None:
        raise ValueError("경로 정보가 없는 설정은 저장할 수 없습니다.")

    payload = {
        "theme_mode": config.theme_mode,
        "use_gpu": config.use_gpu,
        "training_mode": config.training_mode,
        "selected_model_name": config.selected_model_name,
        "selected_model_number": config.selected_model_number,
        "selected_model_size": config.selected_model_size,
        "model_path": _normalize_model_path(config.paths, config.model_path, config.selected_model_name),
        "train_epochs": config.train_epochs,
        "train_batch": config.train_batch,
        "train_imgsz": config.train_imgsz,
        "train_workers": config.train_workers,
        "train_patience": config.train_patience,
    }
    settings_text = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path = config.paths.settings_path.with_suffix(".json.tmp")
    # 저장 도중 앱이 종료되어도 기존 settings.json이 깨지지 않도록 임시 파일을 교체합니다.
    temp_path.write_text(settings_text, encoding="utf-8")
    temp_path.replace(config.paths.settings_path)

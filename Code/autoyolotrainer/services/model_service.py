# -*- coding: utf-8 -*-
"""기본 YOLO 모델 파일명, 경로, 다운로드를 관리합니다."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import urlretrieve

from ..constants import MODEL_NUMBER_OPTIONS, MODEL_SIZE_OPTIONS
from ..models import AppConfig


def size_label_to_code(size_label: str) -> str:
    """표시용 크기 이름을 실제 파일명 접미부로 변환합니다."""
    for label, code in MODEL_SIZE_OPTIONS:
        if label == size_label:
            return code
    return ""


def size_code_to_label(size_code: str) -> str:
    """실제 파일명 접미부를 표시용 크기 이름으로 변환합니다."""
    for label, code in MODEL_SIZE_OPTIONS:
        if code == size_code:
            return label
    return ""


def build_model_filename(model_number: str, size_code: str) -> str:
    """번호와 크기가 모두 선택됐을 때만 모델 파일명을 만듭니다."""
    if not model_number or not size_code:
        return ""
    if model_number not in MODEL_NUMBER_OPTIONS:
        return ""
    return f"yolo{model_number}{size_code}.pt"


def parse_model_filename(model_path: str | Path) -> tuple[str, str, str]:
    """파일명에서 YOLO 번호와 크기 코드를 읽어 설정 동기화에 사용합니다."""
    filename = Path(model_path).name
    match = re.fullmatch(r"yolo(\d+)([nslmx])\.pt", filename, flags=re.IGNORECASE)
    if match is None:
        return filename, "", ""
    model_number = match.group(1)
    size_code = match.group(2).lower()
    if model_number not in MODEL_NUMBER_OPTIONS:
        return filename, "", ""
    if size_code_to_label(size_code) == "":
        return filename, "", ""
    return filename, model_number, size_code


def build_default_model_path(config: AppConfig, model_number: str, size_code: str) -> str:
    """프로젝트 Data\\models 기준 기본 모델 경로를 반환합니다."""
    if config.paths is None:
        return ""

    filename = build_model_filename(model_number, size_code)
    if not filename:
        return ""
    # 항상 절대 경로를 사용해 실행 위치에 따라 모델 저장 위치가 바뀌지 않게 합니다.
    return str((config.paths.model_dir / filename).resolve())


def update_selected_model(config: AppConfig, model_number: str, size_code: str, keep_existing_path: bool = True) -> None:
    """드롭다운 선택 상태를 설정에 반영합니다."""
    old_default = build_default_model_path(config, config.selected_model_number, config.selected_model_size)
    current_path = config.model_path
    config.selected_model_number = model_number
    config.selected_model_size = size_code

    new_default = build_default_model_path(config, model_number, size_code)
    new_filename = build_model_filename(model_number, size_code)

    # 드롭다운 선택이 완성되지 않았다면 초기 상태를 유지합니다.
    if not new_filename:
        config.selected_model_name = ""
        if not keep_existing_path or current_path == old_default:
            config.model_path = ""
        return

    config.selected_model_name = new_filename

    # 기본 경로 기반 선택일 때만 모델 경로를 함께 갱신합니다.
    if not keep_existing_path or not current_path or current_path == old_default:
        config.model_path = new_default


def model_download_url(model_number: str, size_code: str) -> str:
    """기본 YOLO 모델 다운로드 URL을 구성합니다."""
    filename = build_model_filename(model_number, size_code)
    if not filename:
        raise ValueError("다운로드할 YOLO 버전과 크기를 먼저 선택해주세요.")
    return f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{filename}"


def download_selected_model(config: AppConfig) -> Path:
    """현재 선택된 기본 모델을 Data\\models 아래로 다운로드합니다."""
    if config.paths is None:
        raise ValueError("경로 정보가 없는 설정은 사용할 수 없습니다.")
    if not config.selected_model_number or not config.selected_model_size:
        raise ValueError("다운로드할 YOLO 버전과 크기를 먼저 선택해주세요.")

    target_path = Path(build_default_model_path(config, config.selected_model_number, config.selected_model_size)).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        urlretrieve(model_download_url(config.selected_model_number, config.selected_model_size), target_path)

    # 다운로드가 완료되면 실제 저장된 모델 파일을 현재 선택 모델로 확정합니다.
    config.selected_model_name = target_path.name
    config.model_path = str(target_path)
    return target_path

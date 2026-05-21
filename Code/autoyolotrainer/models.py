# -*- coding: utf-8 -*-
"""AutoYoloTrainer에서 공통으로 사용하는 데이터 구조를 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppPaths:
    """프로그램이 사용하는 고정 디렉터리 경로 묶음입니다."""

    root_dir: Path
    code_dir: Path
    data_dir: Path
    document_dir: Path
    work_dir: Path
    model_dir: Path
    ultralytics_dir: Path
    settings_path: Path


@dataclass
class AppConfig:
    """프로그램 설정 파일에서 읽고 쓰는 값입니다."""

    theme_mode: str
    use_gpu: bool
    training_mode: str
    selected_model_name: str
    selected_model_number: str
    selected_model_size: str
    model_path: str
    train_epochs: str
    train_batch: str
    train_imgsz: str
    train_workers: str
    train_patience: str
    paths: AppPaths | None = None


@dataclass
class HardwareStatus:
    """화면에 표시할 하드웨어 및 실행 환경 정보를 담습니다."""

    python_command: list[str]
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    cpu_name: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    memory_gb: float
    os_name: str
    cuda_runtime_available: bool
    storage_summaries: list[str] = field(default_factory=list)


@dataclass
class SoftwareStatus:
    """확장 가능성을 위해 남겨 둔 소프트웨어 버전 정보 구조체입니다."""

    python_version: str
    package_versions: dict[str, str] = field(default_factory=dict)

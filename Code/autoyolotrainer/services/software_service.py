# -*- coding: utf-8 -*-
"""Python 및 AI 학습 관련 패키지 버전을 수집합니다."""

from __future__ import annotations

import sys
from importlib import metadata

from ..models import SoftwareStatus


def _safe_package_version(package_name: str) -> str:
    """패키지가 설치되어 있으면 버전을, 없으면 미설치를 반환합니다."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "미설치"
    except Exception:
        return "확인 실패"


def detect_software_versions() -> SoftwareStatus:
    """YOLO 학습과 관련된 핵심 Python 패키지 버전을 모아 반환합니다."""
    package_versions = {
        "ultralytics": _safe_package_version("ultralytics"),
        "torch": _safe_package_version("torch"),
        "torchvision": _safe_package_version("torchvision"),
        "torchaudio": _safe_package_version("torchaudio"),
        "onnx": _safe_package_version("onnx"),
        "onnxscript": _safe_package_version("onnxscript"),
        "opencv-python": _safe_package_version("opencv-python"),
        "PyQt6": _safe_package_version("PyQt6"),
    }
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return SoftwareStatus(python_version=python_version, package_versions=package_versions)

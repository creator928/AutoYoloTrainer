# -*- coding: utf-8 -*-
"""AutoYoloTrainer 기본 상수와 설정값을 정의합니다."""

from __future__ import annotations

DEFAULT_CONFIG = {
    "theme_mode": "dark",
    "use_gpu": True,
    "training_mode": "new",
    "selected_model_name": "",
    "selected_model_number": "",
    "selected_model_size": "",
    "model_path": "",
    "train_epochs": "100",
    "train_batch": "16",
    "train_imgsz": "640",
    "train_workers": "0",
    "train_patience": "50",
}

# 다운로드 버튼에서 제공하는 기본 모델 버전 목록입니다.
MODEL_NUMBER_OPTIONS = ["", "8", "11"]

# 사용자가 읽기 쉬운 표시명과 실제 파일명 접미사를 함께 관리합니다.
MODEL_SIZE_OPTIONS = [
    ("", ""),
    ("Nano", "n"),
    ("Small", "s"),
    ("Medium", "m"),
    ("Large", "l"),
    ("XLarge", "x"),
]

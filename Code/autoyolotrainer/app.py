# -*- coding: utf-8 -*-
"""AutoYoloTrainer PyQt 애플리케이션을 초기화하고 실행합니다."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from .config import ensure_app_directories, load_app_config
from .ui.main_window import MainWindow


def run() -> int:
    """필수 디렉터리를 준비한 뒤 메인 창을 실행합니다."""
    app = QApplication(sys.argv)
    app.setApplicationName("AutoYoloTrainer")

    ensure_app_directories()
    config = load_app_config()

    window = MainWindow(config)
    window.show()
    return app.exec()

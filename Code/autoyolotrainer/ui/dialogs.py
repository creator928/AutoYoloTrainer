# -*- coding: utf-8 -*-
"""프로그램 자체 설정을 다루는 대화상자입니다."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout

from ..models import AppConfig


class SettingsDialog(QDialog):
    """다크/라이트 모드만 다루는 프로그램 설정창입니다."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AutoYoloTrainer 설정")
        self.setModal(True)

        root_layout = QVBoxLayout(self)
        form = QFormLayout()
        root_layout.addLayout(form)

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(config.theme_mode)
        form.addRow("테마", self.theme_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def apply_to_config(self, config: AppConfig) -> None:
        """대화상자에서 선택한 값을 설정 객체에 반영합니다."""
        config.theme_mode = self.theme_combo.currentText()

# -*- coding: utf-8 -*-
"""UI에서 재사용하는 간단한 위젯을 정의합니다."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QPushButton, QWidget


class MarkedToggleButton(QPushButton):
    """체크 상태를 텍스트로 명확히 보여주는 토글 버튼입니다."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base_label = label
        self.setObjectName("ToggleButton")
        self.setCheckable(True)
        self.toggled.connect(self._refresh_marked_text)
        self._refresh_marked_text(False)

    def _refresh_marked_text(self, checked: bool) -> None:
        """현재 체크 상태에 맞는 표시 문구를 버튼 텍스트에 반영합니다."""
        marker = "[v]" if checked else "[ ]"
        self.setText(f"{marker} {self._base_label}")


class TrainingHistoryWidget(QWidget):
    """학습 진행률 이력을 간단한 선 그래프로 보여주는 위젯입니다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._history: list[float] = []
        self._line_color = QColor("#8fc7ff")
        self._grid_color = QColor("#4a4a4a")
        self._axis_color = QColor("#6a6a6a")
        self.setMinimumHeight(180)

    def clear_history(self) -> None:
        """기존 진행률 이력을 비우고 다시 그립니다."""
        self._history.clear()
        self.update()

    def append_progress(self, progress_percent: float) -> None:
        """새 진행률 값을 추가하고 최근 이력만 유지합니다."""
        clamped_value = max(0.0, min(100.0, progress_percent))
        self._history.append(clamped_value)
        # 지나치게 긴 이력은 잘라서 화면 갱신 비용을 제한합니다.
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """배경, 격자, 축, 진행률 선을 직접 그립니다."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(8, 8, -8, -8)
        if rect.width() <= 0 or rect.height() <= 0:
            return

        background_color = self.palette().window().color()
        painter.fillRect(rect, background_color)

        # 그래프 배경 격자를 그려 진행률 흐름을 읽기 쉽게 만듭니다.
        painter.setPen(QPen(self._grid_color, 1, Qt.PenStyle.DotLine))
        for row_index in range(1, 4):
            y_value = rect.top() + (rect.height() * row_index / 4.0)
            painter.drawLine(rect.left(), int(y_value), rect.right(), int(y_value))
        for column_index in range(1, 5):
            x_value = rect.left() + (rect.width() * column_index / 5.0)
            painter.drawLine(int(x_value), rect.top(), int(x_value), rect.bottom())

        # 기준 축을 먼저 그리고 그 위에 진행률 선을 그립니다.
        painter.setPen(QPen(self._axis_color, 1.2))
        painter.drawRect(rect)

        if not self._history:
            painter.setPen(QPen(self._axis_color, 1))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "학습 대기 중")
            return

        if len(self._history) == 1:
            x_positions = [rect.left()]
        else:
            x_positions = [
                rect.left() + rect.width() * index / (len(self._history) - 1)
                for index in range(len(self._history))
            ]

        points: list[QPointF] = []
        for index, progress_value in enumerate(self._history):
            y_ratio = 1.0 - (progress_value / 100.0)
            y_value = rect.top() + (rect.height() * y_ratio)
            points.append(QPointF(float(x_positions[index]), float(y_value)))

        painter.setPen(QPen(self._line_color, 2.2))
        for start_point, end_point in zip(points, points[1:]):
            painter.drawLine(start_point, end_point)

        # 가장 마지막 진행률 값은 점으로 강조해 현재 위치를 쉽게 알 수 있게 합니다.
        painter.setBrush(self._line_color)
        painter.setPen(QPen(self._line_color, 1))
        last_point = points[-1]
        painter.drawEllipse(last_point, 3.5, 3.5)

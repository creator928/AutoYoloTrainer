# -*- coding: utf-8 -*-
"""기존 AutoLabeler 계열과 유사한 회색조 라이트/다크 테마 스타일을 제공합니다."""

from __future__ import annotations


def build_stylesheet(theme_mode: str) -> str:
    """테마 모드에 맞는 전체 애플리케이션 스타일시트를 반환합니다."""
    if theme_mode == "dark":
        return """
        QWidget {
            background-color: #1f1f1f;
            color: #f2f2f2;
            font-family: 'Malgun Gothic';
            font-size: 9.5pt;
        }
        QMainWindow, QWidget {
            background-color: #1f1f1f;
        }
        QGroupBox {
            background-color: #2a2a2a;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 8px;
            padding: 4px 3px 3px 3px;
            font-weight: 700;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 3px 0 3px;
        }
        QLabel#ProgramTitle {
            font-size: 18pt;
            font-weight: 700;
            padding: 0 2px 2px 2px;
        }
        QLabel#GroupMainText {
            font-size: 11.5pt;
            font-weight: 700;
        }
        QLabel#GroupSubText {
            font-size: 9.6pt;
        }
        QLabel#GroupHintText {
            color: #c7c7c7;
        }
        QLabel#StateBadge {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 4px;
            font-weight: 700;
        }
        QLineEdit, QComboBox {
            background-color: #232323;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 3px 4px;
        }
        QLineEdit[readOnly="true"] {
            background-color: #3a3a3a;
            border: 1px solid #505050;
            color: #f2f2f2;
        }
        QListWidget {
            background-color: #232323;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 2px;
        }
        QListWidget::item {
            padding: 2px 4px;
            border-bottom: 1px solid #2d2d2d;
        }
        QListWidget#LogList {
            padding: 1px;
        }
        QListWidget#LogList::item {
            margin: 1px;
            padding: 1px;
        }
        QListWidget::item:selected {
            background-color: #355b82;
            color: #ffffff;
        }
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 3px 6px;
            margin: 1px;
            min-width: 72px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        QPushButton#ToggleButton {
            text-align: center;
            min-width: 78px;
            font-size: 9.6pt;
            font-weight: 700;
        }
        QPushButton#ToggleButton:checked {
            background-color: #8fc7ff;
            border: 1px solid #b9dcff;
            color: #101820;
        }
        QProgressBar {
            background-color: #232323;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            min-height: 18px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #8fc7ff;
            border-radius: 3px;
        }
        QCheckBox, QRadioButton {
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #d8d8d8;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 1px solid #d8d8d8;
        }
        QCheckBox::indicator:unchecked {
            border: 1px solid #d8d8d8;
        }
        QRadioButton::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #d8d8d8;
            border-radius: 8px;
        }
        QRadioButton::indicator:checked {
            border: 1px solid #d8d8d8;
        }
        QRadioButton::indicator:unchecked {
            border: 1px solid #d8d8d8;
        }
        """

    return """
    QWidget {
        background-color: #f4f4f4;
        color: #202020;
        font-family: 'Malgun Gothic';
        font-size: 9.5pt;
    }
    QMainWindow, QWidget {
        background-color: #f4f4f4;
    }
    QGroupBox {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        margin-top: 8px;
        padding: 4px 3px 3px 3px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 6px;
        padding: 0 3px 0 3px;
    }
    QLabel#ProgramTitle {
        font-size: 18pt;
        font-weight: 700;
        padding: 0 2px 2px 2px;
    }
    QLabel#GroupMainText {
        font-size: 11.5pt;
        font-weight: 700;
    }
    QLabel#GroupSubText {
        font-size: 9.6pt;
    }
    QLabel#GroupHintText {
        color: #555555;
    }
    QLabel#StateBadge {
        background-color: #efefef;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px;
        font-weight: 700;
    }
    QLineEdit, QComboBox {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 3px 4px;
    }
    QLineEdit[readOnly="true"] {
        background-color: #e2e2e2;
        border: 1px solid #bdbdbd;
        color: #202020;
    }
    QListWidget {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 2px;
    }
    QListWidget::item {
        padding: 2px 4px;
        border-bottom: 1px solid #e7e7e7;
    }
    QListWidget#LogList {
        padding: 1px;
    }
    QListWidget#LogList::item {
        margin: 1px;
        padding: 1px;
    }
    QListWidget::item:selected {
        background-color: #8fc7ff;
        color: #102030;
    }
    QPushButton {
        background-color: #e5e5e5;
        border: 1px solid #c8c8c8;
        border-radius: 4px;
        padding: 3px 6px;
        margin: 1px;
        min-width: 72px;
    }
    QPushButton:hover {
        background-color: #d8d8d8;
    }
    QPushButton#ToggleButton {
        text-align: center;
        min-width: 78px;
        font-size: 9.6pt;
        font-weight: 700;
    }
    QPushButton#ToggleButton:checked {
        background-color: #8fc7ff;
        border: 1px solid #5faaf4;
        color: #0d2238;
    }
    QProgressBar {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        min-height: 18px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #8fc7ff;
        border-radius: 3px;
    }
    QCheckBox, QRadioButton {
        spacing: 6px;
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border: 1px solid #4a4a4a;
        border-radius: 3px;
    }
    QCheckBox::indicator:checked {
        border: 1px solid #4a4a4a;
    }
    QCheckBox::indicator:unchecked {
        border: 1px solid #4a4a4a;
    }
    QRadioButton::indicator {
        width: 15px;
        height: 15px;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
    }
    QRadioButton::indicator:checked {
        border: 1px solid #4a4a4a;
    }
    QRadioButton::indicator:unchecked {
        border: 1px solid #4a4a4a;
    }
    """

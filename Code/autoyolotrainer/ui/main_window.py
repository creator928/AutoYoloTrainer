# -*- coding: utf-8 -*-
"""AutoYoloTrainer 메인 윈도우를 정의합니다."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import psutil

from PyQt6.QtCore import QEvent, QThread, QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import save_app_config
from ..constants import MODEL_NUMBER_OPTIONS, MODEL_SIZE_OPTIONS
from ..models import AppConfig
from ..services.dataset_service import build_training_project, scan_dataset_pairs
from ..services.hardware_service import detect_hardware, query_gpu_runtime_usage
from ..services.model_service import (
    download_selected_model,
    parse_model_filename,
    size_code_to_label,
    size_label_to_code,
    update_selected_model,
)
from ..services.process_service import build_clean_python_env, clean_windows_dll_search_path
from ..services.training_worker import TrainingRequest, TrainingWorker
from .dialogs import SettingsDialog
from .styles import build_stylesheet
from .widgets import MarkedToggleButton, TrainingHistoryWidget


class MainWindow(QMainWindow):
    """좌측 설정 패널과 우측 확장 패널을 가지는 메인 창입니다."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.hardware_status = detect_hardware()
        self.selected_dataset_dir: Path | None = None
        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self.training_process_pid: int | None = None
        self.export_model_path: Path | None = None
        self.detect_image_path: Path | None = None
        self.preview_pixmap: QPixmap | None = None
        self.training_resource_timer = QTimer(self)
        self.training_resource_timer.setInterval(1000)
        self.training_resource_timer.timeout.connect(self.update_training_resource_summary)
        self._current_training_progress_text = "-"
        self._current_training_epoch_text = "-"

        self.setWindowTitle("AutoYoloTrainer")
        self.resize(1680, 940)
        self._build_ui()
        self._apply_theme()
        self.showMaximized()

    def _build_ui(self) -> None:
        """3분할 메인 화면과 각 그룹 박스를 구성합니다."""
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        # 상단의 빈 공간이 커지지 않도록 전체 레이아웃을 상단 정렬로 유지합니다.
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.program_title_label = QLabel("AutoYoloTrainer", central)
        self.program_title_label.setObjectName("ProgramTitle")
        root_layout.addWidget(self.program_title_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        root_layout.addWidget(splitter)

        left_panel = QWidget(splitter)
        left_panel.setMinimumWidth(0)
        left_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.addWidget(self._build_program_group())
        left_layout.addWidget(self._build_hardware_group())
        left_layout.addWidget(self._build_software_group())
        left_layout.addWidget(self._build_training_folder_group())
        left_layout.addWidget(self._build_training_option_group())
        left_layout.addStretch(1)

        center_panel = QWidget(splitter)
        center_panel.setMinimumWidth(0)
        center_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        center_splitter = QSplitter(Qt.Orientation.Vertical, center_panel)
        center_splitter.addWidget(self._build_log_group())
        center_splitter.addWidget(self._build_training_info_group())
        center_splitter.setSizes([520, 360])
        center_splitter.setChildrenCollapsible(False)
        center_layout.addWidget(center_splitter, stretch=1)
        # 학습 상태는 로그와 그래프를 보는 중앙 영역의 하단에 고정합니다.
        center_layout.addWidget(self._build_training_status_group(), stretch=0)

        right_panel = QWidget(splitter)
        right_panel.setMinimumWidth(0)
        right_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        right_splitter = QSplitter(Qt.Orientation.Vertical, right_panel)
        right_splitter.addWidget(self._build_model_export_group())
        detection_panel = QWidget(right_splitter)
        detection_layout = QVBoxLayout(detection_panel)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        detection_layout.setSpacing(4)
        detection_layout.addWidget(self._build_image_detection_group(), stretch=3)
        detection_layout.addWidget(self._build_detection_result_group(), stretch=1)
        right_splitter.addWidget(detection_panel)
        right_splitter.setSizes([150, 730])
        right_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(right_splitter, stretch=1)

        # 좌측 설정 영역이 전체 가로 폭의 약 25%를 차지하도록 초기 비율을 맞춥니다.
        splitter.setSizes([420, 630, 630])
        splitter.setChildrenCollapsible(False)

        self._sync_ui_from_config()
        self._refresh_hardware_labels()
        self._refresh_software_labels()
        self._update_dataset_controls_enabled()
        self._refresh_training_info_panel()
        self._apply_responsive_widget_policy()
        self._log_initial_state()

    def _build_program_group(self) -> QGroupBox:
        """프로그램 자체 설정 그룹을 구성합니다."""
        group = QGroupBox("프로그램 환경", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.settings_button = QPushButton("프로그램 설정", group)
        self.settings_button.clicked.connect(self.open_settings)
        layout.addWidget(self.settings_button)

        self.runtime_refresh_button = QPushButton("환경 새로고침", group)
        self.runtime_refresh_button.clicked.connect(self.refresh_runtime_status)
        layout.addWidget(self.runtime_refresh_button)
        layout.addStretch(1)
        return group

    def _build_hardware_group(self) -> QGroupBox:
        """학습 하드웨어 환경 그룹을 구성합니다."""
        group = QGroupBox("학습 하드웨어 환경", self)
        layout = QFormLayout(group)
        # 그룹 제목이 잘리지 않도록 상단 여백을 최소 수준으로 유지합니다.
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)

        self.device_name_edit = QLineEdit(group)
        self.device_name_edit.setReadOnly(True)

        device_row = QWidget(group)
        device_row_layout = QHBoxLayout(device_row)
        device_row_layout.setContentsMargins(0, 0, 0, 0)
        device_row_layout.setSpacing(4)
        device_row_layout.addWidget(QLabel("학습 장치", device_row))
        device_row_layout.addWidget(self.device_name_edit, stretch=1)

        # 체크 상태를 텍스트로 명확히 보이게 토글 버튼으로 표시합니다.
        self.use_gpu_button = MarkedToggleButton("GPU 사용", device_row)
        self.use_gpu_button.toggled.connect(self.on_use_gpu_toggled)
        device_row_layout.addWidget(self.use_gpu_button)
        layout.addRow(device_row)

        self.os_value_label = QLabel(group)
        self.cpu_value_label = QLabel(group)
        self.gpu_value_label = QLabel(group)
        self.ram_value_label = QLabel(group)
        self.storage_value_label = QLabel(group)

        for label in (
            self.os_value_label,
            self.cpu_value_label,
            self.gpu_value_label,
            self.ram_value_label,
            self.storage_value_label,
        ):
            label.setWordWrap(True)

        layout.addRow("OS", self.os_value_label)
        layout.addRow("CPU", self.cpu_value_label)
        layout.addRow("GPU", self.gpu_value_label)
        layout.addRow("RAM", self.ram_value_label)
        layout.addRow("Storage", self.storage_value_label)
        return group

    def _build_software_group(self) -> QGroupBox:
        """학습 소프트웨어 환경 그룹을 구성합니다."""
        group = QGroupBox("학습 소프트웨어 환경", self)
        layout = QFormLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)

        self.yolo_ver_edit = QLineEdit(group)
        self.yolo_ver_edit.setReadOnly(True)
        self.yolo_path_edit = QLineEdit(group)
        self.yolo_path_edit.setReadOnly(True)
        layout.addRow("Yolo Ver.", self.yolo_ver_edit)
        layout.addRow("Yolo Path", self.yolo_path_edit)

        select_row = QWidget(group)
        select_row_layout = QHBoxLayout(select_row)
        select_row_layout.setContentsMargins(0, 0, 0, 0)
        select_row_layout.setSpacing(4)

        self.yolo_select_button = QPushButton("Yolo 선택", select_row)
        self.yolo_select_button.clicked.connect(self.select_model_path)
        select_row_layout.addWidget(self.yolo_select_button)

        self.model_number_combo = QComboBox(select_row)
        self.model_number_combo.addItems(MODEL_NUMBER_OPTIONS)
        self.model_number_combo.currentTextChanged.connect(self.on_model_selection_changed)
        select_row_layout.addWidget(self.model_number_combo)

        self.model_size_combo = QComboBox(select_row)
        self.model_size_combo.addItems([label for label, _code in MODEL_SIZE_OPTIONS])
        self.model_size_combo.currentTextChanged.connect(self.on_model_selection_changed)
        select_row_layout.addWidget(self.model_size_combo)

        self.model_download_button = QPushButton("다운로드", select_row)
        self.model_download_button.clicked.connect(self.download_model)
        select_row_layout.addWidget(self.model_download_button)

        layout.addRow(select_row)
        return group

    def _build_training_folder_group(self) -> QGroupBox:
        """학습 폴더 설정 그룹을 구성합니다."""
        group = QGroupBox("학습 폴더 설정", self)
        layout = QFormLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)

        training_mode_row = QWidget(group)
        training_mode_layout = QHBoxLayout(training_mode_row)
        training_mode_layout.setContentsMargins(0, 0, 0, 0)
        training_mode_layout.setSpacing(4)
        training_mode_layout.addWidget(QLabel("학습 모드", training_mode_row))

        self.new_training_button = MarkedToggleButton("신규 학습", training_mode_row)
        self.resume_training_button = MarkedToggleButton("추가 학습", training_mode_row)
        self.new_training_button.toggled.connect(self.on_training_mode_changed)
        self.resume_training_button.toggled.connect(self.on_training_mode_changed)
        training_mode_layout.addWidget(self.new_training_button)
        training_mode_layout.addWidget(self.resume_training_button)
        training_mode_layout.addStretch(1)
        layout.addRow(training_mode_row)

        workspace_row = QWidget(group)
        workspace_layout = QHBoxLayout(workspace_row)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(4)
        workspace_layout.addWidget(QLabel("워크스페이스", workspace_row))
        self.workspace_path_edit = QLineEdit(group)
        self.workspace_path_edit.setReadOnly(True)
        workspace_layout.addWidget(self.workspace_path_edit, stretch=1)
        layout.addRow(workspace_row)

        project_name_row = QWidget(group)
        project_name_layout = QHBoxLayout(project_name_row)
        project_name_layout.setContentsMargins(0, 0, 0, 0)
        project_name_layout.setSpacing(4)

        self.project_name_edit = QLineEdit(project_name_row)
        # 프로젝트명 변경 즉시 중앙 학습 정보 패널과 결과 경로를 같은 값으로 갱신합니다.
        self.project_name_edit.textChanged.connect(lambda _text: self._refresh_training_info_panel())
        self.project_name_edit.editingFinished.connect(self.on_project_name_edit_finished)
        project_name_layout.addWidget(self.project_name_edit, stretch=1)

        self.project_select_button = QPushButton("선택", project_name_row)
        self.project_select_button.clicked.connect(self.select_training_project_folder)
        project_name_layout.addWidget(self.project_select_button)
        layout.addRow("학습 프로젝트명", project_name_row)

        dataset_row = QWidget(group)
        dataset_layout = QHBoxLayout(dataset_row)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(4)
        self.dataset_path_edit = QLineEdit(group)
        self.dataset_path_edit.setReadOnly(True)
        dataset_layout.addWidget(self.dataset_path_edit, stretch=1)
        self.dataset_load_button = QPushButton("데이터셋 불러오기", dataset_row)
        self.dataset_load_button.clicked.connect(self.load_dataset_folder)
        dataset_layout.addWidget(self.dataset_load_button)
        layout.addRow("데이터셋", dataset_row)

        dataset_option_row = QWidget(group)
        dataset_option_layout = QHBoxLayout(dataset_option_row)
        dataset_option_layout.setContentsMargins(0, 0, 0, 0)
        dataset_option_layout.setSpacing(4)
        self.shuffle_toggle_button = MarkedToggleButton("셔플", dataset_option_row)
        dataset_option_layout.addWidget(self.shuffle_toggle_button)

        self.train_ratio_edit = QLineEdit("80", dataset_option_row)
        self.train_ratio_edit.setMinimumWidth(31)
        self.train_ratio_edit.setMaximumWidth(36)
        dataset_option_layout.addWidget(QLabel("Train", dataset_option_row))
        dataset_option_layout.addWidget(self.train_ratio_edit)
        dataset_option_layout.addWidget(QLabel("%", dataset_option_row))

        self.val_ratio_edit = QLineEdit("10", dataset_option_row)
        self.val_ratio_edit.setMinimumWidth(31)
        self.val_ratio_edit.setMaximumWidth(36)
        dataset_option_layout.addWidget(QLabel("Val", dataset_option_row))
        dataset_option_layout.addWidget(self.val_ratio_edit)
        dataset_option_layout.addWidget(QLabel("%", dataset_option_row))

        self.test_ratio_edit = QLineEdit("10", dataset_option_row)
        self.test_ratio_edit.setMinimumWidth(31)
        self.test_ratio_edit.setMaximumWidth(36)
        dataset_option_layout.addWidget(QLabel("Test", dataset_option_row))
        dataset_option_layout.addWidget(self.test_ratio_edit)
        dataset_option_layout.addWidget(QLabel("%", dataset_option_row))
        dataset_option_layout.addStretch(1)
        layout.addRow("분할 비율", dataset_option_row)

        dataset_action_row = QWidget(group)
        dataset_action_layout = QHBoxLayout(dataset_action_row)
        dataset_action_layout.setContentsMargins(0, 0, 0, 0)
        dataset_action_layout.setSpacing(4)
        self.dataset_copy_button = QPushButton("복사", dataset_action_row)
        self.dataset_copy_button.clicked.connect(lambda: self.process_dataset_distribution(move_files=False))
        dataset_action_layout.addWidget(self.dataset_copy_button)
        self.dataset_move_button = QPushButton("이동", dataset_action_row)
        self.dataset_move_button.clicked.connect(lambda: self.process_dataset_distribution(move_files=True))
        dataset_action_layout.addWidget(self.dataset_move_button)
        dataset_action_layout.addStretch(1)
        layout.addRow("데이터셋 처리", dataset_action_row)
        return group

    def _build_training_option_group(self) -> QGroupBox:
        """학습 설정 그룹을 구성합니다."""
        group = QGroupBox("학습 설정", self)
        layout = QFormLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(3)

        device_row = QWidget(group)
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(4)
        self.training_device_edit = QLineEdit(group)
        self.training_device_edit.setReadOnly(True)
        device_layout.addWidget(self.training_device_edit, stretch=1)
        self.training_gpu_button = MarkedToggleButton("GPU 사용", device_row)
        self.training_gpu_button.toggled.connect(self.on_use_gpu_toggled)
        device_layout.addWidget(self.training_gpu_button)
        layout.addRow("학습 장치", device_row)

        model_row = QWidget(group)
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        self.training_model_edit = QLineEdit(group)
        self.training_model_edit.setReadOnly(True)
        model_layout.addWidget(self.training_model_edit, stretch=1)
        self.training_model_select_button = QPushButton("모델 선택", model_row)
        self.training_model_select_button.clicked.connect(self.select_model_path)
        model_layout.addWidget(self.training_model_select_button)
        layout.addRow("학습 모델", model_row)

        # 자주 조정하는 YOLO 학습 옵션만 간단한 입력칸으로 노출합니다.
        self.train_epochs_edit = QLineEdit(group)
        self.train_batch_edit = QLineEdit(group)
        self.train_imgsz_edit = QLineEdit(group)
        self.train_workers_edit = QLineEdit(group)
        self.train_patience_edit = QLineEdit(group)

        self.train_epochs_edit.editingFinished.connect(self.save_training_option_inputs)
        self.train_batch_edit.editingFinished.connect(self.save_training_option_inputs)
        self.train_imgsz_edit.editingFinished.connect(self.save_training_option_inputs)
        self.train_workers_edit.editingFinished.connect(self.save_training_option_inputs)
        self.train_patience_edit.editingFinished.connect(self.save_training_option_inputs)

        layout.addRow("Epoch", self.train_epochs_edit)
        layout.addRow("Batch", self.train_batch_edit)
        layout.addRow("Image Size", self.train_imgsz_edit)
        layout.addRow("Workers", self.train_workers_edit)
        layout.addRow("Patience", self.train_patience_edit)

        # 현재 설정값을 그대로 사용해 학습을 시작합니다.
        self.training_start_button = QPushButton("학습 시작", group)
        self.training_start_button.clicked.connect(self.start_training)
        layout.addRow(self.training_start_button)
        return group

    def _build_training_status_group(self) -> QGroupBox:
        """학습 상태 그룹을 구성합니다."""
        group = QGroupBox("학습 상태", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        # 진행률, 에포크, 자원 사용량을 두 줄에 요약해 공간을 효율적으로 사용합니다.
        self.training_status_summary_label = QLabel(group)
        self.training_status_summary_label.setObjectName("StateBadge")
        self.training_status_summary_label.setWordWrap(True)

        self.training_progress_bar = QProgressBar(group)
        self.training_progress_bar.setRange(0, 100)
        self.training_progress_bar.setValue(0)
        self.training_progress_bar.setTextVisible(False)

        layout.addWidget(self.training_status_summary_label)
        layout.addWidget(self.training_progress_bar)
        return group

    def _build_log_group(self) -> QGroupBox:
        """프로그램과 학습 관련 로그를 출력하는 중앙 로그 그룹을 구성합니다."""
        group = QGroupBox("실행 로그", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        self.log_list_widget = QListWidget(group)
        self.log_list_widget.setObjectName("LogList")
        self.log_list_widget.setWordWrap(True)
        self.log_list_widget.setUniformItemSizes(False)
        self.log_list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.log_list_widget.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.log_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.log_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.log_list_widget)
        return group

    def _build_training_info_group(self) -> QGroupBox:
        """학습 정보와 진행 그래프를 출력하는 중앙 정보 그룹을 구성합니다."""
        group = QGroupBox("학습 정보", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        # 현재 선택된 프로젝트와 결과 폴더를 요약해 보여줍니다.
        self.training_info_summary_label = QLabel(group)
        self.training_info_summary_label.setObjectName("StateBadge")
        self.training_info_summary_label.setWordWrap(True)

        # 에포크가 쌓이면서 진행률 추이를 작은 그래프로 확인할 수 있습니다.
        self.training_history_widget = TrainingHistoryWidget(group)

        layout.addWidget(self.training_info_summary_label)
        layout.addWidget(self.training_history_widget)
        return group

    def _build_model_export_group(self) -> QGroupBox:
        """우측 모델 선택과 export 기능 그룹을 구성합니다."""
        group = QGroupBox("모델 변환", self)
        layout = QFormLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)

        model_row = QWidget(group)
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)

        self.export_model_path_edit = QLineEdit(model_row)
        self.export_model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.export_model_path_edit, stretch=1)

        self.export_model_select_button = QPushButton("PT 선택", model_row)
        self.export_model_select_button.clicked.connect(self.select_export_model_path)
        model_layout.addWidget(self.export_model_select_button)
        layout.addRow("모델", model_row)

        export_row = QWidget(group)
        export_layout = QHBoxLayout(export_row)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(4)

        self.export_format_combo = QComboBox(export_row)
        self.export_format_combo.addItems(["onnx", "openvino", "engine", "torchscript", "tflite"])
        export_layout.addWidget(self.export_format_combo, stretch=1)

        self.export_model_button = QPushButton("Export", export_row)
        self.export_model_button.clicked.connect(self.export_selected_model)
        export_layout.addWidget(self.export_model_button)
        layout.addRow("형식", export_row)
        return group

    def _build_image_detection_group(self) -> QGroupBox:
        """우측 이미지 선택과 디텍팅 결과 뷰어 그룹을 구성합니다."""
        group = QGroupBox("이미지 디텍팅", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        image_row = QWidget(group)
        image_layout = QHBoxLayout(image_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)

        self.detect_image_path_edit = QLineEdit(image_row)
        self.detect_image_path_edit.setReadOnly(True)
        image_layout.addWidget(self.detect_image_path_edit, stretch=1)

        self.detect_image_select_button = QPushButton("이미지 선택", image_row)
        self.detect_image_select_button.clicked.connect(self.select_detection_image)
        image_layout.addWidget(self.detect_image_select_button)

        self.detect_run_button = QPushButton("디텍팅", image_row)
        self.detect_run_button.clicked.connect(self.run_detection_on_image)
        image_layout.addWidget(self.detect_run_button)
        layout.addWidget(image_row)

        self.detect_preview_label = QLabel("이미지를 선택해주세요.", group)
        self.detect_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detect_preview_label.setMinimumSize(1, 1)
        self.detect_preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.detect_preview_label.setWordWrap(True)
        self.detect_preview_label.setScaledContents(False)

        self.detect_preview_scroll = QScrollArea(group)
        self.detect_preview_scroll.setWidgetResizable(True)
        self.detect_preview_scroll.setWidget(self.detect_preview_label)
        self.detect_preview_scroll.viewport().installEventFilter(self)
        layout.addWidget(self.detect_preview_scroll, stretch=1)
        return group

    def _build_detection_result_group(self) -> QGroupBox:
        """검출된 객체 요약을 출력하는 그룹을 구성합니다."""
        group = QGroupBox("검출 결과", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        self.detection_result_label = QLabel("검출 결과 없음", group)
        self.detection_result_label.setObjectName("StateBadge")
        self.detection_result_label.setWordWrap(True)
        layout.addWidget(self.detection_result_label)
        return group

    def _build_placeholder_group(self, title: str, line1: str, line2: str) -> QGroupBox:
        """추후 확장 영역을 안내하는 그룹을 구성합니다."""
        group = QGroupBox(title, self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        line1_label = QLabel(line1, group)
        line1_label.setWordWrap(True)
        line2_label = QLabel(line2, group)
        line2_label.setWordWrap(True)
        layout.addWidget(line1_label)
        layout.addWidget(line2_label)
        layout.addStretch(1)
        return group

    def _apply_theme(self) -> None:
        """현재 설정된 테마를 전체 스타일시트에 반영합니다."""
        self.setStyleSheet(build_stylesheet(self.config.theme_mode))

    def _apply_responsive_widget_policy(self) -> None:
        """좁은 화면에서도 입력칸이 줄어들고 라벨이 줄바꿈되도록 크기 정책을 보정합니다."""
        for form_layout in self.findChildren(QFormLayout):
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        ratio_inputs = {
            self.train_ratio_edit,
            self.val_ratio_edit,
            self.test_ratio_edit,
        }
        for line_edit in self.findChildren(QLineEdit):
            if line_edit in ratio_inputs:
                # 비율 입력칸은 항상 보이고 수정 가능해야 하므로 폭을 고정해 보장합니다.
                line_edit.setMinimumWidth(31)
                line_edit.setMaximumWidth(36)
                line_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            elif line_edit.isReadOnly():
                # 경로 표시용 읽기 전용 입력칸은 폭이 부족하면 줄어들도록 둡니다.
                line_edit.setMinimumWidth(0)
                line_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            else:
                # 사용자가 직접 입력하는 일반 입력칸은 최소한의 가시성을 유지합니다.
                line_edit.setMinimumWidth(72)
                line_edit.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        for combo_box in self.findChildren(QComboBox):
            combo_box.setMinimumWidth(0)
            combo_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        for label in self.findChildren(QLabel):
            if label is self.program_title_label:
                continue
            if label is self.detect_preview_label:
                # 이미지 뷰어는 스크롤 영역 크기에 맞춰 직접 스케일링하므로 일반 라벨 정책을 적용하지 않습니다.
                label.setMinimumSize(1, 1)
                label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                continue
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        for group_box in self.findChildren(QGroupBox):
            group_box.setMinimumWidth(0)
            group_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.log_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.training_history_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _add_log(self, message: str) -> None:
        """중앙 로그 리스트에 시간과 함께 새 로그를 추가합니다."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_list_widget.addItem(f"[{timestamp}] {message}")
        self.log_list_widget.scrollToBottom()
        # 로그가 과도하게 많아지면 최근 500개만 유지합니다.
        while self.log_list_widget.count() > 500:
            self.log_list_widget.takeItem(0)

    def _log_initial_state(self) -> None:
        """프로그램 시작 직후 주요 설정과 초기 상태를 로그에 남깁니다."""
        self._add_log("프로그램 시작")
        self._add_log(f"테마 {self.config.theme_mode}")
        self._add_log(f"학습 모드 {self.config.training_mode}")
        self._add_log(f"워크스페이스 {self.workspace_path_edit.text()}")
        self._add_log(f"초기 프로젝트명 {self.project_name_edit.text()}")
        self._add_log(
            "장치 "
            + (
                self.hardware_status.gpu_name
                if self.config.use_gpu and self.hardware_status.gpu_available
                else self.hardware_status.cpu_name
            )
        )

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        """이미지 뷰어 영역이 바뀌면 현재 이미지를 새 크기에 맞춰 다시 그립니다."""
        if (
            hasattr(self, "detect_preview_scroll")
            and watched is self.detect_preview_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self._refresh_preview_pixmap()
        return super().eventFilter(watched, event)

    def _refresh_training_info_panel(self) -> None:
        """현재 프로젝트, 모델, 데이터셋, 결과 폴더 정보를 중앙 요약 패널에 반영합니다."""
        workspace_text = self.workspace_path_edit.text() or "-"
        project_name = self.project_name_edit.text().strip() or "-"
        model_name = self.config.selected_model_name or "-"
        dataset_text = self.dataset_path_edit.text() or "-"
        result_dir_text = "-"
        if self.config.paths is not None and project_name != "-":
            result_dir_text = str(self.config.paths.work_dir / project_name / "result")

        self.training_info_summary_label.setText(
            "\n".join(
                [
                    f"프로젝트명  {project_name}",
                    f"워크스페이스  {workspace_text}",
                    f"학습 모델  {model_name}",
                    f"데이터셋  {dataset_text}",
                    f"결과 폴더  {result_dir_text}",
                ]
            )
        )

    def _sync_ui_from_config(self) -> None:
        """설정 객체의 현재 값을 각 입력 컨트롤에 반영합니다."""
        self.new_training_button.blockSignals(True)
        self.resume_training_button.blockSignals(True)
        self.new_training_button.setChecked(self.config.training_mode == "new")
        self.resume_training_button.setChecked(self.config.training_mode == "resume")
        self.new_training_button.blockSignals(False)
        self.resume_training_button.blockSignals(False)
        self.new_training_button._refresh_marked_text(self.new_training_button.isChecked())
        self.resume_training_button._refresh_marked_text(self.resume_training_button.isChecked())

        self.model_number_combo.blockSignals(True)
        self.model_size_combo.blockSignals(True)
        self.model_number_combo.setCurrentText(self.config.selected_model_number)
        self.model_size_combo.setCurrentText(size_code_to_label(self.config.selected_model_size))
        self.model_number_combo.blockSignals(False)
        self.model_size_combo.blockSignals(False)
        self.yolo_ver_edit.setText(self.config.selected_model_name)
        self.yolo_path_edit.setText(self.config.model_path)
        self.workspace_path_edit.setText(str(self.config.paths.work_dir) if self.config.paths else "")
        self.project_name_edit.setText(datetime.now().strftime("work_%Y%m%d_%H%M%S"))
        self.shuffle_toggle_button.setChecked(True)
        self.shuffle_toggle_button._refresh_marked_text(True)
        self.train_epochs_edit.setText(self.config.train_epochs)
        self.train_batch_edit.setText(self.config.train_batch)
        self.train_imgsz_edit.setText(self.config.train_imgsz)
        self.train_workers_edit.setText(self.config.train_workers)
        self.train_patience_edit.setText(self.config.train_patience)
        self._reset_training_status_ui()
        self._refresh_training_info_panel()

    def _refresh_hardware_labels(self) -> None:
        """감지된 하드웨어 상태를 화면 값으로 갱신합니다."""
        self.use_gpu_button.blockSignals(True)
        self.use_gpu_button.setChecked(self.hardware_status.gpu_available and self.config.use_gpu)
        self.use_gpu_button.setEnabled(self.hardware_status.gpu_available)
        self.use_gpu_button.blockSignals(False)
        self.use_gpu_button._refresh_marked_text(self.use_gpu_button.isChecked())

        self.training_gpu_button.blockSignals(True)
        self.training_gpu_button.setChecked(self.hardware_status.gpu_available and self.config.use_gpu)
        self.training_gpu_button.setEnabled(self.hardware_status.gpu_available)
        self.training_gpu_button.blockSignals(False)
        self.training_gpu_button._refresh_marked_text(self.training_gpu_button.isChecked())

        if self.use_gpu_button.isChecked() and self.hardware_status.gpu_available:
            active_device_text = self.hardware_status.gpu_name or "GPU"
        else:
            active_device_text = self.hardware_status.cpu_name

        self.device_name_edit.setText(active_device_text)
        self.training_device_edit.setText(active_device_text)
        self.os_value_label.setText(self.hardware_status.os_name)
        self.cpu_value_label.setText(
            f"{self.hardware_status.cpu_name} / "
            f"{self.hardware_status.cpu_physical_cores}코어 "
            f"{self.hardware_status.cpu_logical_cores}스레드"
        )
        if self.hardware_status.gpu_available:
            self.gpu_value_label.setText(f"{self.hardware_status.gpu_name} / {self.hardware_status.gpu_memory_gb}GB")
        else:
            self.gpu_value_label.setText("감지된 GPU 없음")
        self.ram_value_label.setText(f"{self.hardware_status.memory_gb}GB")
        self.storage_value_label.setText(
            "\n".join(self.hardware_status.storage_summaries)
            if self.hardware_status.storage_summaries
            else "확인된 드라이브 없음"
        )

    def _refresh_software_labels(self) -> None:
        """현재 모델 표시값을 새로 그립니다."""
        self.yolo_ver_edit.setText(self.config.selected_model_name)
        self.yolo_path_edit.setText(self.config.model_path)
        self.training_model_edit.setText(self.config.selected_model_name)
        self._refresh_training_info_panel()

    def _update_dataset_controls_enabled(self) -> None:
        """데이터셋 준비 여부에 따라 분할 관련 컨트롤의 활성 상태를 바꿉니다."""
        enabled = self.selected_dataset_dir is not None
        # 비율 입력과 셔플 여부는 데이터셋을 불러오기 전에도 항상 수정할 수 있게 유지합니다.
        self.shuffle_toggle_button.setEnabled(True)
        self.train_ratio_edit.setEnabled(True)
        self.val_ratio_edit.setEnabled(True)
        self.test_ratio_edit.setEnabled(True)
        self.dataset_copy_button.setEnabled(enabled)
        self.dataset_move_button.setEnabled(enabled)

    def _parse_ratio_value(self, input_edit: QLineEdit, label_text: str) -> float:
        """분할 비율 입력칸의 값을 실수로 읽습니다."""
        try:
            return float(input_edit.text().strip())
        except ValueError as exc:
            raise ValueError(f"{label_text} 비율은 숫자로 입력해주세요.") from exc

    def _parse_training_integer(self, input_edit: QLineEdit, label_text: str) -> int:
        """학습 옵션 입력칸의 정수 값을 읽고 기본 검증을 수행합니다."""
        try:
            value = int(input_edit.text().strip())
        except ValueError as exc:
            raise ValueError(f"{label_text} 값은 정수여야 합니다.") from exc
        if label_text == "Workers" and value < 0:
            raise ValueError(f"{label_text} 값은 0 이상이어야 합니다.")
        if label_text != "Workers" and value <= 0:
            raise ValueError(f"{label_text} 값은 1 이상이어야 합니다.")
        return value

    def _reset_training_status_ui(self) -> None:
        """학습이 끝난 뒤 상태 표시를 기본값으로 되돌립니다."""
        self.training_process_pid = None
        self.training_resource_timer.stop()
        self.training_status_summary_label.setText(
            "진행률 - | 에포크 - | 학습 CPU - | 학습 메모리 -\nGPU 사용률 - | GPU 메모리 - | 앱 CPU - | 앱 메모리 -"
        )
        self.training_progress_bar.setValue(0)

    def _format_memory_text(self, bytes_value: int) -> str:
        """메모리 바이트 값을 보기 쉬운 MB 또는 GB 문자열로 변환합니다."""
        gb_value = bytes_value / (1024 ** 3)
        if gb_value >= 1.0:
            return f"{gb_value:.1f}GB"
        mb_value = bytes_value / (1024 ** 2)
        return f"{mb_value:.0f}MB"

    def _set_training_status_summary(
        self,
        progress_text: str,
        epoch_text: str,
        train_cpu_text: str,
        train_memory_text: str,
        train_gpu_text: str,
        train_gpu_memory_text: str,
        app_cpu_text: str,
        app_memory_text: str,
    ) -> None:
        """학습 상태 요약 라벨을 관련 항목끼리 2줄로 묶어 갱신합니다."""
        # 진행 정보와 자원 사용량을 줄별로 나눠 보여 주면 긴 상태 문자열도 읽기 쉬워집니다.
        self.training_status_summary_label.setText(
            "\n".join(
                [
                    " | ".join(
                        [
                            f"진행률 {progress_text}",
                            f"에포크 {epoch_text}",
                            f"학습 CPU {train_cpu_text}",
                            f"학습 메모리 {train_memory_text}",
                        ]
                    ),
                    " | ".join(
                        [
                            f"GPU 사용률 {train_gpu_text}",
                            f"GPU 메모리 {train_gpu_memory_text}",
                            f"앱 CPU {app_cpu_text}",
                            f"앱 메모리 {app_memory_text}",
                        ]
                    ),
                ]
            )
        )

    def on_training_process_started(self, process_pid: int) -> None:
        """백그라운드 학습 프로세스 PID를 받아 자원 사용량 갱신을 시작합니다."""
        self.training_process_pid = process_pid
        try:
            psutil.Process(process_pid).cpu_percent(None)
        except Exception:
            pass
        try:
            psutil.Process().cpu_percent(None)
        except Exception:
            pass
        self.training_resource_timer.start()
        self.update_training_resource_summary()

    def update_training_resource_summary(self) -> None:
        """학습 프로세스와 앱의 CPU, GPU, 메모리 사용량을 요약 라벨에 반영합니다."""
        current_progress = getattr(self, "_current_training_progress_text", "-")
        current_epoch = getattr(self, "_current_training_epoch_text", "-")
        train_cpu_text = "-"
        train_memory_text = "-"
        train_gpu_text = "-"
        train_gpu_memory_text = "-"
        app_cpu_text = "-"
        app_memory_text = "-"

        try:
            app_process = psutil.Process()
            app_cpu_text = f"{app_process.cpu_percent(None):.0f}%"
            app_memory_text = self._format_memory_text(app_process.memory_info().rss)
        except Exception:
            pass

        if self.training_process_pid is not None:
            try:
                training_process = psutil.Process(self.training_process_pid)
                train_cpu_text = f"{training_process.cpu_percent(None):.0f}%"
                train_memory_text = self._format_memory_text(training_process.memory_info().rss)
            except Exception:
                train_cpu_text = "-"
                train_memory_text = "-"

        # NVML을 직접 호출해 콘솔 창 없이 GPU 사용률과 메모리 사용량을 조회합니다.
        if self.config.use_gpu and self.hardware_status.gpu_available:
            train_gpu_text, train_gpu_memory_text = query_gpu_runtime_usage()

        self._set_training_status_summary(
            progress_text=current_progress,
            epoch_text=current_epoch,
            train_cpu_text=train_cpu_text,
            train_memory_text=train_memory_text,
            train_gpu_text=train_gpu_text,
            train_gpu_memory_text=train_gpu_memory_text,
            app_cpu_text=app_cpu_text,
            app_memory_text=app_memory_text,
        )

    def _set_training_controls_enabled(self, enabled: bool) -> None:
        """학습 시작 전후에 관련 입력 컨트롤의 활성 상태를 제어합니다."""
        self.training_start_button.setEnabled(enabled)
        self.training_gpu_button.setEnabled(enabled and self.hardware_status.gpu_available)
        self.training_model_select_button.setEnabled(enabled)
        self.yolo_select_button.setEnabled(enabled)
        self.model_number_combo.setEnabled(enabled)
        self.model_size_combo.setEnabled(enabled)
        self.model_download_button.setEnabled(enabled)
        self.project_name_edit.setEnabled(enabled)
        self.project_select_button.setEnabled(enabled)
        self.train_epochs_edit.setEnabled(enabled)
        self.train_batch_edit.setEnabled(enabled)
        self.train_imgsz_edit.setEnabled(enabled)
        self.train_workers_edit.setEnabled(enabled)
        self.train_patience_edit.setEnabled(enabled)

    def refresh_runtime_status(self) -> None:
        """하드웨어 상태를 다시 감지하고 화면에 반영합니다."""
        self.hardware_status = detect_hardware()
        if not self.hardware_status.gpu_available:
            self.config.use_gpu = False
        self._refresh_hardware_labels()
        self._refresh_software_labels()
        save_app_config(self.config)
        self._add_log("하드웨어 환경 새로고침")

    def on_use_gpu_toggled(self) -> None:
        """GPU 사용 토글 상태를 설정에 반영합니다."""
        if not self.hardware_status.gpu_available:
            self.config.use_gpu = False
        else:
            sender = self.sender()
            if sender is self.training_gpu_button:
                self.config.use_gpu = self.training_gpu_button.isChecked()
            else:
                self.config.use_gpu = self.use_gpu_button.isChecked()
        self._refresh_hardware_labels()
        save_app_config(self.config)
        self._add_log(f"학습 장치 변경 {'GPU' if self.config.use_gpu else 'CPU'}")

    def on_training_mode_changed(self) -> None:
        """학습 모드 토글 버튼을 라디오 버튼처럼 상호 배타적으로 유지합니다."""
        sender = self.sender()
        if sender is self.new_training_button and self.new_training_button.isChecked():
            self.resume_training_button.blockSignals(True)
            self.resume_training_button.setChecked(False)
            self.resume_training_button.blockSignals(False)
            self.config.training_mode = "new"
        elif sender is self.resume_training_button and self.resume_training_button.isChecked():
            self.new_training_button.blockSignals(True)
            self.new_training_button.setChecked(False)
            self.new_training_button.blockSignals(False)
            self.config.training_mode = "resume"
        elif not self.new_training_button.isChecked() and not self.resume_training_button.isChecked():
            # 둘 다 꺼진 상태는 허용하지 않고 기본값인 신규 학습으로 되돌립니다.
            self.new_training_button.blockSignals(True)
            self.new_training_button.setChecked(True)
            self.new_training_button.blockSignals(False)
            self.config.training_mode = "new"
        else:
            self.config.training_mode = "resume" if self.resume_training_button.isChecked() else "new"

        self.new_training_button._refresh_marked_text(self.new_training_button.isChecked())
        self.resume_training_button._refresh_marked_text(self.resume_training_button.isChecked())
        save_app_config(self.config)
        self._add_log(f"학습 모드 변경 {self.config.training_mode}")

    def on_model_selection_changed(self) -> None:
        """드롭다운 변경값을 설정에 반영하되 모델을 자동으로 선택하지는 않습니다."""
        size_code = size_label_to_code(self.model_size_combo.currentText())
        update_selected_model(
            self.config,
            self.model_number_combo.currentText(),
            size_code,
            keep_existing_path=True,
        )
        self._refresh_software_labels()
        save_app_config(self.config)
        selected_text = f"버전 {self.model_number_combo.currentText() or '-'} / 크기 {self.model_size_combo.currentText() or '-'}"
        self._add_log(f"YOLO 선택 옵션 변경 {selected_text}")

    def save_training_option_inputs(self) -> None:
        """학습 옵션 입력값을 설정 객체와 파일에 저장합니다."""
        self.config.train_epochs = self.train_epochs_edit.text().strip() or "100"
        self.config.train_batch = self.train_batch_edit.text().strip() or "16"
        self.config.train_imgsz = self.train_imgsz_edit.text().strip() or "640"
        self.config.train_workers = self.train_workers_edit.text().strip() or "0"
        self.config.train_patience = self.train_patience_edit.text().strip() or "50"
        save_app_config(self.config)
        self._refresh_training_info_panel()
        self._add_log(
            "학습 옵션 저장 "
            f"Epoch={self.config.train_epochs}, Batch={self.config.train_batch}, "
            f"Image Size={self.config.train_imgsz}, Workers={self.config.train_workers}, "
            f"Patience={self.config.train_patience}"
        )

    def on_project_name_edit_finished(self) -> None:
        """프로젝트명 입력 완료 후 중앙 학습 정보와 로그를 현재 값으로 맞춥니다."""
        project_name = self.project_name_edit.text().strip() or "-"
        self._refresh_training_info_panel()
        self._add_log(f"프로젝트명 변경 {project_name}")

    def start_training(self) -> None:
        """현재 설정값과 프로젝트 폴더를 사용해 백그라운드 학습을 시작합니다."""
        if self.training_thread is not None:
            QMessageBox.information(self, "학습 진행 중", "현재 학습이 이미 진행 중입니다.")
            return
        if self.config.paths is None:
            QMessageBox.warning(self, "경로 오류", "프로젝트 경로 정보를 찾을 수 없습니다.")
            return
        if not self.hardware_status.python_command:
            QMessageBox.warning(self, "Python 오류", "학습을 실행할 Python 명령을 찾지 못했습니다.")
            return
        if not self.config.model_path:
            QMessageBox.warning(self, "모델 미선택", "학습에 사용할 YOLO 모델을 먼저 선택해주세요.")
            return

        project_name = self.project_name_edit.text().strip()
        if not project_name:
            QMessageBox.warning(self, "프로젝트명 오류", "학습 프로젝트명을 입력해주세요.")
            return

        project_dir = self.config.paths.work_dir / project_name
        data_yaml_path = project_dir / "data.yaml"
        result_dir = project_dir / "result"
        # 학습 시에는 항상 절대 경로를 사용해 실행 위치에 따른 자동 다운로드를 막습니다.
        model_path = Path(self.config.model_path).resolve()
        runner_script_path = self.config.paths.code_dir / "training_runner.py"

        if not project_dir.exists():
            QMessageBox.warning(self, "프로젝트 없음", "학습 프로젝트 폴더가 없습니다. 먼저 데이터셋을 준비해주세요.")
            return
        if not data_yaml_path.exists():
            QMessageBox.warning(self, "data.yaml 없음", "프로젝트 폴더에 data.yaml이 없습니다. 데이터셋 복사 또는 이동을 먼저 실행해주세요.")
            return
        if not model_path.exists():
            QMessageBox.warning(self, "모델 경로 오류", "선택한 YOLO 모델 파일을 찾을 수 없습니다.")
            return
        if model_path.suffix.lower() != ".pt":
            QMessageBox.warning(self, "모델 경로 오류", "학습 모델은 .pt 파일만 사용할 수 있습니다.")
            return
        if not runner_script_path.exists():
            QMessageBox.warning(self, "러너 스크립트 오류", "학습 러너 스크립트를 찾을 수 없습니다.")
            return

        try:
            epochs = self._parse_training_integer(self.train_epochs_edit, "Epoch")
            batch_size = self._parse_training_integer(self.train_batch_edit, "Batch")
            image_size = self._parse_training_integer(self.train_imgsz_edit, "Image Size")
            workers = self._parse_training_integer(self.train_workers_edit, "Workers")
            patience = self._parse_training_integer(self.train_patience_edit, "Patience")
        except ValueError as exc:
            QMessageBox.warning(self, "학습 옵션 오류", str(exc))
            return

        if os.name == "nt" and workers > 0:
            # Windows GUI 실행 파일에서는 DataLoader worker가 콘솔 창을 반복 생성할 수 있어 0으로 고정합니다.
            workers = 0
            self.train_workers_edit.setText("0")
            self._add_log("Windows 안정성을 위해 Workers를 0으로 조정")

        # 학습 직전 값을 저장해 러너가 최신 옵션으로 실행되게 합니다.
        self.save_training_option_inputs()
        result_dir.mkdir(parents=True, exist_ok=True)

        request = TrainingRequest(
            python_command=self.hardware_status.python_command,
            runner_script_path=runner_script_path,
            model_path=model_path,
            data_yaml_path=data_yaml_path,
            result_dir=result_dir,
            ultralytics_dir=self.config.paths.ultralytics_dir,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            workers=workers,
            patience=patience,
            use_gpu=self.config.use_gpu and self.hardware_status.gpu_available,
        )

        self.training_worker = TrainingWorker(request)
        self.training_thread = QThread(self)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.process_started.connect(self.on_training_process_started)
        self.training_worker.progress_changed.connect(self.on_training_progress_changed)
        # 워커가 보내는 상태 문자열은 그대로 중앙 로그에 누적합니다.
        self.training_worker.status_changed.connect(self.on_training_status_changed)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.failed.connect(self.on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self._cleanup_training_thread)

        self._set_training_controls_enabled(False)
        self._current_training_progress_text = "0%"
        self._current_training_epoch_text = f"(0/{epochs})"
        self._set_training_status_summary(
            progress_text=self._current_training_progress_text,
            epoch_text=self._current_training_epoch_text,
            train_cpu_text="-",
            train_memory_text="-",
            train_gpu_text="-",
            train_gpu_memory_text="-",
            app_cpu_text="-",
            app_memory_text="-",
        )
        self.training_progress_bar.setValue(0)
        self.training_history_widget.clear_history()
        self.training_history_widget.append_progress(0)
        self._add_log(f"학습 시작 프로젝트={project_name} | 결과={result_dir}")
        self.training_thread.start()

    def on_training_status_changed(self, _status_text: str) -> None:
        """워커가 보낸 상태 문자열을 중앙 로그에 추가합니다."""
        # 현재 화면에는 별도 상태 텍스트 칸이 없으므로 로그만 남깁니다.
        if _status_text:
            self._add_log(_status_text)

    def on_training_progress_changed(self, current_epoch: int, total_epochs: int) -> None:
        """백그라운드 워커가 보낸 에포크 진행률을 상태 UI에 반영합니다."""
        safe_total = max(total_epochs, 1)
        progress_percent = int((max(current_epoch, 0) / safe_total) * 100)
        self._current_training_progress_text = f"{progress_percent}%"
        self._current_training_epoch_text = f"({current_epoch}/{total_epochs})"
        self.training_progress_bar.setValue(progress_percent)
        self.training_history_widget.append_progress(progress_percent)
        self._add_log(f"학습 진행 {self._current_training_epoch_text} | 진행률 {self._current_training_progress_text}")
        self.update_training_resource_summary()

    def on_training_finished(self, result_dir_text: str) -> None:
        """학습이 정상 종료되면 상태 UI를 갱신하고 결과 경로를 안내합니다."""
        self._set_training_controls_enabled(True)
        total_epochs = self.train_epochs_edit.text().strip() or "-"
        self._current_training_progress_text = "100%"
        self._current_training_epoch_text = f"({total_epochs}/{total_epochs})"
        self.training_progress_bar.setValue(100)
        self.update_training_resource_summary()
        self.training_resource_timer.stop()
        self._add_log(f"학습 완료 | 결과 폴더 {result_dir_text}")
        QMessageBox.information(
            self,
            "학습 완료",
            f"학습이 완료되었습니다.\n결과 폴더\n{result_dir_text}",
        )

    def on_training_failed(self, error_text: str) -> None:
        """학습 실패 시 상태를 초기화하고 오류 메시지를 보여줍니다."""
        self._set_training_controls_enabled(True)
        self._reset_training_status_ui()
        self._add_log(f"학습 실패 | {error_text}")
        QMessageBox.warning(self, "학습 실패", error_text)

    def _cleanup_training_thread(self) -> None:
        """학습 스레드가 종료되면 워커와 스레드 참조를 정리합니다."""
        self.training_process_pid = None
        if self.training_worker is not None:
            self.training_worker.deleteLater()
            self.training_worker = None
        if self.training_thread is not None:
            self.training_thread.deleteLater()
            self.training_thread = None

    def load_dataset_folder(self) -> None:
        """원본 데이터셋 폴더를 선택하고 classes.txt와 이미지/라벨 쌍을 검사합니다."""
        start_dir = str((self.config.paths.root_dir.parent / "AutoLabeler" / "Work") if self.config.paths else Path.cwd())
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "원본 데이터셋 폴더 선택",
            start_dir,
        )
        if not selected_dir:
            return

        dataset_dir = Path(selected_dir)
        try:
            _classes_path, pairs = scan_dataset_pairs(dataset_dir)
        except Exception as exc:
            QMessageBox.warning(self, "데이터셋 불러오기 실패", str(exc))
            return

        self.selected_dataset_dir = dataset_dir
        self.dataset_path_edit.setText(str(dataset_dir))
        self._update_dataset_controls_enabled()
        self._refresh_training_info_panel()
        self._add_log(f"데이터셋 불러오기 {dataset_dir} | 유효 쌍 {len(pairs)}")
        QMessageBox.information(
            self,
            "데이터셋 확인 완료",
            f"유효한 이미지/라벨 쌍 {len(pairs)}개를 확인했습니다.",
        )

    def process_dataset_distribution(self, move_files: bool) -> None:
        """선택한 데이터셋을 현재 워크스페이스 프로젝트 폴더로 복사하거나 이동합니다."""
        if self.selected_dataset_dir is None:
            QMessageBox.warning(self, "데이터셋 미선택", "먼저 데이터셋 폴더를 불러와주세요.")
            return
        if self.config.paths is None:
            QMessageBox.warning(self, "경로 오류", "워크스페이스 경로 정보를 찾을 수 없습니다.")
            return

        try:
            train_ratio = self._parse_ratio_value(self.train_ratio_edit, "Train")
            val_ratio = self._parse_ratio_value(self.val_ratio_edit, "Val")
            test_ratio = self._parse_ratio_value(self.test_ratio_edit, "Test")
            result = build_training_project(
                dataset_dir=self.selected_dataset_dir,
                workspace_dir=self.config.paths.work_dir,
                project_name=self.project_name_edit.text(),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                shuffle_enabled=self.shuffle_toggle_button.isChecked(),
                move_files=move_files,
            )
        except Exception as exc:
            QMessageBox.warning(self, "데이터셋 처리 실패", str(exc))
            return

        action_text = "이동" if move_files else "복사"
        self._refresh_training_info_panel()
        self._add_log(
            f"데이터셋 {action_text} 완료 | Train {result.train_count} | Val {result.val_count} | Test {result.test_count}"
        )
        QMessageBox.information(
            self,
            f"데이터셋 {action_text} 완료",
            "\n".join(
                [
                    f"프로젝트 폴더: {result.project_dir}",
                    f"Train: {result.train_count}개",
                    f"Val: {result.val_count}개",
                    f"Test: {result.test_count}개",
                ]
            ),
        )

        # 이동 처리 이후에는 원본 폴더의 파일 상태가 바뀌므로 다시 불러오도록 초기화합니다.
        if move_files:
            self.selected_dataset_dir = None
            self.dataset_path_edit.clear()
            self._update_dataset_controls_enabled()
            self._refresh_training_info_panel()

    def select_export_model_path(self) -> None:
        """변환과 이미지 디텍팅에 사용할 pt 모델 파일을 선택합니다."""
        start_dir = str(self.config.paths.work_dir if self.config.paths else Path.cwd())
        selected_path, _filter = QFileDialog.getOpenFileName(
            self,
            "변환 및 디텍팅용 YOLO .pt 파일 선택",
            start_dir,
            "PyTorch Weights (*.pt)",
        )
        if not selected_path:
            return

        self.export_model_path = Path(selected_path).resolve()
        self.export_model_path_edit.setText(str(self.export_model_path))
        self._add_log(f"우측 모델 선택 {self.export_model_path.name}")

    def export_selected_model(self) -> None:
        """선택한 pt 모델을 지정한 형식으로 export합니다."""
        if self.export_model_path is None or not self.export_model_path.exists():
            QMessageBox.warning(self, "모델 미선택", "먼저 변환할 pt 모델 파일을 선택해주세요.")
            return
        if self.config.paths is None:
            QMessageBox.warning(self, "경로 오류", "프로젝트 경로 정보를 찾을 수 없습니다.")
            return
        if not self.hardware_status.python_command:
            QMessageBox.warning(self, "Python 오류", "모델 변환을 실행할 Python 명령을 찾지 못했습니다.")
            return

        export_format = self.export_format_combo.currentText()
        try:
            self.export_model_button.setEnabled(False)
            self._add_log(f"모델 Export 시작 {self.export_model_path.name} -> {export_format}")
            payload = self._run_model_tool(
                [
                    "export",
                    "--model",
                    str(self.export_model_path),
                    "--format",
                    export_format,
                    "--output-dir",
                    str(self.config.paths.work_dir),
                ]
            )
        except Exception as exc:
            QMessageBox.warning(self, "Export 실패", f"모델 변환 중 문제가 발생했습니다.\n{exc}")
            self._add_log(f"모델 Export 실패 | {exc}")
            return
        finally:
            self.export_model_button.setEnabled(True)

        export_result = payload.get("output", "")
        self._add_log(f"모델 Export 완료 | {export_result}")
        QMessageBox.information(self, "Export 완료", f"모델 변환이 완료되었습니다.\n{export_result}")

    def select_detection_image(self) -> None:
        """디텍팅에 사용할 이미지 파일을 선택하고 뷰어에 표시합니다."""
        start_dir = str(self.config.paths.work_dir if self.config.paths else Path.cwd())
        selected_path, _filter = QFileDialog.getOpenFileName(
            self,
            "디텍팅할 이미지 선택",
            start_dir,
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not selected_path:
            return

        self.detect_image_path = Path(selected_path).resolve()
        self.detect_image_path_edit.setText(str(self.detect_image_path))
        self._display_pixmap(QPixmap(str(self.detect_image_path)))
        self.detection_result_label.setText("검출 대기 중")
        self._add_log(f"디텍팅 이미지 선택 {self.detect_image_path.name}")

    def run_detection_on_image(self) -> None:
        """선택한 pt 모델로 이미지를 디텍팅하고 결과를 화면에 표시합니다."""
        if self.export_model_path is None or not self.export_model_path.exists():
            QMessageBox.warning(self, "모델 미선택", "먼저 디텍팅에 사용할 pt 모델 파일을 선택해주세요.")
            return
        if self.detect_image_path is None or not self.detect_image_path.exists():
            QMessageBox.warning(self, "이미지 미선택", "먼저 디텍팅할 이미지 파일을 선택해주세요.")
            return
        if self.config.paths is None:
            QMessageBox.warning(self, "경로 오류", "프로젝트 경로 정보를 찾을 수 없습니다.")
            return
        if not self.hardware_status.python_command:
            QMessageBox.warning(self, "Python 오류", "이미지 디텍팅을 실행할 Python 명령을 찾지 못했습니다.")
            return

        try:
            self.detect_run_button.setEnabled(False)
            self._add_log(f"이미지 디텍팅 시작 모델={self.export_model_path.name} | 이미지={self.detect_image_path.name}")
            output_path = Path(tempfile.gettempdir()) / "AutoYoloTrainer" / "detect_preview.jpg"
            payload = self._run_model_tool(
                [
                    "detect",
                    "--model",
                    str(self.export_model_path),
                    "--image",
                    str(self.detect_image_path),
                    "--output",
                    str(output_path),
                ]
            )
        except Exception as exc:
            QMessageBox.warning(self, "디텍팅 실패", f"이미지 디텍팅 중 문제가 발생했습니다.\n{exc}")
            self._add_log(f"이미지 디텍팅 실패 | {exc}")
            return
        finally:
            self.detect_run_button.setEnabled(True)

        result_image_path = Path(str(payload.get("output", "")))
        if result_image_path.exists():
            self._display_pixmap(QPixmap(str(result_image_path)))
        self._update_detection_result_label(payload.get("detections", []))
        self._add_log("이미지 디텍팅 완료")

    def _run_model_tool(self, args: list[str]) -> dict[str, object]:
        """별도 Python 러너로 모델 변환/디텍팅을 실행하고 JSON 결과를 받습니다."""
        if self.config.paths is None:
            raise RuntimeError("프로젝트 경로 정보를 찾을 수 없습니다.")

        runner_path = self.config.paths.code_dir / "model_tool_runner.py"
        command = [*self.hardware_status.python_command, str(runner_path), *args]
        process_env = build_clean_python_env()

        startup_info = None
        creation_flags = 0
        if os.name == "nt":
            # 러너 실행 중 콘솔 창이 나타나지 않도록 Windows 생성 플래그를 지정합니다.
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = 0
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        with clean_windows_dll_search_path():
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=process_env,
                startupinfo=startup_info,
                creationflags=creation_flags,
                cwd=str(self.config.paths.root_dir),
            )
        stdout_text = completed.stdout.strip()
        stderr_text = completed.stderr.strip()
        try:
            payload = json.loads(stdout_text.splitlines()[-1] if stdout_text else "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(stderr_text or stdout_text or "러너 결과를 해석하지 못했습니다.") from exc

        if completed.returncode != 0 or not payload.get("ok", False):
            raise RuntimeError(str(payload.get("error") or stderr_text or "러너 실행에 실패했습니다."))
        return payload

    def _display_pixmap(self, pixmap: QPixmap) -> None:
        """QPixmap을 우측 이미지 뷰어 크기에 맞춰 표시합니다."""
        if pixmap.isNull():
            self.preview_pixmap = None
            self.detect_preview_label.setText("이미지를 표시할 수 없습니다.")
            self.detect_preview_label.setPixmap(QPixmap())
            return

        self.preview_pixmap = pixmap
        self.detect_preview_label.setText("")
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self) -> None:
        """원본 비율을 유지하며 이미지 뷰어 영역에 맞게 확대 또는 축소합니다."""
        if self.preview_pixmap is None or self.preview_pixmap.isNull():
            return

        target_size = self.detect_preview_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        scaled_pixmap = self.preview_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.detect_preview_label.setPixmap(scaled_pixmap)

    def _update_detection_result_label(self, detections: list[dict[str, object]]) -> None:
        """검출된 객체의 클래스와 신뢰도를 요약 라벨에 출력합니다."""
        if not detections:
            self.detection_result_label.setText("검출 결과 없음")
            return

        lines = [f"검출 객체 {len(detections)}개"]
        for index, detection in enumerate(detections, start=1):
            class_name = str(detection.get("class_name", "-"))
            confidence = float(detection.get("confidence", 0.0))
            lines.append(f"{index}. {class_name} | 정확도 {confidence:.2%}")
        self.detection_result_label.setText("\n".join(lines))

    def select_training_project_folder(self) -> None:
        """워크스페이스 안의 기존 학습 프로젝트 폴더를 선택해 프로젝트명에 반영합니다."""
        if self.config.paths is None:
            QMessageBox.warning(self, "경로 오류", "워크스페이스 경로 정보를 찾을 수 없습니다.")
            return

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "학습 프로젝트 폴더 선택",
            str(self.config.paths.work_dir),
        )
        if not selected_dir:
            return

        project_dir = Path(selected_dir).resolve()
        work_dir = self.config.paths.work_dir.resolve()
        try:
            project_dir.relative_to(work_dir)
        except ValueError:
            QMessageBox.warning(self, "프로젝트 경로 오류", "워크스페이스 안의 학습 프로젝트 폴더를 선택해주세요.")
            return

        self.project_name_edit.setText(project_dir.name)
        self._refresh_training_info_panel()
        self._add_log(f"학습 프로젝트 선택 {project_dir.name}")

    def select_model_path(self) -> None:
        """Data\\models를 시작 경로로 사용해 모델 가중치 파일을 직접 선택합니다."""
        start_dir = str(self.config.paths.model_dir if self.config.paths else Path.cwd())
        selected_path, _filter = QFileDialog.getOpenFileName(
            self,
            "학습용 YOLO .pt 파일 선택",
            start_dir,
            "PyTorch Weights (*.pt)",
        )
        if not selected_path:
            return

        selected_file = Path(selected_path).resolve()
        model_name, model_number, size_code = parse_model_filename(selected_file)
        self.config.model_path = str(selected_file)
        self.config.selected_model_name = model_name
        self.config.selected_model_number = model_number
        self.config.selected_model_size = size_code
        # 수동 선택한 파일명과 드롭다운 표시를 맞춰 다음 실행 때 다른 모델로 덮이지 않게 합니다.
        self.model_number_combo.blockSignals(True)
        self.model_size_combo.blockSignals(True)
        self.model_number_combo.setCurrentText(model_number)
        self.model_size_combo.setCurrentText(size_code_to_label(size_code))
        self.model_number_combo.blockSignals(False)
        self.model_size_combo.blockSignals(False)
        self._refresh_software_labels()
        save_app_config(self.config)
        self._add_log(f"YOLO 모델 선택 {selected_file.name}")

    def download_model(self) -> None:
        """선택한 기본 모델을 Data\\models 폴더로 다운로드합니다."""
        try:
            downloaded_path = download_selected_model(self.config)
            self._refresh_software_labels()
            save_app_config(self.config)
            self._add_log(f"YOLO 모델 다운로드 {downloaded_path.name}")
            QMessageBox.information(
                self,
                "다운로드 완료",
                f"모델 다운로드 경로\n{downloaded_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "다운로드 실패",
                f"모델 다운로드 중 문제가 발생했습니다.\n{exc}",
            )

    def open_settings(self) -> None:
        """프로그램 설정 창을 열고 적용 결과를 반영합니다."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            dialog.apply_to_config(self.config)
            self._apply_theme()
            save_app_config(self.config)
            self._add_log(f"프로그램 설정 적용 | 테마 {self.config.theme_mode}")

    def closeEvent(self, event) -> None:  # noqa: N802
        """창이 닫힐 때 마지막 설정 상태를 저장합니다."""
        save_app_config(self.config)
        super().closeEvent(event)


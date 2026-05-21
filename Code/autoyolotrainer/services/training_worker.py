# -*- coding: utf-8 -*-
"""AutoYoloTrainer의 백그라운드 학습 워커를 정의합니다."""

from __future__ import annotations

import locale
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .process_service import build_clean_python_env, clean_windows_dll_search_path


@dataclass
class TrainingRequest:
    """백그라운드 학습 실행에 필요한 요청 정보를 담습니다."""

    python_command: list[str]
    runner_script_path: Path
    model_path: Path
    data_yaml_path: Path
    result_dir: Path
    ultralytics_dir: Path
    epochs: int
    image_size: int
    batch_size: int
    workers: int
    patience: int
    use_gpu: bool


class TrainingWorker(QObject):
    """별도 Python 프로세스로 YOLO 학습을 수행하는 워커입니다."""

    process_started = pyqtSignal(int)
    progress_changed = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, request: TrainingRequest) -> None:
        super().__init__()
        self.request = request

    def _decode_output_line(self, raw_line: bytes) -> str:
        """학습 로그를 가능한 인코딩으로 복구해 문자열로 변환합니다."""
        for encoding in ("utf-8", locale.getpreferredencoding(False), "cp949"):
            try:
                return raw_line.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_line.decode("utf-8", errors="replace").strip()

    def run(self) -> None:
        """러너 스크립트를 실행하고 에포크 진행률을 메인 스레드로 전달합니다."""
        try:
            command = [
                *self.request.python_command,
                str(self.request.runner_script_path),
                "--model",
                str(self.request.model_path),
                "--data",
                str(self.request.data_yaml_path),
                "--project-dir",
                str(self.request.result_dir),
                "--epochs",
                str(self.request.epochs),
                "--imgsz",
                str(self.request.image_size),
                "--batch",
                str(self.request.batch_size),
                "--workers",
                str(self.request.workers),
                "--patience",
                str(self.request.patience),
                "--device",
                "0" if self.request.use_gpu else "cpu",
                "--ultralytics-dir",
                str(self.request.ultralytics_dir),
            ]

            # 시작 직후에도 UI가 비어 보이지 않도록 0/전체 에포크를 먼저 전달합니다.
            self.progress_changed.emit(0, self.request.epochs)
            self.status_changed.emit("학습 프로세스 시작")

            process_env = build_clean_python_env()
            process_env["YOLO_CONFIG_DIR"] = str(self.request.ultralytics_dir)
            process_env["MPLCONFIGDIR"] = str(self.request.ultralytics_dir)
            self.request.result_dir.mkdir(parents=True, exist_ok=True)

            startup_info = None
            creation_flags = 0
            if os.name == "nt":
                # GUI 프로그램에서 학습용 Python 프로세스를 띄울 때 콘솔창이 나타나지 않게 숨깁니다.
                startup_info = subprocess.STARTUPINFO()
                startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup_info.wShowWindow = 0
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            with clean_windows_dll_search_path():
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=False,
                    env=process_env,
                    cwd=str(self.request.result_dir),
                    startupinfo=startup_info,
                    creationflags=creation_flags,
                )
            # 메인 UI가 학습 프로세스의 자원 사용량을 추적할 수 있도록 PID를 전달합니다.
            self.process_started.emit(process.pid)

            if process.stdout is None:
                raise RuntimeError("학습 로그 스트림을 열지 못했습니다.")

            recent_lines: list[str] = []
            for raw_line in process.stdout:
                line = self._decode_output_line(raw_line)
                if not line:
                    continue
                recent_lines.append(line)
                if len(recent_lines) > 20:
                    recent_lines.pop(0)

                if line.startswith("AUTO_YOLO_TRAINER_PROGRESS|"):
                    try:
                        _tag, current_text, total_text = line.split("|", 2)
                        self.progress_changed.emit(int(current_text), int(total_text))
                    except ValueError:
                        self.status_changed.emit(line)
                else:
                    self.status_changed.emit(line)

            return_code = process.wait()
            if return_code != 0:
                error_tail = "\n".join(recent_lines[-10:])
                raise RuntimeError(
                    f"학습 프로세스가 비정상 종료되었습니다. code={return_code}\n{error_tail}"
                )

            self.finished.emit(str(self.request.result_dir))
        except Exception as exc:
            self.failed.emit(str(exc))

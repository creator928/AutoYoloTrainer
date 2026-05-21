# -*- coding: utf-8 -*-
"""학습 장치 선택에 필요한 하드웨어 상태를 판별합니다."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from ..models import HardwareStatus
from .process_service import build_clean_python_env, clean_windows_dll_search_path


def _hidden_subprocess_kwargs() -> dict[str, object]:
    """Windows GUI 앱에서 점검용 콘솔 창이 뜨지 않도록 subprocess 옵션을 반환합니다."""
    if os.name != "nt":
        return {}
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup_info.wShowWindow = 0
    return {
        "startupinfo": startup_info,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def _load_nvml():
    """pynvml이 설치된 경우에만 NVML 모듈을 반환합니다."""
    try:
        import pynvml  # type: ignore

        return pynvml
    except Exception:
        return None


def resolve_python_command() -> list[str]:
    """외부 학습 실행에 사용할 Python 명령을 우선순위대로 결정합니다."""
    candidates: list[list[str]] = []
    configured_python = os.environ.get("AUTOYOLO_PYTHON_EXE", "").strip()
    if configured_python and Path(configured_python).exists():
        candidates.append([configured_python])
    if shutil.which("python"):
        candidates.append(["python"])
    if shutil.which("py"):
        candidates.extend((["py", "-3.10"], ["py", "-3"]))
    if Path(sys.executable).name.lower().startswith("python"):
        candidates.append([sys.executable])

    probe_script = (
        "import socket\n"
        "import importlib.metadata\n"
        "import torch\n"
        "import ultralytics\n"
    )
    for candidate in candidates:
        try:
            with clean_windows_dll_search_path():
                completed = subprocess.run(
                    [*candidate, "-c", probe_script],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=build_clean_python_env(),
                    **_hidden_subprocess_kwargs(),
                )
            if completed.returncode == 0:
                return candidate
        except Exception:
            continue
    return []


def _storage_summaries() -> list[str]:
    """로컬 저장소 파티션별 총 용량과 사용량 요약 문자열을 반환합니다."""
    summaries: list[str] = []
    seen_devices: set[str] = set()
    for partition in psutil.disk_partitions(all=False):
        device = partition.device or partition.mountpoint
        if not device or device in seen_devices:
            continue
        seen_devices.add(device)
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except Exception:
            continue
        total_gb = round(usage.total / (1024 ** 3), 1)
        used_gb = round(usage.used / (1024 ** 3), 1)
        # 저장소 표기는 [드라이브 사용량/전체용량] 형식으로 고정합니다.
        summaries.append(f"[{device} {used_gb}GB/{total_gb}GB]")
    return summaries


def _query_gpu_static_info() -> tuple[bool, str, float]:
    """NVML로 GPU 이름과 전체 메모리를 조회합니다."""
    pynvml = _load_nvml()
    if pynvml is None:
        return False, "", 0.0

    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() <= 0:
            return False, "", 0.0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        gpu_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_gb = round(float(memory_info.total) / (1024 ** 3), 1)
        return True, gpu_name, gpu_memory_gb
    except Exception:
        return False, "", 0.0


def detect_hardware() -> HardwareStatus:
    """CPU, GPU, 메모리, 저장소, CUDA 런타임 상태를 한 번에 수집합니다."""
    python_command = resolve_python_command()
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "CPU"
    cpu_physical_cores = psutil.cpu_count(logical=False) or 0
    cpu_logical_cores = psutil.cpu_count(logical=True) or 0
    os_name = f"{platform.system()} {platform.release()}"
    memory_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    gpu_available, gpu_name, gpu_memory_gb = _query_gpu_static_info()

    if not python_command:
        return HardwareStatus(
            python_command=[],
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cpu_name=cpu_name,
            cpu_physical_cores=cpu_physical_cores,
            cpu_logical_cores=cpu_logical_cores,
            memory_gb=memory_gb,
            os_name=os_name,
            cuda_runtime_available=False,
            storage_summaries=_storage_summaries(),
        )

    script = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " gpu=bool(torch.cuda.is_available())\n"
        " gpu_name=torch.cuda.get_device_name(0) if gpu else ''\n"
        " print(json.dumps({'gpu': gpu, 'gpu_name': gpu_name}, ensure_ascii=False))\n"
        "except Exception:\n"
        " print(json.dumps({'gpu': False, 'gpu_name': ''}, ensure_ascii=False))\n"
    )
    try:
        with clean_windows_dll_search_path():
            completed = subprocess.run(
                [*python_command, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                env=build_clean_python_env(),
                **_hidden_subprocess_kwargs(),
            )
        payload = json.loads(completed.stdout.strip())
        cuda_runtime_available = bool(payload.get("gpu", False))
        if not gpu_available and cuda_runtime_available:
            gpu_available = True
        if not gpu_name:
            gpu_name = str(payload.get("gpu_name", ""))
    except Exception:
        cuda_runtime_available = False

    return HardwareStatus(
        python_command=python_command,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cpu_name=cpu_name,
        cpu_physical_cores=cpu_physical_cores,
        cpu_logical_cores=cpu_logical_cores,
        memory_gb=memory_gb,
        os_name=os_name,
        cuda_runtime_available=cuda_runtime_available,
        storage_summaries=_storage_summaries(),
    )


def query_gpu_runtime_usage() -> tuple[str, str]:
    """NVML로 현재 GPU 사용률과 GPU 메모리 사용량을 조회합니다."""
    pynvml = _load_nvml()
    if pynvml is None:
        return "-", "-"

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = float(memory_info.used) / (1024 ** 3)
        total_gb = float(memory_info.total) / (1024 ** 3)
        return f"{float(utilization.gpu):.0f}%", f"{used_gb:.1f}GB/{total_gb:.1f}GB"
    except Exception:
        return "-", "-"

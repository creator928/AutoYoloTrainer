# -*- coding: utf-8 -*-
"""AutoYoloTrainer 전용 YOLO 학습 러너입니다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--patience", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ultralytics-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Ultralytics YOLO 학습을 실행하고 진행률을 표준 출력으로 전달합니다."""
    args = parse_args()
    os.environ["YOLO_CONFIG_DIR"] = args.ultralytics_dir
    os.environ["MPLCONFIGDIR"] = args.ultralytics_dir

    from ultralytics import YOLO

    project_dir = Path(args.project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"선택한 모델 파일을 찾을 수 없습니다: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError("학습 모델은 .pt 파일만 사용할 수 있습니다.")

    # 상대 경로 부작용으로 exe 옆에 기본 모델이 생기지 않도록 실행 위치를 결과 폴더로 고정합니다.
    os.chdir(project_dir)
    model = YOLO(str(model_path))

    def on_train_epoch_end(trainer) -> None:
        """에포크 종료 시 현재 진행률을 메인 프로세스로 전달합니다."""
        current_epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", args.epochs))
        print(f"AUTO_YOLO_TRAINER_PROGRESS|{current_epoch}|{total_epochs}", flush=True)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 모든 학습 결과를 프로젝트 내부 result 폴더에 누적합니다.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        amp=False,
        project=str(project_dir.parent),
        name=project_dir.name,
        exist_ok=True,
    )
    # AMP 검사 등 외부 요인으로 yolo11n.pt가 생기면 사용자가 만든 결과와 분리해 제거합니다.
    stray_amp_model_path = project_dir / "yolo11n.pt"
    if stray_amp_model_path.exists() and stray_amp_model_path.resolve() != model_path:
        stray_amp_model_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""AutoYoloTrainer 우측 모델 변환/이미지 디텍팅 러너입니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--format", required=True)
    export_parser.add_argument("--output-dir", required=True)

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("--model", required=True)
    detect_parser.add_argument("--image", required=True)
    detect_parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_model_path(model_path: Path) -> Path:
    """사용자가 선택한 .pt 모델 파일이 실제로 존재하는지 검증합니다."""
    resolved_model_path = model_path.resolve()
    if resolved_model_path.suffix.lower() != ".pt":
        raise ValueError("모델 파일은 .pt 파일만 사용할 수 있습니다.")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"선택한 모델 파일을 찾을 수 없습니다: {resolved_model_path}")
    return resolved_model_path


def run_export(model_path: Path, export_format: str, output_dir: Path) -> dict[str, object]:
    """선택한 pt 모델을 지정 형식으로 변환합니다."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path = model.export(format=export_format, project=str(output_dir.resolve()))
    return {"ok": True, "output": str(exported_path)}


def run_detect(model_path: Path, image_path: Path, output_path: Path) -> dict[str, object]:
    """선택한 pt 모델로 이미지를 디텍팅하고 결과 이미지를 저장합니다."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path.resolve()), verbose=False)
    if not results:
        return {"ok": True, "output": "", "detections": []}

    result = results[0]
    plotted_image = result.plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), plotted_image)

    names = getattr(result, "names", {}) or {}
    detections: list[dict[str, object]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            class_index = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append(
                {
                    "class_index": class_index,
                    "class_name": names.get(class_index, str(class_index)),
                    "confidence": confidence,
                }
            )
    return {"ok": True, "output": str(output_path), "detections": detections}


def main() -> int:
    """요청된 모델 작업을 실행하고 JSON 결과를 출력합니다."""
    args = parse_args()
    try:
        if args.command == "export":
            model_path = validate_model_path(Path(args.model))
            payload = run_export(model_path, args.format, Path(args.output_dir))
        else:
            model_path = validate_model_path(Path(args.model))
            payload = run_detect(model_path, Path(args.image), Path(args.output))
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}

    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

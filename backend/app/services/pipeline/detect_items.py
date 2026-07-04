from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_YOLO_MODEL = "yolov8n.pt"


def detect_items(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    confidence_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Detect item candidates in an OOTD image with Ultralytics YOLO.

    The default model is a general COCO YOLO model. It will not be fashion-perfect,
    but it gives us real candidate crops. Later, `model_path` can point to a
    fashion-trained YOLO weights file without changing the rest of the pipeline.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `pip install -r requirements.txt` "
            "from the backend folder, then try extraction again."
        ) from exc

    image_path = Path(image_path).expanduser().resolve()
    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    if not results:
        return []

    result = results[0]
    names = result.names
    detections: list[dict[str, Any]] = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]

        detections.append(
            {
                "yolo_label": names.get(class_id, str(class_id)),
                "yolo_class_id": class_id,
                "yolo_confidence": round(confidence, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
            }
        )

    return detections

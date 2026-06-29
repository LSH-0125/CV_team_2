"""감지 래퍼.

- 사람: YOLOv8s (COCO 전용 모델 — person 감지 정확도 우선)
- 짐:   YOLO-World (텍스트 어휘로 카페 아이템 감지)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    xyxy:       np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    cls_name:   str = ""


@dataclass
class DetectionResult:
    person_boxes:  list[Box]
    luggage_boxes: list[Box]


# ── YOLO-World 짐 어휘 ────────────────────────────────────────────────────────
LUGGAGE_CLASSES: list[str] = [
    "cup", "coffee cup", "tumbler", "bottle",
    "laptop", "laptop computer", "smartphone",
    "backpack", "bag", "handbag",
    "book", "notebook", "textbook",
    "earphones", "headphones", "umbrella",
]

# 프론트 표시용
_BELONGING_MAP: dict[str, dict] = {
    "cup":             {"type": "CUP",      "label": "컵"},
    "coffee cup":      {"type": "CUP",      "label": "커피컵"},
    "tumbler":         {"type": "CUP",      "label": "텀블러"},
    "bottle":          {"type": "CUP",      "label": "음료"},
    "laptop":          {"type": "LAPTOP",   "label": "노트북"},
    "laptop computer": {"type": "LAPTOP",   "label": "노트북"},
    "smartphone":      {"type": "UNKNOWN",  "label": "스마트폰"},
    "backpack":        {"type": "BACKPACK", "label": "백팩"},
    "bag":             {"type": "BACKPACK", "label": "가방"},
    "handbag":         {"type": "HANDBAG",  "label": "핸드백"},
    "book":            {"type": "UNKNOWN",  "label": "책"},
    "notebook":        {"type": "UNKNOWN",  "label": "노트"},
    "textbook":        {"type": "UNKNOWN",  "label": "교재"},
    "earphones":       {"type": "UNKNOWN",  "label": "이어폰"},
    "headphones":      {"type": "UNKNOWN",  "label": "헤드폰"},
    "umbrella":        {"type": "UNKNOWN",  "label": "우산"},
}


def belonging_meta(cls_name: str) -> dict:
    return _BELONGING_MAP.get(cls_name, {"type": "UNKNOWN", "label": cls_name})


class Detector:
    def __init__(
        self,
        person_model:  str   = "yolo11s.pt",   # YOLO11 최신 모델
        luggage_model: str   = "yolov8s-worldv2.pt",
        person_conf:   float = 0.25,
        luggage_conf:  float = 0.2,
        imgsz:         int   = 480,
    ) -> None:
        from ultralytics import YOLO, YOLOWorld

        self._person_model  = YOLO(person_model)
        self._person_conf   = person_conf
        self._luggage_model = YOLOWorld(luggage_model)
        self._luggage_model.set_classes(LUGGAGE_CLASSES)
        self._luggage_conf  = luggage_conf
        self._imgsz         = imgsz

    def detect_person_only(self, frame: np.ndarray) -> DetectionResult:
        """사람만 감지 (매 프레임 호출용)."""
        results = self._person_model.predict(
            frame, conf=self._person_conf, classes=[0],
            imgsz=self._imgsz, verbose=False,
        )
        persons: list[Box] = []
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= self._person_conf:
                    persons.append(Box(
                        xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        cls_name="person",
                    ))
        return DetectionResult(person_boxes=persons, luggage_boxes=[])

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """사람 + 짐 동시 감지 (N프레임마다 호출용)."""
        result = self.detect_person_only(frame)

        l_results = self._luggage_model.predict(
            frame, conf=self._luggage_conf,
            imgsz=self._imgsz, verbose=False,
        )
        for r in l_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = (LUGGAGE_CLASSES[cls_id]
                          if cls_id < len(LUGGAGE_CLASSES) else "unknown")
                if conf >= self._luggage_conf:
                    result.luggage_boxes.append(Box(
                        xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=conf,
                        cls_name=name,
                    ))
        return result

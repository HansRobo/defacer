"""ドメインモデル（GUIに依存しない共有データクラス）"""

import json
from dataclasses import dataclass, asdict


@dataclass
class BoundingBox:
    """バウンディングボックス"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def contains_point(self, x: int, y: int, margin: int = 0) -> bool:
        """点がボックス内にあるか判定"""
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )

    def get_resize_handle(self, x: int, y: int, handle_size: int = 10) -> str | None:
        """リサイズハンドルを取得（nw, ne, sw, se, n, s, e, w）"""
        hs = handle_size

        if abs(x - self.x1) < hs and abs(y - self.y1) < hs:
            return "nw"
        if abs(x - self.x2) < hs and abs(y - self.y1) < hs:
            return "ne"
        if abs(x - self.x1) < hs and abs(y - self.y2) < hs:
            return "sw"
        if abs(x - self.x2) < hs and abs(y - self.y2) < hs:
            return "se"

        if abs(y - self.y1) < hs and self.x1 < x < self.x2:
            return "n"
        if abs(y - self.y2) < hs and self.x1 < x < self.x2:
            return "s"
        if abs(x - self.x1) < hs and self.y1 < y < self.y2:
            return "w"
        if abs(x - self.x2) < hs and self.y1 < y < self.y2:
            return "e"

        return None

    def normalize(self) -> "BoundingBox":
        """座標を正規化（x1 < x2, y1 < y2 にする）"""
        return BoundingBox(
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
        )

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """画像境界内に収める"""
        return BoundingBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
        )

    def scale_from_center(
        self, factor: float, image_width: int = 0, image_height: int = 0
    ) -> "BoundingBox":
        """中心を基点にスケールし、指定サイズで上限クランプ（0=クランプなし）"""
        cx, cy = self.center
        new_w = int(self.width * factor)
        new_h = int(self.height * factor)
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = cx + new_w // 2
        y2 = cy + new_h // 2
        if image_width > 0:
            x2 = min(image_width, x2)
        if image_height > 0:
            y2 = min(image_height, y2)
        return BoundingBox(x1, y1, x2, y2)

    @classmethod
    def interpolate(cls, box1: "BoundingBox", box2: "BoundingBox", t: float) -> "BoundingBox":
        """2つのボックス間を線形補間"""
        return cls(
            x1=int(box1.x1 + (box2.x1 - box1.x1) * t),
            y1=int(box1.y1 + (box2.y1 - box1.y1) * t),
            x2=int(box1.x2 + (box2.x2 - box1.x2) * t),
            y2=int(box1.y2 + (box2.y2 - box1.y2) * t),
        )


@dataclass
class Annotation:
    """単一のアノテーション（1フレーム、1領域）"""

    frame: int
    bbox: BoundingBox
    track_id: int | None = None
    is_manual: bool = True
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "bbox": asdict(self.bbox),
            "track_id": self.track_id,
            "is_manual": self.is_manual,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        return cls(
            frame=data["frame"],
            bbox=BoundingBox(**data["bbox"]),
            track_id=data.get("track_id"),
            is_manual=data.get("is_manual", True),
            confidence=data.get("confidence", 1.0),
        )

    @classmethod
    def from_detection(
        cls,
        det,
        frame: int,
        track_id: int | None,
        bbox_scale: float = 1.0,
        image_width: int = 0,
        image_height: int = 0,
    ) -> "Annotation":
        """DetectionまたはTrackedFaceからAnnotationを生成"""
        bbox = det.bbox
        if bbox_scale != 1.0:
            bbox = bbox.scale_from_center(bbox_scale, image_width, image_height)
        return cls(frame=frame, bbox=bbox, track_id=track_id, is_manual=False, confidence=det.confidence)

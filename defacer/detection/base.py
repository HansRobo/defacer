"""顔検知の抽象ベースクラス"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """検出結果を表すデータクラス"""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: np.ndarray | None = None  # 顔のランドマーク（オプション）

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def scale(self, factor: float, image_shape: tuple[int, int]) -> "Detection":
        """バウンディングボックスを拡大/縮小"""
        h, w = image_shape[:2]
        cx, cy = self.center
        new_w = int(self.width * factor)
        new_h = int(self.height * factor)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w, cx + new_w // 2)
        y2 = min(h, cy + new_h // 2)

        return Detection(
            bbox=(x1, y1, x2, y2),
            confidence=self.confidence,
            landmarks=self.landmarks,
        )


def find_best_iou_match(
    target_bbox: tuple[int, int, int, int],
    candidates,
    threshold: float = 0.3,
):
    """
    IoUで最も一致する候補を返す。

    Args:
        target_bbox: 比較対象のbbox (x1, y1, x2, y2)
        candidates: .bbox 属性を持つオブジェクトのリスト
        threshold: この値以上のIoUが必要（未満の場合はNoneを返す）

    Returns:
        最も一致した候補オブジェクト、またはNone
    """
    best_iou = 0.0
    best_match = None
    for candidate in candidates:
        iou = compute_iou(target_bbox, candidate.bbox)
        if iou > best_iou:
            best_iou = iou
            best_match = candidate
    return best_match if best_iou >= threshold else None


def compute_iou(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
    """2つのバウンディングボックスのIoUを計算"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class FaceDetector(ABC):
    """顔検知の抽象ベースクラス"""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        フレームから顔を検出する

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            検出された顔のリスト
        """
        pass

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        複数フレームから顔を検出する（バッチ処理）

        デフォルト実装は単純なループ。サブクラスで最適化可能。
        """
        return [self.detect(frame) for frame in frames]

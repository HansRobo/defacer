"""顔検知の抽象ベースクラス"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from defacer.models import BoundingBox


@dataclass
class Detection:
    """検出結果を表すデータクラス"""

    bbox: BoundingBox
    confidence: float
    landmarks: np.ndarray | None = None  # 顔のランドマーク（オプション）


def find_best_iou_match(
    target_bbox: BoundingBox,
    candidates,
    threshold: float = 0.3,
):
    """
    IoUで最も一致する候補を返す。

    Args:
        target_bbox: 比較対象のbbox
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


def compute_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """2つのバウンディングボックスのIoUを計算"""
    inter_x1 = max(bbox1.x1, bbox2.x1)
    inter_y1 = max(bbox1.y1, bbox2.y1)
    inter_x2 = min(bbox1.x2, bbox2.x2)
    inter_y2 = min(bbox1.y2, bbox2.y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = bbox1.area + bbox2.area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class FaceDetector(ABC):
    """顔検知の抽象ベースクラス"""

    def __init__(self, confidence_threshold: float = 0.25):
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

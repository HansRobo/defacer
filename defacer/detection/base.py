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

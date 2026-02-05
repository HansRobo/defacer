"""トラッキングの抽象ベースクラス"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from defacer.detection.base import Detection


@dataclass
class TrackedFace:
    """追跡された顔を表すデータクラス"""

    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    age: int = 0  # 最後に検出されてからのフレーム数

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

    def to_detection(self) -> Detection:
        """Detection形式に変換"""
        return Detection(bbox=self.bbox, confidence=self.confidence)


class FaceTracker(ABC):
    """顔トラッキングの抽象ベースクラス"""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Args:
            max_age: 見失ってから保持するフレーム数
            min_hits: 確定するまでの検出回数
        """
        self.max_age = max_age
        self.min_hits = min_hits

    @abstractmethod
    def update(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> list[TrackedFace]:
        """
        検出結果でトラッカーを更新

        Args:
            detections: 現在のフレームでの検出結果
            frame: 現在のフレーム（外観特徴抽出用、オプション）

        Returns:
            追跡中の顔リスト
        """
        pass

    def track(self, frame: np.ndarray) -> list[TrackedFace]:
        """
        検出+トラッキング統合API（オプション）

        Args:
            frame: 現在のフレーム

        Returns:
            追跡中の顔リスト

        Raises:
            NotImplementedError: 統合トラッキング未対応の場合
        """
        raise NotImplementedError("統合トラッキング未対応")

    def supports_integrated_tracking(self) -> bool:
        """
        統合トラッキングをサポートするか

        Returns:
            True: track()メソッドが使用可能
            False: update()メソッドのみ使用可能
        """
        return False

    @abstractmethod
    def reset(self) -> None:
        """トラッカーをリセット"""
        pass

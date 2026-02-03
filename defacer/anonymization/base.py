"""匿名化処理の抽象ベースクラス"""

from abc import ABC, abstractmethod

import numpy as np


class Anonymizer(ABC):
    """匿名化処理の抽象ベースクラス"""

    @abstractmethod
    def apply(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        指定領域に匿名化処理を適用

        Args:
            frame: BGR画像（OpenCV形式）
            bbox: (x1, y1, x2, y2) 形式のバウンディングボックス
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        pass

    def apply_multiple(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        複数の領域に匿名化処理を適用

        Args:
            frame: BGR画像（OpenCV形式）
            bboxes: バウンディングボックスのリスト
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        result = frame.copy()
        for bbox in bboxes:
            result = self.apply(result, bbox, ellipse)
        return result

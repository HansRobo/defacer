"""ガウシアンぼかし・塗りつぶし処理"""

import cv2
import numpy as np

from defacer.anonymization.base import Anonymizer


class GaussianBlurAnonymizer(Anonymizer):
    """ガウシアンぼかし処理"""

    def __init__(self, kernel_size: int = 99):
        """
        Args:
            kernel_size: ぼかしのカーネルサイズ（奇数）
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

    def apply(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        指定領域にガウシアンぼかしを適用

        Args:
            frame: BGR画像（OpenCV形式）
            bbox: (x1, y1, x2, y2) 形式のバウンディングボックス
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        kernel_size = self.kernel_size

        def _blur(roi: np.ndarray) -> np.ndarray:
            roi_h, roi_w = roi.shape[:2]
            ksize = min(kernel_size, roi_w, roi_h)
            if ksize % 2 == 0:
                ksize -= 1
            ksize = max(3, ksize)
            return cv2.GaussianBlur(roi, (ksize, ksize), 0)

        return self._apply_roi(frame, bbox, _blur, ellipse)


class SolidFillAnonymizer(Anonymizer):
    """塗りつぶし処理"""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)):
        """
        Args:
            color: 塗りつぶし色 (B, G, R)
        """
        self.color = color

    def apply(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        指定領域を塗りつぶし

        Args:
            frame: BGR画像（OpenCV形式）
            bbox: (x1, y1, x2, y2) 形式のバウンディングボックス
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        color = self.color

        def _solid(roi: np.ndarray) -> np.ndarray:
            return np.full_like(roi, color)

        return self._apply_roi(frame, bbox, _solid, ellipse)

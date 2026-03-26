"""モザイク（ピクセル化）処理"""

import cv2
import numpy as np

from defacer.anonymization.base import Anonymizer


class MosaicAnonymizer(Anonymizer):
    """ピクセル化モザイク処理"""

    def __init__(self, block_size: int = 10):
        """
        Args:
            block_size: モザイクのブロックサイズ（ピクセル）
        """
        self.block_size = block_size

    def apply(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        指定領域にモザイクを適用

        Args:
            frame: BGR画像（OpenCV形式）
            bbox: (x1, y1, x2, y2) 形式のバウンディングボックス
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        block_size = self.block_size

        def _mosaic(roi: np.ndarray) -> np.ndarray:
            roi_h, roi_w = roi.shape[:2]
            # 縮小→拡大でピクセル化効果
            small_w = max(1, roi_w // block_size)
            small_h = max(1, roi_h // block_size)
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        return self._apply_roi(frame, bbox, _mosaic, ellipse)

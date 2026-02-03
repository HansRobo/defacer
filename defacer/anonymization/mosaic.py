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
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # 境界チェック
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return frame

        result = frame.copy()
        roi = result[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        if roi_h == 0 or roi_w == 0:
            return frame

        # モザイク処理：縮小→拡大でピクセル化効果
        # ブロックサイズに基づいて縮小サイズを計算
        small_w = max(1, roi_w // self.block_size)
        small_h = max(1, roi_h // self.block_size)

        # 縮小して拡大（ピクセル化効果）
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        if ellipse:
            # 楕円形マスクを作成
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            center = (roi_w // 2, roi_h // 2)
            axes = (roi_w // 2, roi_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            # マスクを使って合成
            mask_3ch = cv2.merge([mask, mask, mask])
            roi_masked = np.where(mask_3ch > 0, mosaic, roi)
            result[y1:y2, x1:x2] = roi_masked
        else:
            # 矩形でそのまま適用
            result[y1:y2, x1:x2] = mosaic

        return result

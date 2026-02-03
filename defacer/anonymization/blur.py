"""ガウシアンぼかし処理"""

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
        # カーネルサイズは奇数である必要がある
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

        # カーネルサイズをROIサイズに合わせて調整
        ksize = min(self.kernel_size, roi_w, roi_h)
        if ksize % 2 == 0:
            ksize -= 1
        ksize = max(3, ksize)

        # ガウシアンぼかしを適用
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        if ellipse:
            # 楕円形マスクを作成
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            center = (roi_w // 2, roi_h // 2)
            axes = (roi_w // 2, roi_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

            # マスクを使って合成
            mask_3ch = cv2.merge([mask, mask, mask])
            roi_masked = np.where(mask_3ch > 0, blurred, roi)
            result[y1:y2, x1:x2] = roi_masked
        else:
            # 矩形でそのまま適用
            result[y1:y2, x1:x2] = blurred

        return result


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

        if ellipse:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(result, center, axes, 0, 0, 360, self.color, -1)
        else:
            cv2.rectangle(result, (x1, y1), (x2, y2), self.color, -1)

        return result

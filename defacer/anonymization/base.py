"""匿名化処理の抽象ベースクラス"""

from abc import ABC, abstractmethod
from typing import Callable

import cv2
import numpy as np

from defacer.models import BoundingBox


class Anonymizer(ABC):
    """匿名化処理の抽象ベースクラス"""

    @abstractmethod
    def apply(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        ellipse: bool = True,
    ) -> np.ndarray:
        """
        指定領域に匿名化処理を適用

        Args:
            frame: BGR画像（OpenCV形式）
            bbox: バウンディングボックス
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム
        """
        pass

    def _apply_roi(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        transform_roi: Callable[[np.ndarray], np.ndarray],
        ellipse: bool,
    ) -> np.ndarray:
        """
        ROI変換を適用する共通テンプレートメソッド。

        境界クランプ・空ROIチェック・frame.copy・楕円マスク合成を一元化する。
        サブクラスは transform_roi にROI変換ロジックのみを渡す。

        Args:
            frame: 入力フレーム
            bbox: バウンディングボックス
            transform_roi: ROI（numpy配列）を受け取り変換後ROIを返す関数
            ellipse: Trueの場合は楕円形、Falseの場合は矩形でマスク

        Returns:
            処理後のフレーム（変更なしの場合は元のframeオブジェクトを返す）
        """
        h, w = frame.shape[:2]
        b = bbox.clamp(w, h)
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2

        if x2 <= x1 or y2 <= y1:
            return frame

        result = frame.copy()
        roi = result[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        if roi_h == 0 or roi_w == 0:
            return frame

        transformed = transform_roi(roi)

        if ellipse:
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.ellipse(mask, (roi_w // 2, roi_h // 2), (roi_w // 2, roi_h // 2), 0, 0, 360, 255, -1)
            mask_3ch = cv2.merge([mask, mask, mask])
            result[y1:y2, x1:x2] = np.where(mask_3ch > 0, transformed, roi)
        else:
            result[y1:y2, x1:x2] = transformed

        return result

    def apply_multiple(
        self,
        frame: np.ndarray,
        bboxes: list[BoundingBox],
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
        result = frame
        for bbox in bboxes:
            result = self.apply(result, bbox, ellipse)
        return result

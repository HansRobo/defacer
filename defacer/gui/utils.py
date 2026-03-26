"""GUIユーティリティ関数"""

import numpy as np
from PyQt5.QtGui import QImage


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """BGR numpy配列をQImage(RGB888)に変換"""
    frame_rgb = frame[:, :, ::-1].copy()
    h, w, ch = frame_rgb.shape
    return QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

"""顔検知モジュール"""

from defacer.detection.base import FaceDetector, Detection

__all__ = ["FaceDetector", "Detection"]


def get_available_detectors() -> list[str]:
    """利用可能な検出器のリストを取得"""
    available = []

    try:
        from defacer.detection.yolo11_face import is_yolo11_available
        if is_yolo11_available():
            available.append("yolo11-face")
    except ImportError:
        pass

    return available


def create_detector(detector_type: str, **kwargs) -> FaceDetector:
    """指定タイプの検出器を作成"""
    if detector_type == "yolo11-face":
        from defacer.detection.yolo11_face import YOLO11FaceDetector
        return YOLO11FaceDetector(**kwargs)
    else:
        raise ValueError(f"不明な検出器タイプ: {detector_type}")

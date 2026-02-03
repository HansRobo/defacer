"""顔検知モジュール"""

from defacer.detection.base import FaceDetector, Detection

__all__ = ["FaceDetector", "Detection"]


def get_available_detectors() -> list[str]:
    """利用可能な検出器のリストを取得"""
    available = []

    try:
        from defacer.detection.retinaface import is_retinaface_available
        if is_retinaface_available():
            available.append("retinaface")
    except ImportError:
        pass

    try:
        from defacer.detection.yolov8_face import is_yolov8_available
        if is_yolov8_available():
            available.append("yolov8-face")
    except ImportError:
        pass

    return available


def create_detector(detector_type: str, **kwargs) -> FaceDetector:
    """指定タイプの検出器を作成"""
    if detector_type == "retinaface":
        from defacer.detection.retinaface import RetinaFaceDetector
        return RetinaFaceDetector(**kwargs)
    elif detector_type == "yolov8-face":
        from defacer.detection.yolov8_face import YOLOv8FaceDetector
        return YOLOv8FaceDetector(**kwargs)
    else:
        raise ValueError(f"不明な検出器タイプ: {detector_type}")

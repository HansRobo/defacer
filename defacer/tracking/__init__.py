"""トラッキングモジュール"""

from defacer.tracking.base import FaceTracker, TrackedFace
from defacer.tracking.interpolation import interpolate_track, interpolate_all_tracks

__all__ = [
    "FaceTracker",
    "TrackedFace",
    "interpolate_track",
    "interpolate_all_tracks",
]


def is_tracking_available() -> bool:
    """トラッキング機能が利用可能か確認"""
    try:
        from defacer.tracking.sort_tracker import is_deepsort_available
        return is_deepsort_available()
    except ImportError:
        return False


def create_tracker(use_deepsort: bool = True, **kwargs) -> FaceTracker:
    """トラッカーを作成"""
    from defacer.tracking.sort_tracker import create_tracker as _create
    return _create(use_deepsort, **kwargs)

"""トラッキングモジュール"""

from defacer.tracking.base import FaceTracker, TrackedFace
from defacer.tracking.interpolation import interpolate_track, interpolate_all_tracks

__all__ = [
    "FaceTracker",
    "TrackedFace",
    "interpolate_track",
    "interpolate_all_tracks",
]


def create_tracker(**kwargs) -> FaceTracker:
    """トラッカーを作成"""
    from defacer.tracking.sort_tracker import create_tracker as _create
    return _create(**kwargs)

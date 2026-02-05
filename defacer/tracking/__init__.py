"""トラッキングモジュール"""

from defacer.tracking.base import FaceTracker, TrackedFace
from defacer.tracking.interpolation import interpolate_track, interpolate_all_tracks

__all__ = [
    "FaceTracker",
    "TrackedFace",
    "interpolate_track",
    "interpolate_all_tracks",
    "get_available_trackers",
    "create_tracker",
]


def get_available_trackers() -> list[str]:
    """
    利用可能なトラッカーを取得

    Returns:
        利用可能なトラッカー名のリスト（例: ["botsort", "bytetrack"]）
    """
    try:
        from ultralytics import YOLO
        return ["botsort", "bytetrack"]
    except ImportError:
        return []


def create_tracker(tracker_type: str = "botsort", **kwargs) -> FaceTracker:
    """
    トラッカーを作成

    Args:
        tracker_type: トラッカーの種類 ("botsort" または "bytetrack")
        **kwargs: トラッカーに渡す追加パラメータ

    Returns:
        FaceTracker インスタンス

    Raises:
        ValueError: 指定されたトラッカーが利用不可の場合
    """
    available = get_available_trackers()
    if not available:
        raise ValueError(
            "Ultralyticsトラッキングが利用できません。"
            "ultralytics パッケージをインストールしてください。"
        )

    if tracker_type not in available:
        raise ValueError(
            f"トラッカー '{tracker_type}' は利用できません。"
            f"利用可能: {available}"
        )

    from defacer.tracking.ultralytics_tracker import UltralyticsTracker
    return UltralyticsTracker(tracker=tracker_type, **kwargs)

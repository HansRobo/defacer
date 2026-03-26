"""YOLOモデルの共通ローダー"""


def load_yolo_model(model_path: str | None = None):
    """
    YOLOv11顔検出モデルをロードして返す（遅延初期化）。

    Args:
        model_path: カスタムモデルのパス（Noneの場合はデフォルトモデルをダウンロード）

    Returns:
        YOLO モデルインスタンス

    Raises:
        ImportError: ultralytics がインストールされていない場合
        RuntimeError: モデルのダウンロードに失敗した場合
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralyticsがインストールされていません。\n"
            "pip install ultralytics でインストールしてください。"
        )

    if model_path:
        return YOLO(model_path)

    from defacer.detection.yolo11_face import download_yolo11_face_model
    return YOLO(download_yolo11_face_model())

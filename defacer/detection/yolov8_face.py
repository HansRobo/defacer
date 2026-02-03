"""YOLOv8-Face顔検知実装"""

import numpy as np

from defacer.detection.base import FaceDetector, Detection


class YOLOv8FaceDetector(FaceDetector):
    """YOLOv8-Faceを使用した高速顔検知"""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        model_path: str | None = None,
    ):
        """
        Args:
            confidence_threshold: 検出信頼度の閾値（0.0-1.0）
            model_path: カスタムモデルのパス（Noneの場合はデフォルトモデル）
        """
        super().__init__(confidence_threshold)
        self._model = None
        self._model_path = model_path
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """モデルの遅延初期化"""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralyticsがインストールされていません。\n"
                "pip install ultralytics でインストールしてください。"
            )

        if self._model_path:
            self._model = YOLO(self._model_path)
        else:
            # YOLOv8n-faceモデルを使用
            # 注: 実際には事前にダウンロードしたモデルを使用する必要があります
            # デフォルトのYOLOv8nは一般物体検出用なので、
            # face検出用にファインチューニングされたモデルが必要
            self._model = YOLO("yolov8n.pt")  # 一般物体検出モデル

        self._initialized = True

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        フレームから顔を検出

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            検出された顔のリスト
        """
        self._ensure_initialized()

        try:
            # YOLOv8で検出
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            print(f"YOLOv8検出エラー: {e}")
            return []

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # クラスIDが0（person）の場合のみ処理
                # 注: face専用モデルの場合はクラスIDが異なる可能性あり
                cls_id = int(box.cls[0]) if box.cls is not None else -1

                # 一般モデルの場合、personクラス（ID=0）のみを検出
                # face専用モデルの場合はこの条件を変更
                if cls_id != 0:
                    continue

                confidence = float(box.conf[0]) if box.conf is not None else 0.0

                if confidence < self.confidence_threshold:
                    continue

                # バウンディングボックスを取得
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    landmarks=None,
                )
                detections.append(detection)

        return detections


def is_yolov8_available() -> bool:
    """YOLOv8が利用可能か確認"""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False

"""YOLOv11-Face顔検知実装"""

import numpy as np

from defacer.detection.base import FaceDetector, Detection


class YOLO11FaceDetector(FaceDetector):
    """YOLOv11-Faceを使用した高速顔検知（WIDERFACE訓練済み）"""

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
            # YOLOv11n顔検出モデルを使用（WIDERFACE訓練済み）
            # HuggingFace Hubから自動ダウンロード
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "huggingface_hubがインストールされていません。\n"
                    "pip install huggingface-hub でインストールしてください。"
                )

            print("YOLOv11顔検出モデルをダウンロード中...")
            try:
                model_path = hf_hub_download(
                    repo_id="AdamCodd/YOLOv11n-face-detection",
                    filename="model.pt",
                    cache_dir=None,  # デフォルトキャッシュを使用
                )
                print(f"ダウンロード完了: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"YOLOv11顔検出モデルのダウンロードに失敗しました: {e}\n"
                    "手動でダウンロードしてください: "
                    "https://huggingface.co/AdamCodd/YOLOv11n-face-detection"
                )

            self._model = YOLO(model_path)

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
            # YOLOv11で検出
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as e:
            print(f"YOLOv11検出エラー: {e}")
            return []

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # 顔検出専用モデルのため、クラスフィルタリング不要
                # すべての検出結果が顔として扱われる
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


def is_yolo11_available() -> bool:
    """YOLOv11が利用可能か確認"""
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        return True
    except ImportError:
        return False

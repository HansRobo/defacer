"""RetinaFace顔検知実装"""

import numpy as np

from defacer.detection.base import FaceDetector, Detection


class RetinaFaceDetector(FaceDetector):
    """RetinaFaceを使用した高精度顔検知"""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: 検出信頼度の閾値（0.0-1.0）
        """
        super().__init__(confidence_threshold)
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """モデルの遅延初期化"""
        if self._initialized:
            return

        try:
            from retinaface import RetinaFace as RF
            self._rf_module = RF
            self._initialized = True
        except ImportError:
            raise ImportError(
                "RetinaFaceがインストールされていません。\n"
                "pip install retina-face でインストールしてください。"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        フレームから顔を検出

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            検出された顔のリスト
        """
        self._ensure_initialized()

        # RetinaFaceはRGB画像を期待
        rgb_frame = frame[:, :, ::-1]

        try:
            # 顔検出を実行
            faces = self._rf_module.detect_faces(rgb_frame)
        except Exception as e:
            print(f"RetinaFace検出エラー: {e}")
            return []

        if faces is None or not isinstance(faces, dict):
            return []

        detections = []
        for face_key, face_data in faces.items():
            confidence = face_data.get("score", 0.0)

            if confidence < self.confidence_threshold:
                continue

            facial_area = face_data.get("facial_area", [])
            if len(facial_area) != 4:
                continue

            x1, y1, x2, y2 = facial_area

            # ランドマークを取得
            landmarks = None
            if "landmarks" in face_data:
                lm = face_data["landmarks"]
                landmarks = np.array([
                    lm.get("left_eye", [0, 0]),
                    lm.get("right_eye", [0, 0]),
                    lm.get("nose", [0, 0]),
                    lm.get("mouth_left", [0, 0]),
                    lm.get("mouth_right", [0, 0]),
                ])

            detection = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(confidence),
                landmarks=landmarks,
            )
            detections.append(detection)

        return detections


def is_retinaface_available() -> bool:
    """RetinaFaceが利用可能か確認"""
    try:
        from retinaface import RetinaFace
        return True
    except ImportError:
        return False

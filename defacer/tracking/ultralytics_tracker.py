"""Ultralytics YOLO 組み込みトラッキング実装"""

import numpy as np

from defacer.tracking.base import FaceTracker, TrackedFace
from defacer.detection.base import Detection, compute_iou
from defacer.detection.yolo11_face import download_yolo11_face_model


class UltralyticsTracker(FaceTracker):
    """Ultralytics YOLO 組み込みトラッキング（BotSORT/ByteTrack）"""

    def __init__(
        self,
        tracker: str = "botsort",
        confidence_threshold: float = 0.25,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        """
        Args:
            tracker: 使用するトラッカー ("botsort" または "bytetrack")
            confidence_threshold: 検出の信頼度閾値
            max_age: 見失ってから保持するフレーム数
            min_hits: 確定するまでの検出回数
        """
        super().__init__(max_age, min_hits)
        self._tracker_type = tracker
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """YOLOモデルの遅延初期化"""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralyticsがインストールされていません。\n"
                "pip install ultralytics でインストールしてください。"
            )

        model_path = download_yolo11_face_model()
        self._model = YOLO(model_path)
        self._initialized = True

    def track(self, frame: np.ndarray) -> list[TrackedFace]:
        """
        検出+トラッキング統合API

        Args:
            frame: 現在のフレーム

        Returns:
            追跡中の顔リスト
        """
        self._ensure_initialized()

        # YOLO トラッキングを実行
        results = self._model.track(
            frame,
            persist=True,
            tracker=f"{self._tracker_type}.yaml",
            conf=self._confidence_threshold,
            verbose=False,
        )

        tracked_faces = []
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, conf, track_id in zip(boxes, confidences, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    tracked_face = TrackedFace(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        age=0,
                    )
                    tracked_faces.append(tracked_face)

        return tracked_faces

    def update(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> list[TrackedFace]:
        """
        既存API互換 - RetrackDialog用

        Args:
            detections: 現在のフレームでの検出結果
            frame: 現在のフレーム（オプション）

        Returns:
            追跡中の顔リスト
        """
        if frame is None:
            # フレームがない場合は単純に連番を割り当て
            return [
                TrackedFace(
                    track_id=i,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                for i, det in enumerate(detections)
            ]

        # フレームがある場合はtrack()を実行してマッチング
        tracked = self.track(frame)
        return self._match_with_detections(tracked, detections)

    def _match_with_detections(
        self, tracked: list[TrackedFace], detections: list[Detection]
    ) -> list[TrackedFace]:
        """
        トラッキング結果と検出結果をIoUでマッチング

        Args:
            tracked: track()の結果
            detections: 元の検出結果

        Returns:
            マッチした顔リスト（元のbboxを使用）
        """
        if not tracked or not detections:
            return tracked

        # マッチング
        matched_faces = []
        for track_face in tracked:
            best_iou = 0.0
            best_det = None

            for det in detections:
                iou = compute_iou(track_face.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_det is not None and best_iou > 0.3:  # IoU閾値
                # 元のbboxを使用
                matched_face = TrackedFace(
                    track_id=track_face.track_id,
                    bbox=best_det.bbox,
                    confidence=best_det.confidence,
                    age=track_face.age,
                )
                matched_faces.append(matched_face)

        return matched_faces

    def supports_integrated_tracking(self) -> bool:
        """統合トラッキングをサポート"""
        return True

    def reset(self) -> None:
        """トラッカーをリセット（新しいビデオ用）"""
        if self._model is not None:
            # Ultralytics の内部状態をクリア
            self._model.predictor = None
            self._initialized = False
            self._model = None

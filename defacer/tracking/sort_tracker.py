"""DeepSORTトラッキング実装"""

import numpy as np

from defacer.tracking.base import FaceTracker, TrackedFace
from defacer.detection.base import Detection


class DeepSORTFaceTracker(FaceTracker):
    """DeepSORTを使用した顔トラッキング"""

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Args:
            max_age: 見失ってから保持するフレーム数
            min_hits: 確定するまでの検出回数
        """
        super().__init__(max_age, min_hits)
        self._tracker = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """トラッカーの遅延初期化"""
        if self._initialized:
            return

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                nms_max_overlap=0.7,
                max_cosine_distance=0.3,
                nn_budget=100,
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "deep-sort-realtimeがインストールされていません。\n"
                "pip install deep-sort-realtime でインストールしてください。"
            )

    def update(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> list[TrackedFace]:
        """
        検出結果でトラッカーを更新

        Args:
            detections: 現在のフレームでの検出結果
            frame: 現在のフレーム（外観特徴抽出用）

        Returns:
            追跡中の顔リスト
        """
        self._ensure_initialized()

        if frame is None:
            return [
                TrackedFace(
                    track_id=i,
                    bbox=det.bbox,
                    confidence=det.confidence,
                )
                for i, det in enumerate(detections)
            ]

        # DeepSORT形式に変換 [[x1, y1, w, h, confidence], ...]
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            dets.append(([x1, y1, w, h], det.confidence, "face"))

        # トラッカーを更新
        tracks = self._tracker.update_tracks(dets, frame=frame)

        tracked_faces = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]

            tracked_face = TrackedFace(
                track_id=int(track_id),
                bbox=(int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])),
                confidence=1.0,
                age=track.time_since_update,
            )
            tracked_faces.append(tracked_face)

        return tracked_faces

    def reset(self) -> None:
        """トラッカーをリセット"""
        if self._tracker is not None:
            self._tracker = None
            self._initialized = False


class SimpleTracker(FaceTracker):
    """シンプルなIoUベーストラッカー（DeepSORTが利用できない場合用）"""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        super().__init__(max_age, min_hits)
        self.iou_threshold = iou_threshold
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> list[TrackedFace]:
        """IoUベースのトラッキング"""
        if not detections:
            # 全トラックをエージング
            for track_id in list(self._tracks.keys()):
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]
            return []

        # 既存トラックと検出をマッチング
        matched = set()
        for det in detections:
            best_iou = 0
            best_track_id = None

            for track_id, track in self._tracks.items():
                if track_id in matched:
                    continue
                iou = self._compute_iou(det.bbox, track["bbox"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                # 既存トラックを更新
                self._tracks[best_track_id]["bbox"] = det.bbox
                self._tracks[best_track_id]["confidence"] = det.confidence
                self._tracks[best_track_id]["age"] = 0
                self._tracks[best_track_id]["hits"] += 1
                matched.add(best_track_id)
            else:
                # 新規トラック
                self._tracks[self._next_id] = {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "age": 0,
                    "hits": 1,
                }
                matched.add(self._next_id)
                self._next_id += 1

        # マッチしなかったトラックをエージング
        for track_id in list(self._tracks.keys()):
            if track_id not in matched:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]

        # 確定したトラックを返す
        tracked_faces = []
        for track_id, track in self._tracks.items():
            if track["hits"] >= self.min_hits:
                tracked_faces.append(TrackedFace(
                    track_id=track_id,
                    bbox=track["bbox"],
                    confidence=track["confidence"],
                    age=track["age"],
                ))

        return tracked_faces

    def _compute_iou(
        self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
    ) -> float:
        """IoU（Intersection over Union）を計算"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0

        return inter_area / union_area

    def reset(self) -> None:
        """トラッカーをリセット"""
        self._tracks.clear()
        self._next_id = 1


def is_deepsort_available() -> bool:
    """DeepSORTが利用可能か確認"""
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        return True
    except ImportError:
        return False


def create_tracker(use_deepsort: bool = True, **kwargs) -> FaceTracker:
    """トラッカーを作成"""
    if use_deepsort and is_deepsort_available():
        return DeepSORTFaceTracker(**kwargs)
    return SimpleTracker(**kwargs)

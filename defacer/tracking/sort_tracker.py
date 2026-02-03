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

        import os
        import torch
        from deep_sort_realtime.deepsort_tracker import DeepSort

        # GPU使用の判定
        use_gpu = False
        use_half = False

        if torch.cuda.is_available():
            if torch.version.hip is not None:
                # ROCm環境
                # gfx1103などの新しいアーキテクチャは公式PyTorchでサポートされていない可能性がある
                # 環境変数DEFACER_FORCE_ROCMが設定されている場合のみGPUを試す
                if os.environ.get("DEFACER_FORCE_ROCM") == "1":
                    use_gpu = True
                    use_half = False
                else:
                    # デフォルトはCPUモード（安定性優先）
                    import warnings
                    warnings.warn(
                        "ROCm環境が検出されましたが、gfx1103など一部のGPUは公式PyTorchでサポートされていません。"
                        "CPUモードで実行します。GPUを強制的に使用する場合は環境変数 DEFACER_FORCE_ROCM=1 を設定してください。",
                        RuntimeWarning
                    )
                    use_gpu = False
            else:
                # CUDA環境ではFP16を使用
                use_gpu = True
                use_half = True

        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder_gpu=use_gpu,
            half=use_half,
        )
        self._initialized = True

    def update(
        self, detections: list[Detection], frame: np.ndarray | None = None
    ) -> list[TrackedFace]:
        """
        検出結果でトラッカーを更新

        Args:
            detections: 現在のフレームでの検出結果
            frame: 現在のフレーム(外観特徴抽出用)

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


def create_tracker(**kwargs) -> FaceTracker:
    """トラッカーを作成"""
    return DeepSORTFaceTracker(**kwargs)

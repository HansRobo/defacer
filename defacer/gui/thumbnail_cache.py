"""トラックの顔サムネイルをキャッシュ"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from defacer.gui.annotation import AnnotationStore, BoundingBox
from defacer.gui.utils import bgr_to_qimage


class ThumbnailCache(QObject):
    """トラックの顔サムネイルをキャッシュ"""

    thumbnail_loaded = pyqtSignal(int, QPixmap)  # track_id, pixmap

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._cache: dict[int, QPixmap] = {}  # track_id -> QPixmap
        self._cap: Optional[cv2.VideoCapture] = None

    def _ensure_video_opened(self) -> bool:
        """動画が開けることを確認"""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._video_path)
        return self._cap.isOpened()

    def get_track_thumbnail(
        self, track_id: int, store: AnnotationStore, size: tuple[int, int] = (64, 64)
    ) -> Optional[QPixmap]:
        """トラックの最初のフレームからサムネイルを取得

        Args:
            track_id: トラックID
            store: アノテーションストア
            size: サムネイルサイズ (width, height)

        Returns:
            QPixmap or None
        """
        # キャッシュチェック
        if track_id in self._cache:
            return self._cache[track_id]

        frames = store.get_track_frames(track_id)
        if not frames:
            return None

        first_ann = store.get_annotation_by_frame_track(frames[0], track_id)
        thumbnail = self.get_frame_thumbnail(first_ann.frame, first_ann.bbox, size)

        # キャッシュに保存
        if thumbnail is not None:
            self._cache[track_id] = thumbnail

        return thumbnail

    def get_frame_thumbnail(
        self, frame: int, bbox: BoundingBox, size: tuple[int, int] = (64, 64)
    ) -> Optional[QPixmap]:
        """指定フレーム・bboxから顔サムネイルを取得

        Args:
            frame: フレーム番号
            bbox: バウンディングボックス
            size: サムネイルサイズ (width, height)

        Returns:
            QPixmap or None
        """
        if not self._ensure_video_opened():
            return None

        # フレームを読み込む
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self._cap.read()
        if not ret:
            return None

        # bboxで顔領域をクロップ
        face_img = bbox.crop_from(img)
        if face_img is None:
            return None

        # リサイズ
        face_img = cv2.resize(face_img, size, interpolation=cv2.INTER_AREA)

        return QPixmap.fromImage(bgr_to_qimage(face_img))

    def preload_tracks(self, track_ids: list[int], store: AnnotationStore) -> None:
        """複数トラックのサムネイルをバックグラウンドでプリロード

        Args:
            track_ids: プリロードするトラックIDのリスト
            store: アノテーションストア
        """
        for track_id in track_ids:
            if track_id not in self._cache:
                self.get_track_thumbnail(track_id, store)

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()

    def close(self) -> None:
        """リソースを解放"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.clear_cache()


class ThumbnailLoader(QThread):
    """バックグラウンドでサムネイルをロードするスレッド"""

    thumbnail_loaded = pyqtSignal(int, QPixmap)  # track_id, pixmap

    def __init__(
        self,
        cache: ThumbnailCache,
        track_ids: list[int],
        store: AnnotationStore,
        parent=None,
    ):
        super().__init__(parent)
        self._cache = cache
        self._track_ids = track_ids
        self._store = store

    def run(self):
        """バックグラウンド実行"""
        for track_id in self._track_ids:
            thumbnail = self._cache.get_track_thumbnail(track_id, self._store)
            if thumbnail is not None:
                self.thumbnail_loaded.emit(track_id, thumbnail)

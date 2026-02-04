"""トラックの顔サムネイルをキャッシュ"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from defacer.gui.annotation import AnnotationStore, BoundingBox


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

        # トラックのアノテーションをフィルタして最初のフレームを取得
        annotations = [ann for ann in store if ann.track_id == track_id]
        if not annotations:
            return None

        # フレーム番号でソートして最初のアノテーションを使用
        annotations.sort(key=lambda ann: ann.frame)
        first_ann = annotations[0]
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
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # 境界チェック
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_img = img[y1:y2, x1:x2]

        # リサイズ
        face_img = cv2.resize(face_img, size, interpolation=cv2.INTER_AREA)

        # BGR -> RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # numpy -> QPixmap
        h, w, ch = face_img.shape
        bytes_per_line = ch * w
        q_image = QImage(face_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        return pixmap

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

"""動画プレーヤーウィジェット"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QCursor,
)
from PyQt5.QtWidgets import QLabel, QSizePolicy

import numpy as np

from defacer.video.reader import VideoReader
from defacer.gui.annotation import BoundingBox, Annotation, AnnotationStore


class VideoPlayerWidget(QLabel):
    """動画を表示するウィジェット"""

    frame_changed = pyqtSignal(int)  # フレーム番号が変わった時
    playback_state_changed = pyqtSignal(bool)  # 再生状態が変わった時
    annotation_added = pyqtSignal(object)  # アノテーションが追加された時
    annotation_selected = pyqtSignal(object)  # アノテーションが選択された時
    annotations_changed = pyqtSignal()  # アノテーションが変更された時

    # 編集モード
    MODE_VIEW = "view"
    MODE_DRAW = "draw"
    MODE_EDIT = "edit"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._reader: VideoReader | None = None
        self._current_frame: np.ndarray | None = None
        self._current_frame_number: int = 0
        self._is_playing: bool = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        # 自動補間モード
        self._auto_interpolate: bool = False

        # アノテーション管理
        self._annotation_store = AnnotationStore()
        self._selected_annotation: Annotation | None = None
        self._selected_index: int = -1

        # 描画中の矩形
        self._drawing_rect: BoundingBox | None = None
        self._mouse_start: tuple[int, int] | None = None
        self._is_drawing = False

        # 編集中の状態
        self._edit_mode = self.MODE_DRAW
        self._resize_handle: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_offset: tuple[int, int] = (0, 0)

        # 画像のスケールとオフセット（座標変換用）
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def annotation_store(self) -> AnnotationStore:
        return self._annotation_store

    def set_annotation_store(self, store: AnnotationStore) -> None:
        """アノテーションストアを設定"""
        self._annotation_store = store
        self._selected_annotation = None
        self._selected_index = -1
        self._update_display()

    def set_edit_mode(self, mode: str) -> None:
        """編集モードを設定"""
        self._edit_mode = mode
        if mode == self.MODE_VIEW:
            self._selected_annotation = None
            self._selected_index = -1
        self._update_display()

    def set_auto_interpolate(self, enabled: bool) -> None:
        """自動補間モードを設定"""
        self._auto_interpolate = enabled

    @property
    def auto_interpolate(self) -> bool:
        return self._auto_interpolate

    def load_video(self, path: str) -> bool:
        """動画を読み込む"""
        try:
            self.stop()
            if self._reader is not None:
                self._reader.release()

            self._reader = VideoReader(path)
            self._current_frame_number = 0
            self._annotation_store.clear(save_undo=False)
            self._show_frame(0)
            return True
        except Exception as e:
            print(f"動画読み込みエラー: {e}")
            return False

    def _show_frame(self, frame_number: int) -> bool:
        """指定フレームを表示"""
        if self._reader is None:
            return False

        frame = self._reader.read_frame(frame_number)
        if frame is None:
            return False

        self._current_frame = frame
        self._current_frame_number = frame_number

        # 選択を解除（フレームが変わったら）
        self._selected_annotation = None
        self._selected_index = -1

        self._update_display()
        self.frame_changed.emit(frame_number)
        return True

    def _update_display(self) -> None:
        """表示を更新"""
        if self._current_frame is None:
            return

        # BGRからRGBに変換
        frame_rgb = self._current_frame[:, :, ::-1].copy()
        h, w, ch = frame_rgb.shape

        # QImageに変換
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # ウィジェットサイズに合わせてスケール
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # スケールとオフセットを計算（座標変換用）
        self._scale = scaled_pixmap.width() / w
        self._offset_x = (self.width() - scaled_pixmap.width()) // 2
        self._offset_y = (self.height() - scaled_pixmap.height()) // 2

        # アノテーションを描画
        painter = QPainter(scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 現在のフレームのアノテーションを描画
        annotations = self._annotation_store.get_frame_annotations(self._current_frame_number)
        for i, ann in enumerate(annotations):
            is_selected = ann is self._selected_annotation
            self._draw_annotation(painter, ann, is_selected)

        # 描画中の矩形
        if self._drawing_rect:
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 0, 30)))
            self._draw_bbox(painter, self._drawing_rect)

        painter.end()
        self.setPixmap(scaled_pixmap)

    def _draw_annotation(self, painter: QPainter, ann: Annotation, is_selected: bool) -> None:
        """アノテーションを描画"""
        if is_selected:
            pen = QPen(QColor(0, 200, 255), 3)
            brush = QBrush(QColor(0, 200, 255, 40))
        elif ann.is_manual:
            pen = QPen(QColor(0, 255, 0), 2)
            brush = QBrush(QColor(0, 255, 0, 30))
        else:
            pen = QPen(QColor(255, 165, 0), 2)
            brush = QBrush(QColor(255, 165, 0, 30))

        painter.setPen(pen)
        painter.setBrush(brush)
        self._draw_bbox(painter, ann.bbox)

        # 選択時はリサイズハンドルを描画
        if is_selected and self._edit_mode == self.MODE_EDIT:
            self._draw_resize_handles(painter, ann.bbox)

    def _draw_bbox(self, painter: QPainter, bbox: BoundingBox) -> None:
        """バウンディングボックスを描画"""
        x1 = int(bbox.x1 * self._scale)
        y1 = int(bbox.y1 * self._scale)
        w = int(bbox.width * self._scale)
        h = int(bbox.height * self._scale)
        painter.drawRect(x1, y1, w, h)

    def _draw_resize_handles(self, painter: QPainter, bbox: BoundingBox) -> None:
        """リサイズハンドルを描画"""
        handle_size = 8
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        # スケール済み座標
        x1 = int(bbox.x1 * self._scale)
        y1 = int(bbox.y1 * self._scale)
        x2 = int(bbox.x2 * self._scale)
        y2 = int(bbox.y2 * self._scale)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        handles = [
            (x1, y1),  # nw
            (cx, y1),  # n
            (x2, y1),  # ne
            (x1, cy),  # w
            (x2, cy),  # e
            (x1, y2),  # sw
            (cx, y2),  # s
            (x2, y2),  # se
        ]

        for hx, hy in handles:
            painter.drawRect(
                hx - handle_size // 2,
                hy - handle_size // 2,
                handle_size,
                handle_size,
            )

    def _widget_to_frame_coords(self, x: int, y: int) -> tuple[int, int] | None:
        """ウィジェット座標をフレーム座標に変換"""
        if self._reader is None:
            return None

        # オフセットを引いてスケールで割る
        frame_x = (x - self._offset_x) / self._scale
        frame_y = (y - self._offset_y) / self._scale

        # 範囲チェック
        if 0 <= frame_x < self._reader.width and 0 <= frame_y < self._reader.height:
            return (int(frame_x), int(frame_y))
        return None

    def mousePressEvent(self, event) -> None:
        """マウスボタン押下"""
        if event.button() != Qt.LeftButton:
            return

        coords = self._widget_to_frame_coords(event.x(), event.y())
        if coords is None:
            return

        x, y = coords

        if self._edit_mode == self.MODE_VIEW:
            return

        if self._edit_mode == self.MODE_EDIT:
            # 選択中のアノテーションのリサイズハンドルをチェック
            if self._selected_annotation:
                handle = self._selected_annotation.bbox.get_resize_handle(
                    x, y, int(10 / self._scale)
                )
                if handle:
                    self._resize_handle = handle
                    self._drag_start = coords
                    return

            # アノテーションの選択/移動
            result = self._annotation_store.get_annotation_at_point(
                self._current_frame_number, x, y
            )
            if result:
                ann, idx = result
                self._selected_annotation = ann
                self._selected_index = idx
                self._drag_start = coords
                self._drag_offset = (x - ann.bbox.x1, y - ann.bbox.y1)
                self.annotation_selected.emit(ann)
                self._update_display()
                return

            # 何もない場所をクリック→選択解除
            self._selected_annotation = None
            self._selected_index = -1
            self.annotation_selected.emit(None)
            self._update_display()

        elif self._edit_mode == self.MODE_DRAW:
            # 新規描画開始
            self._is_drawing = True
            self._mouse_start = coords
            self._drawing_rect = BoundingBox(x, y, x, y)

    def mouseMoveEvent(self, event) -> None:
        """マウス移動"""
        coords = self._widget_to_frame_coords(event.x(), event.y())

        if self._edit_mode == self.MODE_VIEW:
            return

        # カーソル形状の更新
        if self._edit_mode == self.MODE_EDIT and coords:
            self._update_cursor(coords[0], coords[1])

        if coords is None:
            return

        x, y = coords

        if self._edit_mode == self.MODE_DRAW and self._is_drawing:
            # 描画中
            if self._mouse_start:
                self._drawing_rect = BoundingBox(
                    self._mouse_start[0], self._mouse_start[1], x, y
                ).normalize()
                self._update_display()

        elif self._edit_mode == self.MODE_EDIT:
            if self._resize_handle and self._selected_annotation and self._drag_start:
                # リサイズ中
                self._resize_annotation(x, y)
                self._update_display()
            elif self._drag_start and self._selected_annotation:
                # 移動中
                new_x1 = x - self._drag_offset[0]
                new_y1 = y - self._drag_offset[1]
                new_x2 = new_x1 + self._selected_annotation.bbox.width
                new_y2 = new_y1 + self._selected_annotation.bbox.height

                self._selected_annotation.bbox = BoundingBox(
                    new_x1, new_y1, new_x2, new_y2
                ).clamp(self._reader.width, self._reader.height)
                self._update_display()

    def mouseReleaseEvent(self, event) -> None:
        """マウスボタン解放"""
        if event.button() != Qt.LeftButton:
            return

        if self._edit_mode == self.MODE_DRAW and self._is_drawing:
            # 描画完了
            if self._drawing_rect and self._drawing_rect.area > 100:
                normalized = self._drawing_rect.normalize()
                if self._reader:
                    normalized = normalized.clamp(self._reader.width, self._reader.height)

                ann = Annotation(
                    frame=self._current_frame_number,
                    bbox=normalized,
                    track_id=self._annotation_store.new_track_id(),
                    is_manual=True,
                )
                self._annotation_store.add(ann)
                self.annotation_added.emit(ann)
                self.annotations_changed.emit()

            self._drawing_rect = None
            self._is_drawing = False
            self._mouse_start = None
            self._update_display()

        elif self._edit_mode == self.MODE_EDIT:
            if self._resize_handle or self._drag_start:
                # 編集完了、Undoスタックに保存
                self.annotations_changed.emit()

            self._resize_handle = None
            self._drag_start = None

    def _resize_annotation(self, x: int, y: int) -> None:
        """アノテーションをリサイズ"""
        if not self._selected_annotation or not self._resize_handle:
            return

        bbox = self._selected_annotation.bbox
        new_x1, new_y1, new_x2, new_y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        if "n" in self._resize_handle:
            new_y1 = y
        if "s" in self._resize_handle:
            new_y2 = y
        if "w" in self._resize_handle:
            new_x1 = x
        if "e" in self._resize_handle:
            new_x2 = x

        new_bbox = BoundingBox(new_x1, new_y1, new_x2, new_y2).normalize()
        if self._reader:
            new_bbox = new_bbox.clamp(self._reader.width, self._reader.height)

        if new_bbox.width > 10 and new_bbox.height > 10:
            self._selected_annotation.bbox = new_bbox

    def _update_cursor(self, x: int, y: int) -> None:
        """カーソル形状を更新"""
        if self._selected_annotation:
            handle = self._selected_annotation.bbox.get_resize_handle(x, y, int(10 / self._scale))
            if handle:
                cursors = {
                    "nw": Qt.SizeFDiagCursor,
                    "se": Qt.SizeFDiagCursor,
                    "ne": Qt.SizeBDiagCursor,
                    "sw": Qt.SizeBDiagCursor,
                    "n": Qt.SizeVerCursor,
                    "s": Qt.SizeVerCursor,
                    "e": Qt.SizeHorCursor,
                    "w": Qt.SizeHorCursor,
                }
                self.setCursor(cursors.get(handle, Qt.ArrowCursor))
                return

            if self._selected_annotation.bbox.contains_point(x, y):
                self.setCursor(Qt.SizeAllCursor)
                return

        self.setCursor(Qt.CrossCursor if self._edit_mode == self.MODE_DRAW else Qt.ArrowCursor)

    def keyPressEvent(self, event) -> None:
        """キー入力"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self._selected_annotation:
                self._annotation_store.remove_annotation(self._selected_annotation)
                self._selected_annotation = None
                self._selected_index = -1
                self.annotations_changed.emit()
                self._update_display()
        else:
            super().keyPressEvent(event)

    def delete_selected_annotation(self) -> bool:
        """選択中のアノテーションを削除"""
        if self._selected_annotation:
            self._annotation_store.remove_annotation(self._selected_annotation)
            self._selected_annotation = None
            self._selected_index = -1
            self.annotations_changed.emit()
            self._update_display()
            return True
        return False

    def copy_to_next_frame(self) -> bool:
        """選択中のアノテーションを次のフレームにコピー"""
        if not self._selected_annotation or not self._reader:
            return False

        next_frame = self._current_frame_number + 1
        if next_frame >= self._reader.frame_count:
            return False

        # 同じtrack_idのアノテーションが次のフレームに既にあるか確認
        existing = None
        for ann in self._annotation_store.get_frame_annotations(next_frame):
            if ann.track_id == self._selected_annotation.track_id:
                existing = ann
                break

        if existing:
            # 既存のものを更新
            existing.bbox = BoundingBox(
                self._selected_annotation.bbox.x1,
                self._selected_annotation.bbox.y1,
                self._selected_annotation.bbox.x2,
                self._selected_annotation.bbox.y2,
            )
        else:
            # 新規作成
            new_ann = Annotation(
                frame=next_frame,
                bbox=BoundingBox(
                    self._selected_annotation.bbox.x1,
                    self._selected_annotation.bbox.y1,
                    self._selected_annotation.bbox.x2,
                    self._selected_annotation.bbox.y2,
                ),
                track_id=self._selected_annotation.track_id,
                is_manual=True,
            )
            self._annotation_store.add(new_ann)

        self.annotations_changed.emit()
        # 次のフレームに移動
        self.seek(next_frame)
        return True

    def play(self) -> None:
        """再生開始"""
        if self._reader is None or self._is_playing:
            return

        self._is_playing = True
        interval = int(1000 / self._reader.fps) if self._reader.fps > 0 else 33
        self._playback_timer.start(interval)
        self.playback_state_changed.emit(True)

    def pause(self) -> None:
        """一時停止"""
        self._is_playing = False
        self._playback_timer.stop()
        self.playback_state_changed.emit(False)

    def stop(self) -> None:
        """停止"""
        self.pause()
        self._show_frame(0)

    def toggle_playback(self) -> None:
        """再生/一時停止を切り替え"""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def _on_playback_tick(self) -> None:
        """再生タイマーのコールバック"""
        if self._reader is None:
            self.pause()
            return

        next_frame = self._current_frame_number + 1
        if next_frame >= self._reader.frame_count:
            self.pause()
            return

        self._show_frame(next_frame)

    def seek(self, frame_number: int) -> None:
        """指定フレームにシーク"""
        if self._reader is None:
            return

        frame_number = max(0, min(frame_number, self._reader.frame_count - 1))

        # 自動補間: 選択中のアノテーションがあり、フレームをスキップする場合
        if (self._auto_interpolate and
            self._selected_annotation and
            self._selected_annotation.track_id is not None and
            abs(frame_number - self._current_frame_number) > 1):

            self._auto_interpolate_on_seek(
                self._current_frame_number,
                frame_number,
                self._selected_annotation.track_id
            )

        self._show_frame(frame_number)

    def step_forward(self, frames: int = 1) -> None:
        """指定フレーム数進める"""
        self.seek(self._current_frame_number + frames)

    def step_backward(self, frames: int = 1) -> None:
        """指定フレーム数戻る"""
        self.seek(self._current_frame_number - frames)

    def _auto_interpolate_on_seek(
        self, from_frame: int, to_frame: int, track_id: int
    ) -> None:
        """フレーム移動時の自動補間"""
        # 移動先フレームに同じtrack_idのアノテーションがあるか確認
        existing_ann = None
        for ann in self._annotation_store.get_frame_annotations(to_frame):
            if ann.track_id == track_id:
                existing_ann = ann
                break

        # なければ、移動元のアノテーションをコピー
        if existing_ann is None:
            source_ann = None
            for ann in self._annotation_store.get_frame_annotations(from_frame):
                if ann.track_id == track_id:
                    source_ann = ann
                    break

            if source_ann:
                new_ann = Annotation(
                    frame=to_frame,
                    bbox=BoundingBox(
                        source_ann.bbox.x1,
                        source_ann.bbox.y1,
                        source_ann.bbox.x2,
                        source_ann.bbox.y2,
                    ),
                    track_id=track_id,
                    is_manual=True,
                )
                self._annotation_store.add(new_ann, save_undo=False)

        # 2つのフレーム間を補間
        start_frame = min(from_frame, to_frame)
        end_frame = max(from_frame, to_frame)
        self._annotation_store.interpolate_frames(
            track_id, start_frame, end_frame, save_undo=False
        )
        self.annotations_changed.emit()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_frame_number(self) -> int:
        return self._current_frame_number

    @property
    def current_frame(self) -> np.ndarray | None:
        return self._current_frame

    @property
    def frame_count(self) -> int:
        return self._reader.frame_count if self._reader else 0

    @property
    def fps(self) -> float:
        return self._reader.fps if self._reader else 0.0

    @property
    def video_width(self) -> int:
        return self._reader.width if self._reader else 0

    @property
    def video_height(self) -> int:
        return self._reader.height if self._reader else 0

    @property
    def selected_annotation(self) -> Annotation | None:
        return self._selected_annotation

    def resizeEvent(self, event) -> None:
        """リサイズ時に再描画"""
        super().resizeEvent(event)
        if self._current_frame is not None:
            self._update_display()

    def release(self) -> None:
        """リソースを解放"""
        self.stop()
        if self._reader is not None:
            self._reader.release()
            self._reader = None

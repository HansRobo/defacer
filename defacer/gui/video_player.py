"""動画プレーヤーウィジェット"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QCursor,
    QFont,
)
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget, QHBoxLayout, QToolButton

import numpy as np

from defacer.video.reader import VideoReader
from defacer.gui.annotation import BoundingBox, Annotation, AnnotationStore


class AnnotationToolbar(QWidget):
    """アノテーション選択時に表示されるフローティングツールバー"""

    delete_clicked = pyqtSignal()
    delete_track_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # レイアウト設定
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 削除ボタン
        self.delete_btn = QToolButton(self)
        self.delete_btn.setText("🗑")
        self.delete_btn.setToolTip("このアノテーションを削除")
        self.delete_btn.clicked.connect(self.delete_clicked.emit)
        layout.addWidget(self.delete_btn)

        # トラック削除ボタン
        self.delete_track_btn = QToolButton(self)
        self.delete_track_btn.setText("⊗")
        self.delete_track_btn.setToolTip("このトラックの全アノテーションを削除")
        self.delete_track_btn.clicked.connect(self.delete_track_clicked.emit)
        layout.addWidget(self.delete_track_btn)

        # ダークテーマスタイル
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 220);
                border: 1px solid rgba(100, 100, 100, 200);
                border-radius: 4px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                color: white;
                font-size: 16px;
                padding: 4px 8px;
                min-width: 24px;
                min-height: 24px;
            }
            QToolButton:hover {
                background-color: rgba(80, 80, 80, 150);
                border-radius: 2px;
            }
            QToolButton:pressed {
                background-color: rgba(100, 100, 100, 150);
            }
        """)

        self.hide()


class VideoPlayerWidget(QLabel):
    """動画を表示するウィジェット"""

    frame_changed = pyqtSignal(int)  # フレーム番号が変わった時
    playback_state_changed = pyqtSignal(bool)  # 再生状態が変わった時
    annotation_added = pyqtSignal(object)  # アノテーションが追加された時
    annotation_selected = pyqtSignal(object)  # アノテーションが選択された時
    annotations_changed = pyqtSignal(bool)  # アノテーションが変更された時 (引数: トラック構造変更か)

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
        self._pending_draw_start: tuple[int, int] | None = None  # 描画開始候補

        # 編集中の状態
        self._edit_mode = self.MODE_DRAW
        self._resize_handle: str | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_offset: tuple[int, int] = (0, 0)
        self._is_nudging: bool = False  # キーボード微調整中フラグ

        # 画像のスケールとオフセット（座標変換用）
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        # フローティングツールバー
        self._toolbar = AnnotationToolbar(self)
        self._toolbar.delete_clicked.connect(self._delete_current_annotation)
        self._toolbar.delete_track_clicked.connect(self._delete_current_track)

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
        """編集モードを設定（モードレス化により、外部呼び出しは無視）"""
        # モードレス化: 常に統合モードで動作するため、外部からの呼び出しは無視
        pass

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
        self._hide_toolbar()

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

        # ツールバー位置を更新
        if self._toolbar.isVisible():
            self._update_toolbar_position()

    def _get_track_color(self, track_id: int | None) -> tuple[int, int, int]:
        """トラックIDに基づいて色を生成（HSVベース）"""
        if track_id is None:
            return (200, 200, 200)  # グレー

        # トラックIDを使って色相を分散
        # 黄金角（137.5度）を使って視覚的に区別しやすい色を生成
        hue = (track_id * 137.5) % 360
        color = QColor.fromHsvF(hue / 360, 0.8, 0.95)
        return (color.red(), color.green(), color.blue())

    def _draw_annotation(self, painter: QPainter, ann: Annotation, is_selected: bool) -> None:
        """アノテーションを描画"""
        if is_selected:
            # 選択時は明るいシアン
            r, g, b = 0, 200, 255
            pen = QPen(QColor(r, g, b), 3)
            brush = QBrush(QColor(r, g, b, 40))
        else:
            # トラックIDに基づいて色を決定
            r, g, b = self._get_track_color(ann.track_id)
            pen = QPen(QColor(r, g, b), 2)
            brush = QBrush(QColor(r, g, b, 30))

        painter.setPen(pen)
        painter.setBrush(brush)
        self._draw_bbox(painter, ann.bbox)

        # トラックIDを表示
        if ann.track_id is not None:
            self._draw_track_label(painter, ann.bbox, ann.track_id, QColor(r, g, b))

        # 選択時はリサイズハンドルを描画（モードレス: 常に表示）
        if is_selected:
            self._draw_resize_handles(painter, ann.bbox)

    def _draw_track_label(self, painter: QPainter, bbox: BoundingBox, track_id: int, color: QColor) -> None:
        """トラックIDラベルを描画"""
        x1 = int(bbox.x1 * self._scale)
        y1 = int(bbox.y1 * self._scale)

        # ラベルテキスト
        label_text = f"#{track_id}"

        # フォント設定
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)

        # テキストサイズを取得
        text_rect = painter.fontMetrics().boundingRect(label_text)
        padding = 4
        label_width = text_rect.width() + padding * 2
        label_height = text_rect.height() + padding * 2

        # ラベル背景を描画（バウンディングボックスの左上）
        label_x = x1
        label_y = y1 - label_height - 2

        # 画面外に出る場合は内側に表示
        if label_y < 0:
            label_y = y1 + 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(label_x, label_y, label_width, label_height)

        # テキストを描画
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(
            label_x + padding,
            label_y + padding + text_rect.height() - painter.fontMetrics().descent(),
            label_text
        )

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

    def contextMenuEvent(self, event) -> None:
        """右クリックメニューを表示"""
        from PyQt5.QtWidgets import QMenu

        coords = self._widget_to_frame_coords(event.x(), event.y())
        if coords is None or self._edit_mode == self.MODE_VIEW:
            return

        x, y = coords
        result = self._annotation_store.get_annotation_at_point(
            self._current_frame_number, x, y
        )

        if result is None:
            return

        ann, idx = result

        # メニュー構築
        menu = QMenu(self)

        # トラックIDがある場合のみ統合メニューを表示
        if ann.track_id is not None:
            merge_action = menu.addAction(f"トラック {ann.track_id} を別のトラックに統合...")
            merge_action.triggered.connect(lambda: self._show_merge_dialog(ann))

            menu.addSeparator()

        # 削除メニュー
        delete_action = menu.addAction("このアノテーションを削除")
        delete_action.triggered.connect(lambda: self._delete_annotation_at_point(ann))

        menu.exec_(event.globalPos())

    def mousePressEvent(self, event) -> None:
        """マウスボタン押下（モードレス統合版）"""
        if event.button() != Qt.LeftButton:
            return

        coords = self._widget_to_frame_coords(event.x(), event.y())
        if coords is None:
            return

        x, y = coords

        # 1. リサイズハンドルチェック（最優先）
        if self._selected_annotation:
            handle = self._selected_annotation.bbox.get_resize_handle(
                x, y, int(10 / self._scale)
            )
            if handle:
                self._resize_handle = handle
                self._drag_start = coords
                return

        # 2. アノテーション上をクリック → 選択＆移動準備
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
            self._show_toolbar()
            return

        # 3. 空白領域をクリック → 選択解除 OR 描画準備
        self._selected_annotation = None
        self._selected_index = -1
        self.annotation_selected.emit(None)
        self._hide_toolbar()
        self._update_display()

        # 描画開始候補として座標を保存（移動後に判定）
        self._pending_draw_start = coords

    def mouseMoveEvent(self, event) -> None:
        """マウス移動（モードレス統合版）"""
        coords = self._widget_to_frame_coords(event.x(), event.y())

        # カーソル形状の更新
        if coords:
            self._update_cursor(coords[0], coords[1])

        if coords is None:
            return

        x, y = coords

        # 描画開始候補がある場合、5px以上移動したら描画開始
        if self._pending_draw_start and not self._is_drawing:
            dx = abs(x - self._pending_draw_start[0])
            dy = abs(y - self._pending_draw_start[1])
            if dx > 5 or dy > 5:
                self._is_drawing = True
                self._mouse_start = self._pending_draw_start
                self._pending_draw_start = None
                self._drawing_rect = BoundingBox(
                    self._mouse_start[0], self._mouse_start[1], x, y
                ).normalize()
                self._update_display()
                return

        # リサイズ中
        if self._resize_handle and self._selected_annotation and self._drag_start:
            self._resize_annotation(x, y)
            self._update_display()
            return

        # 移動中
        if self._drag_start and self._selected_annotation:
            new_x1 = x - self._drag_offset[0]
            new_y1 = y - self._drag_offset[1]
            new_x2 = new_x1 + self._selected_annotation.bbox.width
            new_y2 = new_y1 + self._selected_annotation.bbox.height

            self._selected_annotation.bbox = BoundingBox(
                new_x1, new_y1, new_x2, new_y2
            ).clamp(self._reader.width, self._reader.height)
            self._update_display()
            return

        # 描画中
        if self._is_drawing and self._mouse_start:
            self._drawing_rect = BoundingBox(
                self._mouse_start[0], self._mouse_start[1], x, y
            ).normalize()
            self._update_display()

    def mouseReleaseEvent(self, event) -> None:
        """マウスボタン解放（モードレス統合版）"""
        if event.button() != Qt.LeftButton:
            return

        # 描画完了処理（閾値強化: area > 400 かつ width > 15 and height > 15）
        if self._is_drawing:
            if self._drawing_rect:
                # まず正規化してから閾値チェック（どの方向のドラッグでも対応）
                normalized = self._drawing_rect.normalize()
                if (normalized.area > 400 and
                    normalized.width > 15 and
                    normalized.height > 15):
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
                    self.annotations_changed.emit(True)  # 構造変更

            self._drawing_rect = None
            self._is_drawing = False
            self._mouse_start = None
            self._update_display()

        # 編集完了処理
        if self._resize_handle or self._drag_start:
            # 編集完了（位置変更のみ、トラック構造は不変）
            self.annotations_changed.emit(False)
            self._update_display()

        # 状態リセット
        self._pending_draw_start = None
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
        """カーソル形状を更新（モードレス統合版）"""
        # 選択中のアノテーションがある場合
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

        # アノテーション上ならポインター、空白領域なら十字
        result = self._annotation_store.get_annotation_at_point(
            self._current_frame_number, x, y
        )
        if result:
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def keyPressEvent(self, event) -> None:
        """キー入力"""
        key = event.key()
        modifiers = event.modifiers()

        # 削除キー
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._selected_annotation:
                self._annotation_store.remove_annotation(self._selected_annotation)
                self._selected_annotation = None
                self._selected_index = -1
                self._hide_toolbar()
                self.annotations_changed.emit(True)  # 構造変更
                self._update_display()
            return

        # 矢印キーによる微調整（選択中のみ）
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            if self._selected_annotation:
                # 最初のキー押下時のみフラグをセット
                if not event.isAutoRepeat():
                    self._is_nudging = True
                self._nudge_annotation(key, modifiers)
                return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """キー解放"""
        # オートリピート無視
        if event.isAutoRepeat():
            return

        key = event.key()

        # 矢印キー解放時に変更を確定
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            if self._is_nudging:
                self._is_nudging = False
                self.annotations_changed.emit(False)  # 位置変更のみ
            return

        super().keyReleaseEvent(event)

    def _nudge_annotation(self, key: int, modifiers) -> None:
        """矢印キーでアノテーションを微調整（表示更新のみ、変更通知はkeyReleaseで）"""
        if not self._selected_annotation or not self._reader:
            return

        bbox = self._selected_annotation.bbox
        is_shift = modifiers & Qt.ShiftModifier
        is_ctrl = modifiers & Qt.ControlModifier

        # 移動量
        step = 10 if is_ctrl else 1

        if is_shift:
            # Shift: 右下角をリサイズ
            new_x1, new_y1, new_x2, new_y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

            if key == Qt.Key_Up:
                new_y2 -= step
            elif key == Qt.Key_Down:
                new_y2 += step
            elif key == Qt.Key_Left:
                new_x2 -= step
            elif key == Qt.Key_Right:
                new_x2 += step

            new_bbox = BoundingBox(new_x1, new_y1, new_x2, new_y2).normalize()
            if new_bbox.width > 10 and new_bbox.height > 10:
                self._selected_annotation.bbox = new_bbox.clamp(
                    self._reader.width, self._reader.height
                )
        else:
            # 通常: 移動
            dx, dy = 0, 0

            if key == Qt.Key_Up:
                dy = -step
            elif key == Qt.Key_Down:
                dy = step
            elif key == Qt.Key_Left:
                dx = -step
            elif key == Qt.Key_Right:
                dx = step

            new_bbox = BoundingBox(
                bbox.x1 + dx, bbox.y1 + dy, bbox.x2 + dx, bbox.y2 + dy
            ).clamp(self._reader.width, self._reader.height)

            self._selected_annotation.bbox = new_bbox

        # 表示更新のみ（変更通知はキーリリース時）
        self._update_display()

    def _show_toolbar(self) -> None:
        """ツールバーを表示（モードレス: 常に表示可能）"""
        if not self._selected_annotation:
            return

        self._update_toolbar_position()
        self._toolbar.show()

    def _hide_toolbar(self) -> None:
        """ツールバーを非表示"""
        self._toolbar.hide()

    def _update_toolbar_position(self) -> None:
        """ツールバーの位置を更新"""
        if not self._selected_annotation:
            return

        bbox = self._selected_annotation.bbox

        # アノテーションの右上（スケール変換 + オフセット）
        x = int(bbox.x2 * self._scale) + self._offset_x + 8
        y = int(bbox.y1 * self._scale) + self._offset_y

        toolbar_width = self._toolbar.sizeHint().width()
        toolbar_height = self._toolbar.sizeHint().height()

        # 画面端調整（右端）
        if x + toolbar_width > self.width():
            x = int(bbox.x1 * self._scale) + self._offset_x - toolbar_width - 8

        # 画面端調整（上端）
        if y < 0:
            y = int(bbox.y2 * self._scale) + self._offset_y + 8

        # 画面端調整（下端）
        if y + toolbar_height > self.height():
            y = self.height() - toolbar_height

        self._toolbar.move(x, y)

    def _delete_current_annotation(self) -> None:
        """選択中のアノテーションを削除"""
        if self._selected_annotation:
            self._annotation_store.remove_annotation(self._selected_annotation)
            self._selected_annotation = None
            self._selected_index = -1
            self._hide_toolbar()
            self.annotations_changed.emit(True)  # 構造変更
            self._update_display()

    def _delete_current_track(self) -> None:
        """選択中のアノテーションのトラック全体を削除"""
        from PyQt5.QtWidgets import QMessageBox

        if not self._selected_annotation or self._selected_annotation.track_id is None:
            return

        track_id = self._selected_annotation.track_id
        track_info = self._annotation_store.get_track_info(track_id)

        if not track_info.get("exists"):
            return

        # 確認ダイアログ
        reply = QMessageBox.question(
            self,
            "トラック削除の確認",
            f"トラック #{track_id} の全アノテーション（{track_info['annotation_count']}個）を削除しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            count = self._annotation_store.remove_track(track_id)
            self._selected_annotation = None
            self._selected_index = -1
            self._hide_toolbar()
            self.annotations_changed.emit(True)  # 構造変更
            self._update_display()

            QMessageBox.information(
                self,
                "トラック削除",
                f"トラック #{track_id} の全アノテーション（{count}個）を削除しました。",
            )

    def delete_selected_annotation(self) -> bool:
        """選択中のアノテーションを削除"""
        if self._selected_annotation:
            self._annotation_store.remove_annotation(self._selected_annotation)
            self._selected_annotation = None
            self._selected_index = -1
            self._hide_toolbar()
            self.annotations_changed.emit(True)  # 構造変更
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

        self.annotations_changed.emit(True)  # 構造変更
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
        self.annotations_changed.emit(True)  # 構造変更

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
    def video_path(self) -> str | None:
        """動画ファイルパスを取得"""
        return str(self._reader.path) if self._reader else None

    @property
    def selected_annotation(self) -> Annotation | None:
        return self._selected_annotation

    def resizeEvent(self, event) -> None:
        """リサイズ時に再描画"""
        super().resizeEvent(event)
        if self._current_frame is not None:
            self._update_display()
        if self._toolbar.isVisible():
            self._update_toolbar_position()

    def _show_merge_dialog(self, annotation: Annotation) -> None:
        """トラック統合ダイアログを表示"""
        from PyQt5.QtWidgets import QInputDialog, QMessageBox

        source_track_id = annotation.track_id
        if source_track_id is None:
            return

        # 利用可能なトラックIDを取得
        available_tracks = sorted(self._annotation_store.get_all_track_ids())
        available_tracks = [t for t in available_tracks if t != source_track_id]

        if not available_tracks:
            QMessageBox.warning(
                self,
                "トラック統合",
                "統合先のトラックが存在しません。",
            )
            return

        # トラック選択ダイアログ
        items = [f"トラック {tid}" for tid in available_tracks]
        item, ok = QInputDialog.getItem(
            self,
            "トラック統合",
            f"トラック {source_track_id} の統合先を選択してください:",
            items,
            0,
            False,
        )

        if ok and item:
            target_track_id = available_tracks[items.index(item)]
            self._merge_tracks(source_track_id, target_track_id)

    def _merge_tracks(self, source_track_id: int, target_track_id: int) -> None:
        """トラックを統合"""
        from PyQt5.QtWidgets import QMessageBox

        # 衝突チェック（同じフレームに両方のトラックが存在するか）
        conflicts = self._check_track_conflicts(source_track_id, target_track_id)

        if conflicts:
            reply = QMessageBox.question(
                self,
                "トラック統合の確認",
                f"以下のフレームで統合元と統合先が重複しています:\n"
                f"{', '.join(map(str, conflicts[:10]))}"
                f"{'...' if len(conflicts) > 10 else ''}\n\n"
                f"統合元のアノテーション（トラック {source_track_id}）を削除して続行しますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

            # 重複フレームの統合元アノテーションを削除
            self._remove_track_from_frames(source_track_id, conflicts)

        # トラック統合を実行
        count = self._annotation_store.merge_tracks(source_track_id, target_track_id)

        self.annotations_changed.emit(True)  # 構造変更
        self._update_display()

        # 完了メッセージ
        QMessageBox.information(
            self,
            "トラック統合",
            f"トラック {source_track_id} を {target_track_id} に統合しました\n"
            f"({count}個のアノテーションを更新)",
        )

    def _check_track_conflicts(
        self, source_track_id: int, target_track_id: int
    ) -> list[int]:
        """2つのトラックが同じフレームに存在するフレームのリストを返す"""
        source_frames = set()
        target_frames = set()

        for ann in self._annotation_store:
            if ann.track_id == source_track_id:
                source_frames.add(ann.frame)
            elif ann.track_id == target_track_id:
                target_frames.add(ann.frame)

        conflicts = sorted(source_frames & target_frames)
        return conflicts

    def _remove_track_from_frames(
        self, track_id: int, frames: list[int]
    ) -> None:
        """指定トラックIDの指定フレームのアノテーションを削除"""
        for frame in frames:
            anns = self._annotation_store.get_frame_annotations(frame)
            to_remove = [ann for ann in anns if ann.track_id == track_id]
            for ann in to_remove:
                self._annotation_store.remove_annotation(ann, save_undo=False)

    def _delete_annotation_at_point(self, annotation: Annotation) -> None:
        """指定のアノテーションを削除"""
        self._annotation_store.remove_annotation(annotation)
        self.annotations_changed.emit(True)  # 構造変更
        self._update_display()

    def release(self) -> None:
        """リソースを解放"""
        self.stop()
        if self._reader is not None:
            self._reader.release()
            self._reader = None

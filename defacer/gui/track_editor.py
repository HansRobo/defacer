"""トラック編集専用ダイアログ"""

from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize, QThread
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QScrollArea,
    QLabel,
    QSplitter,
    QMessageBox,
    QSlider,
    QMenu,
    QAction,
    QFrame,
    QProgressBar,
    QGroupBox,
)

from defacer.gui.annotation import AnnotationStore
from defacer.gui.thumbnail_cache import ThumbnailCache
from defacer.tracking.merge_suggestion import (
    compute_merge_suggestions,
    collect_track_infos,
    MergeSuggestion,
)
from defacer.tracking.interpolation import interpolate_all_tracks


class ThumbnailLoaderWorker(QThread):
    """サムネイル生成ワーカー"""

    thumbnail_loaded = pyqtSignal(int, QPixmap)  # track_id, pixmap
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, cache, track_ids, store, parent=None):
        super().__init__(parent)
        self._cache = cache
        self._track_ids = track_ids
        self._store = store
        self._cancelled = False

    def run(self):
        """バックグラウンド実行"""
        total = len(self._track_ids)
        for i, track_id in enumerate(self._track_ids):
            if self._cancelled:
                break

            # サムネイルを生成
            thumbnail = self._cache.get_track_thumbnail(track_id, self._store)
            if thumbnail is not None:
                self.thumbnail_loaded.emit(track_id, thumbnail)

            self.progress.emit(i + 1, total)

    def cancel(self):
        """キャンセル"""
        self._cancelled = True


class TrackTimelineWidget(QWidget):
    """トラックタイムラインウィジェット"""

    track_selected = pyqtSignal(int)  # track_id
    tracks_selected = pyqtSignal(list)  # [track_id, ...]
    tracks_merge_requested = pyqtSignal(list)  # [track_id, ...]
    frame_changed = pyqtSignal(int)  # frame_number

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)

        self._store: AnnotationStore | None = None
        self._track_infos = []
        self._selected_track_ids: set[int] = set()  # 複数選択対応
        self._last_selected_track_id: int | None = None  # 範囲選択用
        self._current_frame = 0
        self._total_frames = 1000
        self._thumbnail_cache: ThumbnailCache | None = None
        self._suggestions: list[MergeSuggestion] = []
        self._show_all_tracks = False  # 全トラック表示フラグ
        self._filtered_track_ids: set[int] = set()  # フィルタ対象トラックID

        # ドラッグ状態
        self._dragging_track_id: int | None = None
        self._drag_start_pos = None

        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    def set_store(self, store: AnnotationStore, total_frames: int) -> None:
        """ストアを設定"""
        self._store = store
        self._total_frames = total_frames
        self._track_infos = collect_track_infos(store)
        self.update()

    def set_thumbnail_cache(self, cache: ThumbnailCache) -> None:
        """サムネイルキャッシュを設定"""
        self._thumbnail_cache = cache
        self.update()

    def set_suggestions(self, suggestions: list[MergeSuggestion]) -> None:
        """統合候補を設定（接続線描画用）"""
        self._suggestions = list(suggestions)  # コピーして保持
        # 統合候補に含まれるトラックIDを抽出
        self._filtered_track_ids.clear()
        for suggestion in suggestions:
            for track_id in suggestion.track_ids:
                self._filtered_track_ids.add(track_id)

        # 存在しないトラックIDの選択を解除
        valid_track_ids = {info.track_id for info in self._track_infos}
        self._selected_track_ids = self._selected_track_ids & valid_track_ids
        if self._last_selected_track_id not in valid_track_ids:
            self._last_selected_track_id = None

        self.update()

    def set_show_all_tracks(self, show_all: bool) -> None:
        """全トラック表示モードを設定"""
        self._show_all_tracks = show_all
        self.update()

    def _get_display_track_infos(self):
        """表示するトラック情報を取得"""
        if self._show_all_tracks:
            return self._track_infos
        else:
            # 統合候補に含まれるトラックのみ表示
            return [info for info in self._track_infos if info.track_id in self._filtered_track_ids]

    def refresh(self) -> None:
        """トラック情報を再計算して再描画"""
        if self._store is not None:
            self._track_infos = collect_track_infos(self._store)

            # 存在しないトラックIDの選択を解除
            valid_track_ids = {info.track_id for info in self._track_infos}
            self._selected_track_ids = self._selected_track_ids & valid_track_ids
            if self._last_selected_track_id not in valid_track_ids:
                self._last_selected_track_id = None

            self.update()

    def set_current_frame(self, frame: int) -> None:
        """現在フレームを設定"""
        self._current_frame = frame
        self.update()

    def set_selected_track(self, track_id: int | None) -> None:
        """選択トラックを設定（単一選択）"""
        self._selected_track_ids.clear()
        if track_id is not None:
            self._selected_track_ids.add(track_id)
            self._last_selected_track_id = track_id
        self.update()

    def get_selected_tracks(self) -> list[int]:
        """選択中のトラックIDリストを取得"""
        return sorted(self._selected_track_ids)

    def paintEvent(self, event):
        """描画イベント"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        display_infos = self._get_display_track_infos()
        if not display_infos:
            # トラックがない場合のメッセージ
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawText(self.rect(), Qt.AlignCenter, "統合候補がありません")
            return

        # 描画パラメータ（コンパクト版、サムネイルなし）
        margin_left = 60  # サムネイルなしで左マージン削減
        margin_right = 20
        margin_top = 30
        margin_bottom = 20
        row_height = 20  # さらにコンパクト化
        row_spacing = 1

        timeline_width = self.width() - margin_left - margin_right
        timeline_height = len(display_infos) * (row_height + row_spacing)

        # タイムラインの高さを調整
        self.setMinimumHeight(timeline_height + margin_top + margin_bottom)

        # タイムスケールを描画
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Arial", 8)
        painter.setFont(font)

        # フレーム目盛り
        frame_interval = max(1, self._total_frames // 10)
        for i in range(0, self._total_frames + 1, frame_interval):
            x = margin_left + int(i / self._total_frames * timeline_width)
            painter.drawLine(x, margin_top - 8, x, margin_top - 3)
            painter.drawText(x - 20, margin_top - 20, 40, 12, Qt.AlignCenter, str(i))

        # 統合候補の接続線を描画（トラックバーより前に）
        self._draw_suggestion_connections(painter, margin_left, margin_top, row_height, row_spacing, timeline_width, display_infos)

        # 現在フレームのインジケータ
        current_x = margin_left + int(self._current_frame / self._total_frames * timeline_width)
        painter.setPen(QPen(QColor(255, 165, 0), 2))
        painter.drawLine(current_x, margin_top, current_x, margin_top + timeline_height)

        # 各トラックを描画
        for idx, track_info in enumerate(display_infos):
            y = margin_top + idx * (row_height + row_spacing)

            # トラックID（サムネイルなし、テキストのみ）
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(5, y, 80, row_height, Qt.AlignVCenter | Qt.AlignLeft, f"#{track_info.track_id}")
            painter.setFont(font)

            # トラックバー
            start_x = margin_left + int(track_info.frame_min / self._total_frames * timeline_width)
            end_x = margin_left + int(track_info.frame_max / self._total_frames * timeline_width)
            bar_width = max(2, end_x - start_x)

            # 選択状態に応じて色を変更
            if track_info.track_id in self._selected_track_ids:
                bar_color = QColor(0, 120, 212)  # 選択中（青）
            else:
                bar_color = QColor(80, 80, 120)  # 通常（グレー）

            painter.fillRect(start_x, y + 4, bar_width, row_height - 8, bar_color)

            # 枠線（選択中は太く）
            if track_info.track_id in self._selected_track_ids:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawRect(start_x, y + 4, bar_width, row_height - 8)

    def _draw_suggestion_connections(self, painter, margin_left, margin_top, row_height, row_spacing, timeline_width, display_infos):
        """統合候補の接続線を描画"""
        if not self._suggestions:
            return

        # 表示中のトラックIDからインデックスへのマップを作成
        track_id_to_idx = {info.track_id: idx for idx, info in enumerate(display_infos)}
        track_id_to_info = {info.track_id: info for info in display_infos}

        for suggestion in self._suggestions:
            # 統合候補のトラック間に点線を描画
            pen = QPen(QColor(76, 175, 80, 180), 1, Qt.DashLine)  # 緑の点線
            painter.setPen(pen)

            for i in range(len(suggestion.track_ids) - 1):
                track_id_1 = suggestion.track_ids[i]
                track_id_2 = suggestion.track_ids[i + 1]

                if track_id_1 not in track_id_to_idx or track_id_2 not in track_id_to_idx:
                    continue

                idx_1 = track_id_to_idx[track_id_1]
                idx_2 = track_id_to_idx[track_id_2]

                y1 = margin_top + idx_1 * (row_height + row_spacing) + row_height // 2
                y2 = margin_top + idx_2 * (row_height + row_spacing) + row_height // 2

                info_1 = track_id_to_info[track_id_1]
                info_2 = track_id_to_info[track_id_2]

                x1 = margin_left + int(info_1.frame_max / self._total_frames * timeline_width)
                x2 = margin_left + int(info_2.frame_min / self._total_frames * timeline_width)

                # 曲線で接続
                from PyQt5.QtGui import QPainterPath
                path = QPainterPath()
                path.moveTo(x1, y1)
                ctrl_x = (x1 + x2) // 2
                path.cubicTo(ctrl_x, y1, ctrl_x, y2, x2, y2)
                painter.drawPath(path)

    def mousePressEvent(self, event):
        """マウスプレスイベント"""
        if event.button() == Qt.LeftButton:
            track_id = self._get_track_at_pos(event.pos())
            if track_id is not None:
                modifiers = event.modifiers()

                if modifiers & Qt.ControlModifier:
                    # Ctrl+クリック: トグル選択
                    if track_id in self._selected_track_ids:
                        self._selected_track_ids.remove(track_id)
                    else:
                        self._selected_track_ids.add(track_id)
                        self._last_selected_track_id = track_id
                elif modifiers & Qt.ShiftModifier:
                    # Shift+クリック: 範囲選択
                    if self._last_selected_track_id is not None:
                        self._select_range(self._last_selected_track_id, track_id)
                    else:
                        self._selected_track_ids = {track_id}
                        self._last_selected_track_id = track_id
                else:
                    # 通常クリック: 単一選択
                    self._selected_track_ids = {track_id}
                    self._last_selected_track_id = track_id
                    self._dragging_track_id = track_id
                    self._drag_start_pos = event.pos()

                self.update()
                self.tracks_selected.emit(self.get_selected_tracks())

    def mouseMoveEvent(self, event):
        """マウス移動イベント"""
        if self._dragging_track_id is not None:
            # ドラッグ中の視覚フィードバック（オプション）
            pass

    def mouseReleaseEvent(self, event):
        """マウスリリースイベント"""
        if event.button() == Qt.LeftButton and self._dragging_track_id is not None:
            target_track_id = self._get_track_at_pos(event.pos())

            if target_track_id is not None and target_track_id != self._dragging_track_id:
                # ドロップ先が別のトラックの場合、統合を要求（リスト形式）
                self.tracks_merge_requested.emit([self._dragging_track_id, target_track_id])

            self._dragging_track_id = None
            self._drag_start_pos = None

    def mouseDoubleClickEvent(self, event):
        """ダブルクリックイベント"""
        if event.button() == Qt.LeftButton:
            track_id = self._get_track_at_pos(event.pos())
            if track_id is not None:
                # そのトラックの最初のフレームにジャンプ
                for track_info in self._track_infos:
                    if track_info.track_id == track_id:
                        self.frame_changed.emit(track_info.frame_min)
                        break

    def contextMenuEvent(self, event):
        """右クリックメニュー"""
        if len(self._selected_track_ids) < 2:
            return

        menu = QMenu(self)

        merge_action = QAction(f"選択トラックを統合 ({len(self._selected_track_ids)}個)", self)
        merge_action.triggered.connect(self._merge_selected_tracks)
        menu.addAction(merge_action)

        menu.exec_(event.globalPos())

    def _merge_selected_tracks(self):
        """選択中のトラックを統合"""
        if len(self._selected_track_ids) >= 2:
            self.tracks_merge_requested.emit(self.get_selected_tracks())

    def _get_track_at_pos(self, pos) -> int | None:
        """指定位置のトラックIDを取得"""
        margin_left = 60
        margin_top = 30
        row_height = 20
        row_spacing = 1

        display_infos = self._get_display_track_infos()
        for idx, track_info in enumerate(display_infos):
            y = margin_top + idx * (row_height + row_spacing)
            if y <= pos.y() <= y + row_height:
                return track_info.track_id

        return None

    def _select_range(self, start_track_id: int, end_track_id: int) -> None:
        """範囲選択"""
        # トラックIDのインデックスを取得
        track_ids = [info.track_id for info in self._track_infos]
        try:
            start_idx = track_ids.index(start_track_id)
            end_idx = track_ids.index(end_track_id)
        except ValueError:
            return

        # 範囲を選択
        min_idx = min(start_idx, end_idx)
        max_idx = max(start_idx, end_idx)
        for i in range(min_idx, max_idx + 1):
            self._selected_track_ids.add(track_ids[i])


class MiniMapWidget(QWidget):
    """ミニマップウィジェット"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setMaximumSize(300, 200)

        self._store: AnnotationStore | None = None
        self._current_frame = 0
        self._video_width = 1920
        self._video_height = 1080

    def set_store(self, store: AnnotationStore, video_width: int, video_height: int) -> None:
        """ストアと動画サイズを設定"""
        self._store = store
        self._video_width = video_width
        self._video_height = video_height
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """現在フレームを設定"""
        self._current_frame = frame
        self.update()

    def paintEvent(self, event):
        """描画イベント"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._store is None:
            return

        # アスペクト比を保持してスケール計算
        aspect_ratio = self._video_width / self._video_height
        widget_aspect = self.width() / self.height()

        if aspect_ratio > widget_aspect:
            # 幅に合わせる
            scale = self.width() / self._video_width
            offset_x = 0
            offset_y = (self.height() - self._video_height * scale) / 2
        else:
            # 高さに合わせる
            scale = self.height() / self._video_height
            offset_x = (self.width() - self._video_width * scale) / 2
            offset_y = 0

        # フレーム境界を描画
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(int(offset_x), int(offset_y), int(self._video_width * scale), int(self._video_height * scale))

        # 現在フレームのアノテーションを描画
        annotations = self._store.get_frame_annotations(self._current_frame)

        for ann in annotations:
            bbox = ann.bbox

            x = int(offset_x + bbox.x1 * scale)
            y = int(offset_y + bbox.y1 * scale)
            w = int(bbox.width * scale)
            h = int(bbox.height * scale)

            # トラックIDに応じて色を変える
            if ann.track_id is not None:
                color = QColor.fromHsv((ann.track_id * 30) % 360, 200, 255, 150)
            else:
                color = QColor(255, 255, 255, 150)

            painter.fillRect(x, y, w, h, color)
            painter.setPen(QPen(color.darker(150), 1))
            painter.drawRect(x, y, w, h)


class TrackThumbnailLabel(QLabel):
    """サムネイル+トラックIDを表示（コンパクト版）"""

    def __init__(self, track_id: int, thumbnail: QPixmap = None, parent=None):
        super().__init__(parent)
        self._track_id = track_id
        self._thumbnail = thumbnail
        self.setMinimumSize(50, 60)
        self.setMaximumSize(50, 60)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555; font-size: 8pt;")
        self._update_display()

    def set_thumbnail(self, thumbnail: QPixmap) -> None:
        """サムネイルを設定"""
        self._thumbnail = thumbnail
        self._update_display()

    def _update_display(self) -> None:
        """表示を更新"""
        if self._thumbnail is not None:
            scaled = self._thumbnail.scaled(
                44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
        else:
            self.setText(f"#{self._track_id}")


class MergeSuggestionDetailDialog(QDialog):
    """統合候補の詳細ダイアログ"""

    merge_requested = pyqtSignal(list)  # [track_id, ...]

    def __init__(self, suggestion: MergeSuggestion, cache: ThumbnailCache, store: AnnotationStore,
                 video_width: int, video_height: int, parent=None):
        super().__init__(parent)
        self._suggestion = suggestion
        self._cache = cache
        self._store = store
        self._video_width = video_width
        self._video_height = video_height

        self.setWindowTitle("統合候補の詳細")
        self.setModal(True)
        self.setMinimumSize(600, 500)

        self._setup_ui()

    def _setup_ui(self):
        """UIを構築"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # タイトル
        title_label = QLabel(f"統合候補の詳細 ({len(self._suggestion.track_ids)}個のトラック)")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # スクロールエリア
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # サムネイル一覧
        thumbnail_group = QGroupBox("全トラックのサムネイル")
        thumbnail_layout = QVBoxLayout(thumbnail_group)

        # サムネイルグリッド
        from PyQt5.QtWidgets import QGridLayout
        thumbnail_grid = QGridLayout()
        thumbnail_grid.setSpacing(10)

        max_cols = 5
        for idx, track_id in enumerate(self._suggestion.track_ids):
            col = idx % max_cols
            row = idx // max_cols

            # サムネイル列
            thumb_container = QWidget()
            thumb_layout = QVBoxLayout(thumb_container)
            thumb_layout.setSpacing(5)
            thumb_layout.setAlignment(Qt.AlignCenter)

            # サムネイル
            thumb_label = TrackThumbnailLabel(track_id)
            thumb_label.setMinimumSize(64, 74)
            thumb_label.setMaximumSize(64, 74)

            # キャッシュから取得または生成
            thumbnail = self._cache.get_track_thumbnail(track_id, self._store)
            if thumbnail:
                thumb_label.set_thumbnail(thumbnail)

            thumb_layout.addWidget(thumb_label, alignment=Qt.AlignCenter)

            # トラックID
            id_label = QLabel(f"#{track_id}")
            id_label.setAlignment(Qt.AlignCenter)
            id_label.setStyleSheet("font-weight: bold;")
            thumb_layout.addWidget(id_label)

            # フレーム範囲
            track_info = next((info for info in collect_track_infos(self._store) if info.track_id == track_id), None)
            if track_info:
                frame_label = QLabel(f"{track_info.frame_min}-{track_info.frame_max}")
                frame_label.setAlignment(Qt.AlignCenter)
                frame_label.setStyleSheet("font-size: 9pt; color: #888;")
                thumb_layout.addWidget(frame_label)

            thumbnail_grid.addWidget(thumb_container, row, col)

        thumbnail_layout.addLayout(thumbnail_grid)
        content_layout.addWidget(thumbnail_group)

        # 移動軌跡
        trajectory_group = QGroupBox("移動軌跡（動画フレーム座標系）")
        trajectory_layout = QVBoxLayout(trajectory_group)

        trajectory_widget = TrajectoryVisualizationWidget(
            self._suggestion, self._store, self._video_width, self._video_height
        )
        trajectory_widget.setMinimumHeight(250)
        trajectory_layout.addWidget(trajectory_widget)

        content_layout.addWidget(trajectory_group)

        # 統計情報
        stats_group = QGroupBox("統計情報")
        stats_layout = QVBoxLayout(stats_group)

        confidence_percent = int(self._suggestion.confidence * 100)
        avg_time = sum(self._suggestion.time_gaps) // len(self._suggestion.time_gaps) if self._suggestion.time_gaps else 0
        avg_pos = sum(self._suggestion.position_distances) / len(self._suggestion.position_distances) if self._suggestion.position_distances else 0

        # トラック情報を収集
        track_infos = [info for info in collect_track_infos(self._store) if info.track_id in self._suggestion.track_ids]
        if track_infos:
            frame_min = min(info.frame_min for info in track_infos)
            frame_max = max(info.frame_max for info in track_infos)
        else:
            frame_min = frame_max = 0

        stats_text = f"""
<table style="width: 100%;">
<tr><td><b>トラック数:</b></td><td>{len(self._suggestion.track_ids)}個</td></tr>
<tr><td><b>フレーム範囲:</b></td><td>{frame_min} - {frame_max}</td></tr>
<tr><td><b>平均時間差:</b></td><td>{avg_time}フレーム</td></tr>
<tr><td><b>平均位置差:</b></td><td>{int(avg_pos)}ピクセル</td></tr>
<tr><td><b>信頼度:</b></td><td>{confidence_percent}%</td></tr>
</table>
        """
        stats_label = QLabel(stats_text)
        stats_label.setTextFormat(Qt.RichText)
        stats_layout.addWidget(stats_label)

        content_layout.addWidget(stats_group)

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        merge_btn = QPushButton("統合")
        merge_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 3px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        merge_btn.clicked.connect(self._on_merge_clicked)
        button_layout.addWidget(merge_btn)

        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _on_merge_clicked(self):
        """統合ボタンクリック"""
        self.merge_requested.emit(self._suggestion.track_ids)
        self.accept()


class TrajectoryVisualizationWidget(QWidget):
    """移動軌跡可視化ウィジェット（動画フレーム座標系）"""

    def __init__(self, suggestion: MergeSuggestion, store: AnnotationStore, video_width: int, video_height: int, parent=None):
        super().__init__(parent)
        self._suggestion = suggestion
        self._store = store
        self._track_infos = collect_track_infos(store)
        self._video_width = video_width
        self._video_height = video_height

    def paintEvent(self, event):
        """描画イベント（動画フレーム座標系でマッピング）"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # 軌跡を描画
        track_infos = [info for info in self._track_infos if info.track_id in self._suggestion.track_ids]
        if not track_infos:
            return

        # アスペクト比を保持してスケール計算
        video_aspect = self._video_width / self._video_height
        widget_aspect = self.width() / self.height()

        margin = 20

        if video_aspect > widget_aspect:
            # 幅に合わせる
            scale = (self.width() - 2 * margin) / self._video_width
            offset_x = margin
            offset_y = (self.height() - self._video_height * scale) / 2
        else:
            # 高さに合わせる
            scale = (self.height() - 2 * margin) / self._video_height
            offset_x = (self.width() - self._video_width * scale) / 2
            offset_y = margin

        # フレーム境界を描画
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawRect(
            int(offset_x),
            int(offset_y),
            int(self._video_width * scale),
            int(self._video_height * scale)
        )

        # 座標変換関数
        def to_screen(bbox_x, bbox_y):
            return (
                int(offset_x + bbox_x * scale),
                int(offset_y + bbox_y * scale)
            )

        # 各トラックを描画
        total_tracks = len(track_infos)
        for idx, info in enumerate(track_infos):
            # 色のグラデーション（青→赤）
            # 各トラック内でグラデーション
            start_color = QColor(100, 150, 255)  # 青（開始）
            end_color = QColor(255, 100, 100)    # 赤（終了）

            # 開始位置
            x1, y1, x2, y2 = info.first_bbox
            start_x, start_y = to_screen((x1 + x2) / 2, (y1 + y2) / 2)

            # 終了位置
            x1, y1, x2, y2 = info.last_bbox
            end_x, end_y = to_screen((x1 + x2) / 2, (y1 + y2) / 2)

            # グラデーション線を描画
            from PyQt5.QtGui import QLinearGradient
            gradient = QLinearGradient(start_x, start_y, end_x, end_y)
            gradient.setColorAt(0, start_color)
            gradient.setColorAt(1, end_color)

            painter.setPen(QPen(QBrush(gradient), 3))
            painter.drawLine(start_x, start_y, end_x, end_y)

            # 開始点（青の○）
            painter.setBrush(QBrush(start_color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(start_x - 6, start_y - 6, 12, 12)

            # 終了点（赤の●）
            painter.setBrush(QBrush(end_color))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(end_x - 6, end_y - 6, 12, 12)

            # 次のトラックとの接続線（あれば）
            if idx < len(track_infos) - 1:
                next_info = track_infos[idx + 1]
                x1, y1, x2, y2 = next_info.first_bbox
                next_start_x, next_start_y = to_screen((x1 + x2) / 2, (y1 + y2) / 2)

                # 薄いグレーの点線で接続
                painter.setPen(QPen(QColor(100, 100, 100, 128), 2, Qt.DashLine))
                painter.drawLine(end_x, end_y, next_start_x, next_start_y)

        # 凡例
        legend_x = int(offset_x) + 10
        legend_y = int(offset_y) + 10

        # 開始点の凡例
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(legend_x, legend_y, 10, 10)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(legend_x + 15, legend_y + 9, "開始")

        # 終了点の凡例
        painter.setBrush(QBrush(QColor(255, 100, 100)))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(legend_x + 60, legend_y, 10, 10)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 75, legend_y + 9, "終了")


class MergeSuggestionCard(QFrame):
    """1つの統合候補をカード形式で表示"""

    merge_requested = pyqtSignal(list)  # [track_id, ...]
    ignored = pyqtSignal()

    def __init__(self, suggestion: MergeSuggestion, cache: ThumbnailCache, store: AnnotationStore,
                 video_width: int, video_height: int, parent=None):
        super().__init__(parent)
        self._suggestion = suggestion
        self._cache = cache
        self._store = store
        self._video_width = video_width
        self._video_height = video_height
        self._thumbnail_labels: list[TrackThumbnailLabel] = []
        self._visible_track_ids: list[int] = []  # 実際に表示されるトラックID
        self._thumbnails_loaded = False

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            MergeSuggestionCard {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 3px;
            }
            MergeSuggestionCard:hover {
                background-color: #4a4a4a;
                border: 1px solid #0078d7;
            }
        """)

        self._setup_ui()
        # サムネイルは遅延ロード（showEventで）

        # ツールチップを設定
        self._update_tooltip()

        # マウストラッキングを有効化
        self.setMouseTracking(True)

    def _setup_ui(self):
        """UIを構築（改善版: 統合ボタンを左端、長い候補は省略）"""
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(5, 5, 5, 5)

        # 統合ボタン（最優先で左端に配置）
        merge_btn = QPushButton("統合")
        merge_btn.setFixedSize(50, 50)
        merge_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        merge_btn.clicked.connect(self._on_merge_clicked)
        layout.addWidget(merge_btn)

        # サムネイル一覧（長い場合は省略表示）
        max_visible_tracks = 5  # 最大表示数
        track_count = len(self._suggestion.track_ids)

        if track_count <= max_visible_tracks:
            # 全て表示
            for i, track_id in enumerate(self._suggestion.track_ids):
                thumb_label = TrackThumbnailLabel(track_id)
                self._thumbnail_labels.append(thumb_label)
                self._visible_track_ids.append(track_id)
                layout.addWidget(thumb_label)

                if i < track_count - 1:
                    arrow = QLabel("→")
                    arrow.setStyleSheet("font-size: 14pt; color: #888;")
                    layout.addWidget(arrow)
        else:
            # 省略表示: 最初の2つ → ... → 最後の2つ
            visible_indices = [0, 1, track_count - 2, track_count - 1]

            for idx, pos in enumerate(visible_indices):
                track_id = self._suggestion.track_ids[pos]
                thumb_label = TrackThumbnailLabel(track_id)
                self._thumbnail_labels.append(thumb_label)
                self._visible_track_ids.append(track_id)
                layout.addWidget(thumb_label)

                if idx == 1:
                    # 省略記号
                    ellipsis = QLabel(f"... ({track_count - 4}個) ...")
                    ellipsis.setStyleSheet("color: #888; font-size: 9pt;")
                    ellipsis.setAlignment(Qt.AlignCenter)
                    ellipsis.setMinimumWidth(80)
                    layout.addWidget(ellipsis)
                elif idx < len(visible_indices) - 1:
                    arrow = QLabel("→")
                    arrow.setStyleSheet("font-size: 14pt; color: #888;")
                    layout.addWidget(arrow)

        # 統計情報
        confidence_percent = int(self._suggestion.confidence * 100)
        stats_text = f"{confidence_percent}%"

        if self._suggestion.time_gaps:
            avg_time = sum(self._suggestion.time_gaps) // len(self._suggestion.time_gaps)
            stats_text += f" | {avg_time}f"

        if track_count > max_visible_tracks:
            stats_text = f"({track_count}個) " + stats_text

        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("color: #bbb; font-size: 10pt;")
        stats_label.setMinimumWidth(80)
        layout.addWidget(stats_label)

        layout.addStretch()

        # 無視ボタン（右端）
        ignore_btn = QPushButton("✕")
        ignore_btn.setFixedSize(28, 28)
        ignore_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
        ignore_btn.clicked.connect(self.ignored.emit)
        layout.addWidget(ignore_btn)

    def _load_thumbnails(self):
        """サムネイルをロード（キャッシュ済みのもののみ、表示されるトラックのみ）"""
        if self._thumbnails_loaded:
            return
        self._thumbnails_loaded = True

        # 表示されるトラックIDのみをロード
        for i, track_id in enumerate(self._visible_track_ids):
            # キャッシュに既に存在する場合のみ取得（同期生成を避ける）
            if track_id in self._cache._cache:
                thumbnail = self._cache._cache[track_id]
                if thumbnail is not None:
                    self._thumbnail_labels[i].set_thumbnail(thumbnail)

    def showEvent(self, event):
        """表示時にサムネイルを遅延ロード"""
        super().showEvent(event)
        if not self._thumbnails_loaded:
            self._load_thumbnails()

    def _update_tooltip(self):
        """ツールチップを更新"""
        track_ids_str = " → ".join(f"#{tid}" for tid in self._suggestion.track_ids)
        confidence_percent = int(self._suggestion.confidence * 100)
        avg_time = sum(self._suggestion.time_gaps) // len(self._suggestion.time_gaps) if self._suggestion.time_gaps else 0
        avg_pos = sum(self._suggestion.position_distances) / len(self._suggestion.position_distances) if self._suggestion.position_distances else 0

        tooltip = f"""<b>統合候補</b><br>
トラック: {track_ids_str}<br>
信頼度: {confidence_percent}%<br>
平均時間差: {avg_time}フレーム<br>
平均位置差: {int(avg_pos)}ピクセル<br>
<br>
<i>ダブルクリックで詳細表示</i>
"""
        self.setToolTip(tooltip)

    def mouseDoubleClickEvent(self, event):
        """ダブルクリックで詳細ダイアログを表示"""
        if event.button() == Qt.LeftButton:
            self._show_detail_dialog()

    def _show_detail_dialog(self):
        """詳細ダイアログを表示"""
        dialog = MergeSuggestionDetailDialog(
            self._suggestion, self._cache, self._store,
            self._video_width, self._video_height, self
        )
        dialog.merge_requested.connect(self.merge_requested.emit)
        dialog.exec_()

    def _on_merge_clicked(self):
        """統合ボタンクリック"""
        self.merge_requested.emit(self._suggestion.track_ids)


class ImprovedMergeSuggestionWidget(QWidget):
    """改善版統合候補ウィジェット（カード形式）"""

    merge_requested = pyqtSignal(list)  # [track_id, ...]

    def __init__(self, cache: ThumbnailCache, store: AnnotationStore,
                 video_width: int, video_height: int, parent=None):
        super().__init__(parent)
        self._cache = cache
        self._store = store
        self._video_width = video_width
        self._video_height = video_height
        self._suggestions: list[MergeSuggestion] = []
        self._cards: list[MergeSuggestionCard] = []
        self._setup_ui()

    def _setup_ui(self):
        """UIを構築"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # タイトル
        self._title_label = QLabel("統合候補")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #ddd;")
        layout.addWidget(self._title_label)

        # スクロールエリア
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2b2b2b; }")

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(10)
        self._card_layout.addStretch()

        scroll_area.setWidget(self._card_container)
        layout.addWidget(scroll_area)

    def clear(self) -> None:
        """全てのカードをクリア"""
        # まずカードリストをクリア
        self._cards.clear()
        self._suggestions.clear()

        # レイアウトから全ウィジェットを即座に削除
        while self._card_layout.count() > 0:
            item = self._card_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def set_suggestions(self, suggestions: list[MergeSuggestion]) -> None:
        """統合候補を設定"""
        # 既存のカードを完全にクリア
        self.clear()

        self._suggestions = list(suggestions)  # コピーして保持

        # 新しいカードを追加
        for suggestion in suggestions:
            card = MergeSuggestionCard(
                suggestion, self._cache, self._store,
                self._video_width, self._video_height, self
            )
            card.merge_requested.connect(self.merge_requested.emit)
            card.ignored.connect(lambda c=card: self._on_card_ignored(c))
            self._cards.append(card)
            self._card_layout.addWidget(card)

        # 末尾のストレッチ
        self._card_layout.addStretch()

        # タイトルを更新
        self._title_label.setText(f"統合候補 ({len(suggestions)}件)")

    def _on_card_ignored(self, card: MergeSuggestionCard):
        """カードが無視された"""
        if card in self._cards:
            self._cards.remove(card)
            # サジェストリストからも削除
            if card._suggestion in self._suggestions:
                self._suggestions.remove(card._suggestion)
        card.setParent(None)
        card.deleteLater()
        # タイトルを更新
        self._title_label.setText(f"統合候補 ({len(self._cards)}件)")


class MergeSuggestionWidget(QWidget):
    """統合サジェストウィジェット（複数トラック対応）"""

    merge_requested = pyqtSignal(list)  # [track_id, ...]
    suggestion_ignored = pyqtSignal(object)  # MergeSuggestion

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suggestions: list[MergeSuggestion] = []
        self._setup_ui()

    def _setup_ui(self):
        """UIを構築"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # タイトル
        self._title_label = QLabel("統合候補")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self._title_label)

        # スクロールエリア
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)

        self._suggestion_container = QWidget()
        self._suggestion_layout = QVBoxLayout(self._suggestion_container)
        self._suggestion_layout.setSpacing(10)
        self._suggestion_layout.addStretch()

        scroll_area.setWidget(self._suggestion_container)
        layout.addWidget(scroll_area)

    def clear(self) -> None:
        """全てのサジェストをクリア"""
        self._suggestions.clear()

        # レイアウトから全ウィジェットを即座に削除
        while self._suggestion_layout.count() > 0:
            item = self._suggestion_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

    def set_suggestions(self, suggestions: list[MergeSuggestion]) -> None:
        """サジェストを設定"""
        # 既存を完全にクリア
        self.clear()

        self._suggestions = list(suggestions)  # コピーして保持

        # 新しいサジェストを追加
        for suggestion in suggestions:
            self._add_suggestion_item(suggestion)

        self._suggestion_layout.addStretch()

        # タイトルを更新
        self._title_label.setText(f"統合候補 ({len(suggestions)}件)")

    def _add_suggestion_item(self, suggestion: MergeSuggestion) -> None:
        """サジェストアイテムを追加（複数トラック対応）"""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 5, 5, 5)

        # アイコン（複数トラックの場合は異なるアイコン）
        if suggestion.is_multi_track:
            icon_label = QLabel("🔗")  # チェーンアイコン
        else:
            icon_label = QLabel("⚠")
        icon_label.setStyleSheet("font-size: 16pt;")
        item_layout.addWidget(icon_label)

        # テキスト
        confidence_percent = int(suggestion.confidence * 100)

        if suggestion.is_multi_track:
            # 複数トラック: #1 → #3 → #5 (3個)
            track_chain = " → ".join(f"#{tid}" for tid in suggestion.track_ids)
            avg_time = sum(suggestion.time_gaps) // len(suggestion.time_gaps) if suggestion.time_gaps else 0
            avg_pos = sum(suggestion.position_distances) / len(suggestion.position_distances) if suggestion.position_distances else 0
            text = (
                f"{track_chain} ({suggestion.track_count}個) "
                f"{confidence_percent}% "
                f"平均時間:{avg_time}f 平均位置:{int(avg_pos)}px"
            )
        else:
            # 2トラック: #1 → #3
            text = (
                f"#{suggestion.track_ids[0]} → #{suggestion.track_ids[1]} "
                f"({confidence_percent}%) "
                f"位置差:{int(suggestion.position_distances[0])}px "
                f"時間:{suggestion.time_gaps[0]}f"
            )

        text_label = QLabel(text)
        text_label.setWordWrap(True)
        item_layout.addWidget(text_label, stretch=1)

        # 統合ボタン
        if suggestion.is_multi_track:
            merge_btn = QPushButton(f"統合({suggestion.track_count})")
        else:
            merge_btn = QPushButton("統合")
        merge_btn.clicked.connect(lambda: self.merge_requested.emit(suggestion.track_ids))
        item_layout.addWidget(merge_btn)

        # 無視ボタン
        ignore_btn = QPushButton("無視")
        ignore_btn.clicked.connect(lambda: self._ignore_suggestion(suggestion, item_widget))
        item_layout.addWidget(ignore_btn)

        self._suggestion_layout.insertWidget(self._suggestion_layout.count() - 1, item_widget)

    def _ignore_suggestion(self, suggestion: MergeSuggestion, widget: QWidget) -> None:
        """サジェストを無視"""
        self.suggestion_ignored.emit(suggestion)
        widget.deleteLater()


class MergeThresholdDialog(QDialog):
    """統合候補しきい値設定ダイアログ"""

    def __init__(self, current_time_gap=40, current_position=200.0, current_confidence=0.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle("統合候補の詳細設定")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.max_time_gap = current_time_gap
        self.max_position_distance = current_position
        self.min_confidence = current_confidence

        self._setup_ui()

    def _setup_ui(self):
        """UIを構築"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 説明
        desc_label = QLabel(
            "統合候補の検出条件を調整します。\n"
            "厳しくすると候補が減り、緩くすると候補が増えます。"
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888;")
        layout.addWidget(desc_label)

        # しきい値設定
        from PyQt5.QtWidgets import QGridLayout
        settings_grid = QGridLayout()
        settings_grid.setSpacing(10)
        settings_grid.setColumnStretch(1, 1)

        # 時間差しきい値
        settings_grid.addWidget(QLabel("時間差しきい値:"), 0, 0)
        self._time_gap_slider = QSlider(Qt.Horizontal)
        self._time_gap_slider.setMinimum(10)
        self._time_gap_slider.setMaximum(120)
        self._time_gap_slider.setValue(int(self.max_time_gap))
        settings_grid.addWidget(self._time_gap_slider, 0, 1)
        self._time_gap_label = QLabel(f"{int(self.max_time_gap)}フレーム")
        self._time_gap_label.setMinimumWidth(80)
        settings_grid.addWidget(self._time_gap_label, 0, 2)

        time_desc = QLabel("トラック間の最大時間差（小さいほど厳しい）")
        time_desc.setStyleSheet("color: #888; font-size: 9pt;")
        settings_grid.addWidget(time_desc, 1, 1, 1, 2)

        # 位置差しきい値
        settings_grid.addWidget(QLabel("位置差しきい値:"), 2, 0)
        self._position_slider = QSlider(Qt.Horizontal)
        self._position_slider.setMinimum(50)
        self._position_slider.setMaximum(500)
        self._position_slider.setValue(int(self.max_position_distance))
        settings_grid.addWidget(self._position_slider, 2, 1)
        self._position_label = QLabel(f"{int(self.max_position_distance)}ピクセル")
        self._position_label.setMinimumWidth(80)
        settings_grid.addWidget(self._position_label, 2, 2)

        pos_desc = QLabel("トラック間の最大位置差（小さいほど厳しい）")
        pos_desc.setStyleSheet("color: #888; font-size: 9pt;")
        settings_grid.addWidget(pos_desc, 3, 1, 1, 2)

        # 信頼度しきい値
        settings_grid.addWidget(QLabel("信頼度しきい値:"), 4, 0)
        self._confidence_slider = QSlider(Qt.Horizontal)
        self._confidence_slider.setMinimum(30)
        self._confidence_slider.setMaximum(90)
        self._confidence_slider.setValue(int(self.min_confidence * 100))
        settings_grid.addWidget(self._confidence_slider, 4, 1)
        self._confidence_label = QLabel(f"{int(self.min_confidence * 100)}%")
        self._confidence_label.setMinimumWidth(80)
        settings_grid.addWidget(self._confidence_label, 4, 2)

        conf_desc = QLabel("統合候補の最小信頼度（高いほど厳しい）")
        conf_desc.setStyleSheet("color: #888; font-size: 9pt;")
        settings_grid.addWidget(conf_desc, 5, 1, 1, 2)

        layout.addLayout(settings_grid)

        # スライダーの変更イベント
        self._time_gap_slider.valueChanged.connect(self._update_time_gap)
        self._position_slider.valueChanged.connect(self._update_position)
        self._confidence_slider.valueChanged.connect(self._update_confidence)

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("検出を実行")
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 5px 15px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

    def _update_time_gap(self, value):
        """時間差しきい値を更新"""
        self.max_time_gap = value
        self._time_gap_label.setText(f"{value}フレーム")

    def _update_position(self, value):
        """位置差しきい値を更新"""
        self.max_position_distance = float(value)
        self._position_label.setText(f"{value}ピクセル")

    def _update_confidence(self, value):
        """信頼度しきい値を更新"""
        self.min_confidence = value / 100.0
        self._confidence_label.setText(f"{value}%")


class TrackEditorDialog(QDialog):
    """トラック編集ダイアログ"""

    def __init__(
        self,
        parent,
        store: AnnotationStore,
        total_frames: int,
        video_width: int,
        video_height: int,
        video_path: str = None,
        current_frame: int = 0,
        precomputed_suggestions: list[MergeSuggestion] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("トラック編集")
        self.setMinimumSize(1000, 700)

        self._store = store
        self._total_frames = total_frames
        self._video_width = video_width
        self._video_height = video_height
        self._video_path = video_path
        self._current_frame = current_frame
        self._precomputed_suggestions = precomputed_suggestions

        # サムネイルキャッシュ
        self._thumbnail_cache = None
        if video_path:
            self._thumbnail_cache = ThumbnailCache(video_path, self)

        # Undo用に初期状態を保存
        self._initial_state = store.to_dict()
        self._change_count = 0

        self._thumbnail_worker = None

        self._setup_ui()
        self._load_suggestions()

        # サムネイルをバックグラウンドで生成
        if self._thumbnail_cache:
            self._start_thumbnail_loading()

    def _setup_ui(self):
        """UIを構築"""
        layout = QVBoxLayout(self)

        # ステータスラベル
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self._status_label)

        # 上部: ミニマップ + 統合候補
        top_splitter = QSplitter(Qt.Horizontal)

        # ミニマップ
        minimap_container = QWidget()
        minimap_layout = QVBoxLayout(minimap_container)
        minimap_layout.addWidget(QLabel("ミニマップ"))
        self._minimap = MiniMapWidget()
        self._minimap.set_store(self._store, self._video_width, self._video_height)
        self._minimap.set_current_frame(self._current_frame)
        minimap_layout.addWidget(self._minimap)
        top_splitter.addWidget(minimap_container)

        # しきい値デフォルト値
        self._max_time_gap = 40
        self._max_position_distance = 200.0
        self._min_confidence = 0.5

        # 統合候補（改善版 or 旧版）
        if self._thumbnail_cache:
            self._suggestion_widget = ImprovedMergeSuggestionWidget(
                self._thumbnail_cache, self._store,
                self._video_width, self._video_height
            )
        else:
            self._suggestion_widget = MergeSuggestionWidget()
        self._suggestion_widget.merge_requested.connect(self._on_merge_multiple_tracks)
        top_splitter.addWidget(self._suggestion_widget)

        top_splitter.setSizes([300, 700])
        layout.addWidget(top_splitter)

        # タイムラインヘッダー（タイトル + トグル）
        timeline_header = QHBoxLayout()
        timeline_label = QLabel("トラックタイムライン")
        timeline_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        timeline_header.addWidget(timeline_label)

        self._track_count_label = QLabel("")
        self._track_count_label.setStyleSheet("color: #888;")
        timeline_header.addWidget(self._track_count_label)

        timeline_header.addStretch()

        from PyQt5.QtWidgets import QCheckBox
        self._show_all_checkbox = QCheckBox("全トラック表示")
        self._show_all_checkbox.setChecked(False)
        self._show_all_checkbox.toggled.connect(self._on_show_all_toggled)
        timeline_header.addWidget(self._show_all_checkbox)

        layout.addLayout(timeline_header)

        # スクロールエリアでタイムラインを包む
        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setMinimumHeight(200)
        timeline_scroll.setMaximumHeight(300)

        self._timeline = TrackTimelineWidget()
        self._timeline.set_store(self._store, self._total_frames)
        self._timeline.set_current_frame(self._current_frame)
        if self._thumbnail_cache:
            self._timeline.set_thumbnail_cache(self._thumbnail_cache)
        self._timeline.track_selected.connect(self._on_track_selected)
        self._timeline.tracks_selected.connect(self._on_tracks_selected)
        self._timeline.tracks_merge_requested.connect(self._on_merge_multiple_tracks)
        self._timeline.frame_changed.connect(self._on_frame_changed)

        timeline_scroll.setWidget(self._timeline)
        layout.addWidget(timeline_scroll)

        # フレームスライダー
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("フレーム:"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(self._total_frames - 1)
        self._frame_slider.setValue(self._current_frame)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self._frame_slider, stretch=1)
        self._frame_label = QLabel(f"{self._current_frame}")
        slider_layout.addWidget(self._frame_label)
        layout.addLayout(slider_layout)

        # ボタン
        button_layout = QHBoxLayout()

        # 自動検出ボタン（ドロップダウンメニュー付き）
        detect_btn = QPushButton("自動検出を実行")
        detect_btn.clicked.connect(self._reload_suggestions)

        # ドロップダウンメニュー
        detect_menu = QMenu(self)
        detect_normal_action = QAction("通常検出", self)
        detect_normal_action.triggered.connect(self._reload_suggestions)
        detect_menu.addAction(detect_normal_action)

        detect_advanced_action = QAction("詳細設定で検出...", self)
        detect_advanced_action.triggered.connect(self._reload_suggestions_with_dialog)
        detect_menu.addAction(detect_advanced_action)

        detect_btn.setMenu(detect_menu)
        button_layout.addWidget(detect_btn)

        interpolate_btn = QPushButton("全トラック補間")
        interpolate_btn.clicked.connect(self._interpolate_all_tracks)
        button_layout.addWidget(interpolate_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("キャンセル")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("適用")
        apply_btn.clicked.connect(self.accept)
        button_layout.addWidget(apply_btn)

        layout.addLayout(button_layout)

    def _stop_thumbnail_worker(self):
        """サムネイルワーカーを停止"""
        if self._thumbnail_worker:
            self._thumbnail_worker.cancel()
            self._thumbnail_worker.wait()
            self._thumbnail_worker.deleteLater()
            self._thumbnail_worker = None

    def _load_suggestions(self):
        """統合サジェストを読み込み（初回用）"""
        if self._precomputed_suggestions is not None:
            # 事前計算された統合候補を使用
            suggestions = self._precomputed_suggestions
            self._precomputed_suggestions = None  # 一度使ったらクリア
        else:
            # その場で計算（デフォルトしきい値）
            suggestions = compute_merge_suggestions(self._store)

        self._apply_suggestions(suggestions)

    def _refresh_suggestions_with_current_thresholds(self):
        """現在のしきい値で統合サジェストを再計算"""
        suggestions = compute_merge_suggestions(
            self._store,
            max_time_gap=self._max_time_gap,
            max_position_distance=self._max_position_distance,
            min_confidence=self._min_confidence,
        )
        self._apply_suggestions(suggestions)

    def _apply_suggestions(self, suggestions: list[MergeSuggestion]):
        """統合候補をUIに適用"""
        self._suggestion_widget.set_suggestions(suggestions)
        # タイムラインにも統合候補を設定（接続線描画用）
        self._timeline.set_suggestions(suggestions)
        # トラック数ラベルを更新
        self._update_track_count_label()

    def _reload_suggestions(self):
        """統合サジェストを再読み込み（現在のしきい値）"""
        # ワーカーを停止
        self._stop_thumbnail_worker()

        # 統合候補を再計算
        self._refresh_suggestions_with_current_thresholds()

        # サムネイル生成を再開
        if self._thumbnail_cache:
            self._start_thumbnail_loading()

        # 統合候補の数を取得
        suggestion_count = len(self._suggestion_widget._suggestions) if hasattr(self._suggestion_widget, '_suggestions') else 0
        self._status_label.setText(f"✓ 統合候補を再検出しました ({suggestion_count}件)")

    def _reload_suggestions_with_dialog(self):
        """統合サジェストを再読み込み（ダイアログで設定）"""
        # しきい値設定ダイアログを表示
        dialog = MergeThresholdDialog(
            self._max_time_gap,
            self._max_position_distance,
            self._min_confidence,
            self
        )

        if dialog.exec_() == QDialog.Accepted:
            # ワーカーを停止
            self._stop_thumbnail_worker()

            # ダイアログから値を取得
            self._max_time_gap = dialog.max_time_gap
            self._max_position_distance = dialog.max_position_distance
            self._min_confidence = dialog.min_confidence

            # カスタムしきい値で再計算
            self._refresh_suggestions_with_current_thresholds()

            # サムネイル生成を再開
            if self._thumbnail_cache:
                self._start_thumbnail_loading()

            # 統合候補の数を取得
            suggestion_count = len(self._suggestion_widget._suggestions) if hasattr(self._suggestion_widget, '_suggestions') else 0

            self._status_label.setText(
                f"✓ カスタムしきい値で再検出 ({suggestion_count}件) "
                f"[時間:{self._max_time_gap}f, 位置:{int(self._max_position_distance)}px, 信頼度:{int(self._min_confidence*100)}%]"
            )

    def _on_track_selected(self, track_id: int):
        """トラック選択時（単一選択）"""
        self._timeline.set_selected_track(track_id)

    def _on_tracks_selected(self, track_ids: list[int]):
        """トラック選択時（複数選択）"""
        if len(track_ids) > 1:
            self._status_label.setText(f"{len(track_ids)}個のトラックを選択中（右クリックで統合）")
        elif len(track_ids) == 1:
            self._status_label.setText(f"トラック #{track_ids[0]} を選択中")
        else:
            self._status_label.setText("")

    def _on_merge_multiple_tracks(self, track_ids: list[int]):
        """複数トラック統合要求時"""
        if len(track_ids) < 2:
            return

        # サムネイルワーカーを停止（古いデータを参照し続けないよう）
        self._stop_thumbnail_worker()

        # 最初のトラックIDを統合先とする
        target_track_id = track_ids[0]
        total_count = 0

        # 他のすべてのトラックを統合先に統合
        merged_track_ids = []
        for source_track_id in track_ids[1:]:
            count = self._store.merge_tracks(source_track_id, target_track_id, save_undo=False)
            total_count += count
            merged_track_ids.append(source_track_id)

        self._change_count += 1

        # 最後に1回だけUndoスタックに保存
        self._store._save_undo_state()

        # 統合されたトラックのサムネイルキャッシュをクリア
        if self._thumbnail_cache:
            for tid in merged_track_ids:
                if tid in self._thumbnail_cache._cache:
                    del self._thumbnail_cache._cache[tid]

        # タイムラインの選択状態をクリア
        self._timeline._selected_track_ids.clear()
        self._timeline._last_selected_track_id = None

        # UIを更新
        self._timeline.refresh()
        self._minimap.set_current_frame(self._current_frame)

        # 統合候補を再計算（現在のしきい値を使用）
        self._refresh_suggestions_with_current_thresholds()

        # サムネイル生成を再開
        if self._thumbnail_cache:
            self._start_thumbnail_loading()

        # ステータス表示
        track_list = ", ".join(f"#{tid}" for tid in track_ids[1:])
        self._status_label.setText(
            f"✓ {len(track_ids)}個のトラック ({track_list}) → #{target_track_id} に統合しました ({total_count}個のアノテーション)"
        )

    def _on_frame_changed(self, frame: int):
        """フレーム変更時"""
        self._current_frame = frame
        self._frame_slider.setValue(frame)
        self._frame_label.setText(str(frame))
        self._timeline.set_current_frame(frame)
        self._minimap.set_current_frame(frame)

    def _on_slider_changed(self, value: int):
        """スライダー変更時"""
        self._on_frame_changed(value)

    def _on_show_all_toggled(self, checked: bool):
        """全トラック表示トグル"""
        self._timeline.set_show_all_tracks(checked)
        self._update_track_count_label()

    def _update_track_count_label(self):
        """トラック数ラベルを更新"""
        total = len(self._timeline._track_infos)
        filtered = len(self._timeline._filtered_track_ids)
        if self._show_all_checkbox.isChecked():
            self._track_count_label.setText(f"({total}トラック)")
        else:
            self._track_count_label.setText(f"({filtered}/{total}トラック)")

    def _start_thumbnail_loading(self):
        """サムネイルをバックグラウンドで生成（統合候補カードに表示されるトラックのみ）"""
        # 統合候補カードに実際に表示されるトラックIDのみを収集
        track_ids_to_load = set()
        max_visible_tracks = 5  # MergeSuggestionCardと同じ値

        if hasattr(self._suggestion_widget, '_cards'):
            for card in self._suggestion_widget._cards:
                suggestion = card._suggestion
                track_count = len(suggestion.track_ids)

                if track_count <= max_visible_tracks:
                    # 全て表示される場合
                    track_ids_to_load.update(suggestion.track_ids)
                else:
                    # 省略表示の場合: 最初の2つと最後の2つのみ
                    track_ids_to_load.add(suggestion.track_ids[0])
                    track_ids_to_load.add(suggestion.track_ids[1])
                    track_ids_to_load.add(suggestion.track_ids[-2])
                    track_ids_to_load.add(suggestion.track_ids[-1])

        track_ids = list(track_ids_to_load)

        if not track_ids:
            return

        # ワーカーを起動
        self._thumbnail_worker = ThumbnailLoaderWorker(
            self._thumbnail_cache,
            track_ids,
            self._store,
            self
        )

        def on_thumbnail_loaded(track_id, pixmap):
            # 統合候補カードを更新（即座に表示）
            if hasattr(self._suggestion_widget, '_cards'):
                for card in self._suggestion_widget._cards:
                    # 表示されるトラックIDのみ更新
                    if track_id in card._visible_track_ids:
                        idx = card._visible_track_ids.index(track_id)
                        if idx < len(card._thumbnail_labels):
                            card._thumbnail_labels[idx].set_thumbnail(pixmap)

        def on_progress(current, total):
            # 進捗をステータスに表示（控えめに）
            if current % 5 == 0:  # 5個ごとに更新
                self._status_label.setText(f"サムネイル生成中... ({current}/{total})")

        def on_finished():
            self._status_label.setText("")
            if self._thumbnail_worker:
                self._thumbnail_worker.deleteLater()
                self._thumbnail_worker = None

        self._thumbnail_worker.thumbnail_loaded.connect(on_thumbnail_loaded)
        self._thumbnail_worker.progress.connect(on_progress)
        self._thumbnail_worker.finished.connect(on_finished)
        self._thumbnail_worker.start()

    def _interpolate_all_tracks(self):
        """全トラック補間"""
        # ワーカーを停止
        self._stop_thumbnail_worker()

        count = interpolate_all_tracks(self._store)

        if count > 0:
            self._change_count += 1

            # UIを更新
            self._timeline.refresh()
            self._minimap.set_current_frame(self._current_frame)

            # 統合候補を再計算（補間でフレーム範囲が変わる可能性）
            self._refresh_suggestions_with_current_thresholds()

            # サムネイル生成を再開
            if self._thumbnail_cache:
                self._start_thumbnail_loading()

            self._status_label.setText(f"✓ 全トラック補間完了 ({count}個のアノテーションを追加)")
        else:
            self._status_label.setText("ℹ 補間するフレームがありません")

    def reject(self):
        """キャンセル時"""
        # サムネイルワーカーをキャンセル
        if self._thumbnail_worker:
            self._thumbnail_worker.cancel()
            self._thumbnail_worker.wait()
            self._thumbnail_worker.deleteLater()
            self._thumbnail_worker = None

        # 変更がある場合のみ確認
        if self._change_count > 0:
            reply = QMessageBox.question(
                self,
                "変更を破棄",
                f"{self._change_count}件の変更を破棄してよろしいですか？",
                QMessageBox.Yes | QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                return

            # 初期状態に復元
            self._store._restore_state(self._initial_state)

        super().reject()

    def accept(self):
        """適用時"""
        # サムネイルワーカーをキャンセル
        if self._thumbnail_worker:
            self._thumbnail_worker.cancel()
            self._thumbnail_worker.wait()
            self._thumbnail_worker.deleteLater()
            self._thumbnail_worker = None

        super().accept()

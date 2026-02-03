"""トラック編集専用ダイアログ"""

from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
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
)

from defacer.gui.annotation import AnnotationStore
from defacer.tracking.merge_suggestion import (
    compute_merge_suggestions,
    collect_track_infos,
    MergeSuggestion,
)
from defacer.tracking.interpolation import interpolate_all_tracks


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

    def refresh(self) -> None:
        """トラック情報を再計算して再描画"""
        if self._store is not None:
            self._track_infos = collect_track_infos(self._store)
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

        if not self._track_infos:
            return

        # 描画パラメータ
        margin_left = 80
        margin_right = 20
        margin_top = 40
        margin_bottom = 20
        row_height = 30
        row_spacing = 5

        timeline_width = self.width() - margin_left - margin_right
        timeline_height = len(self._track_infos) * (row_height + row_spacing)

        # タイムラインの高さを調整
        self.setMinimumHeight(timeline_height + margin_top + margin_bottom)

        # タイムスケールを描画
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("Arial", 9)
        painter.setFont(font)

        # フレーム目盛り
        frame_interval = max(1, self._total_frames // 10)
        for i in range(0, self._total_frames + 1, frame_interval):
            x = margin_left + int(i / self._total_frames * timeline_width)
            painter.drawLine(x, margin_top - 10, x, margin_top - 5)
            painter.drawText(x - 20, margin_top - 15, 40, 15, Qt.AlignCenter, str(i))

        # 現在フレームのインジケータ
        current_x = margin_left + int(self._current_frame / self._total_frames * timeline_width)
        painter.setPen(QPen(QColor(255, 165, 0), 2))
        painter.drawLine(current_x, margin_top, current_x, margin_top + timeline_height)
        painter.drawText(
            current_x - 30,
            margin_top + timeline_height + 5,
            60,
            15,
            Qt.AlignCenter,
            f"▼ {self._current_frame}",
        )

        # 各トラックを描画
        for idx, track_info in enumerate(self._track_infos):
            y = margin_top + idx * (row_height + row_spacing)

            # トラックID
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawText(10, y, 60, row_height, Qt.AlignVCenter | Qt.AlignRight, f"#{track_info.track_id}")

            # トラックバー
            start_x = margin_left + int(track_info.frame_min / self._total_frames * timeline_width)
            end_x = margin_left + int(track_info.frame_max / self._total_frames * timeline_width)
            bar_width = max(2, end_x - start_x)

            # 選択状態に応じて色を変更
            if track_info.track_id in self._selected_track_ids:
                bar_color = QColor(0, 120, 212)  # 選択中（青）
            else:
                bar_color = QColor(80, 80, 120)  # 通常（グレー）

            painter.fillRect(start_x, y + 5, bar_width, row_height - 10, bar_color)

            # 枠線（選択中は太く）
            if track_info.track_id in self._selected_track_ids:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(QPen(QColor(150, 150, 150), 1))
            painter.drawRect(start_x, y + 5, bar_width, row_height - 10)

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
        margin_left = 80
        margin_top = 40
        row_height = 30
        row_spacing = 5

        for idx, track_info in enumerate(self._track_infos):
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
        title_label = QLabel("統合候補")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title_label)

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

    def set_suggestions(self, suggestions: list[MergeSuggestion]) -> None:
        """サジェストを設定"""
        self._suggestions = suggestions

        # 既存のウィジェットをクリア
        for i in reversed(range(self._suggestion_layout.count())):
            widget = self._suggestion_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # 新しいサジェストを追加
        for suggestion in suggestions:
            self._add_suggestion_item(suggestion)

        self._suggestion_layout.addStretch()

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


class TrackEditorDialog(QDialog):
    """トラック編集ダイアログ"""

    def __init__(
        self,
        parent,
        store: AnnotationStore,
        total_frames: int,
        video_width: int,
        video_height: int,
        current_frame: int = 0,
    ):
        super().__init__(parent)
        self.setWindowTitle("トラック編集")
        self.setMinimumSize(1000, 700)

        self._store = store
        self._total_frames = total_frames
        self._video_width = video_width
        self._video_height = video_height
        self._current_frame = current_frame

        # Undo用に初期状態を保存
        self._initial_state = store.to_dict()
        self._change_count = 0

        self._setup_ui()
        self._load_suggestions()

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

        # 統合候補
        self._suggestion_widget = MergeSuggestionWidget()
        self._suggestion_widget.merge_requested.connect(self._on_merge_multiple_tracks)
        top_splitter.addWidget(self._suggestion_widget)

        top_splitter.setSizes([300, 700])
        layout.addWidget(top_splitter)

        # タイムライン
        timeline_label = QLabel("トラックタイムライン")
        timeline_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin-top: 10px;")
        layout.addWidget(timeline_label)

        # スクロールエリアでタイムラインを包む
        timeline_scroll = QScrollArea()
        timeline_scroll.setWidgetResizable(True)
        timeline_scroll.setMinimumHeight(250)

        self._timeline = TrackTimelineWidget()
        self._timeline.set_store(self._store, self._total_frames)
        self._timeline.set_current_frame(self._current_frame)
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

        detect_btn = QPushButton("自動検出を実行")
        detect_btn.clicked.connect(self._reload_suggestions)
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

    def _load_suggestions(self):
        """統合サジェストを読み込み"""
        suggestions = compute_merge_suggestions(self._store)
        self._suggestion_widget.set_suggestions(suggestions)

    def _reload_suggestions(self):
        """統合サジェストを再読み込み"""
        self._load_suggestions()
        suggestion_count = len(self._suggestion_widget._suggestions)
        self._status_label.setText(f"✓ 統合候補を再検出しました ({suggestion_count}件)")

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

        # 最初のトラックIDを統合先とする
        target_track_id = track_ids[0]
        total_count = 0

        # 他のすべてのトラックを統合先に統合
        for source_track_id in track_ids[1:]:
            count = self._store.merge_tracks(source_track_id, target_track_id, save_undo=False)
            total_count += count

        self._change_count += 1

        # 最後に1回だけUndoスタックに保存
        self._store._save_undo_state()

        # UIを更新
        self._timeline.refresh()
        self._minimap.set_current_frame(self._current_frame)

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

    def _interpolate_all_tracks(self):
        """全トラック補間"""
        count = interpolate_all_tracks(self._store)

        if count > 0:
            self._change_count += 1
            self._timeline.refresh()
            self._minimap.set_current_frame(self._current_frame)
            self._status_label.setText(f"✓ 全トラック補間完了 ({count}個のアノテーションを追加)")
        else:
            self._status_label.setText("ℹ 補間するフレームがありません")

    def reject(self):
        """キャンセル時"""
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

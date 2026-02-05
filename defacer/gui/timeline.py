"""タイムラインウィジェット"""

from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent, QImage
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QStyle,
    QStyleOptionSlider,
    QSizePolicy,
    QMenu,
    QAction,
    QScrollBar,
)


def format_time(seconds: float) -> str:
    """秒数を MM:SS.mmm 形式に変換"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


class TimelineSlider(QSlider):
    """カスタムタイムラインスライダー"""

    selection_changed = pyqtSignal(int, int)  # start, end (inclusive)
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(0)
        self._annotations: dict[int, list] = {}  # frame_number -> annotations
        self._selection_start: int | None = None
        self._selection_end: int | None = None
        self._is_selecting = False
        
        # 選択中トラックの可視化用
        self._track_frames: list[int] = []
        self._track_thumbnails: list[tuple[int, QImage]] = []  # (frame, image)
        
        # ズーム・スクロール状態
        self._zoom = 1.0  # 1.0 = 全体表示, 2.0 = 2倍拡大
        self._zoom_offset = 0.0  # 表示開始フレーム
        self._fps = 30.0
        
        self.setFixedHeight(60)  # 高さ固定

    def set_fps(self, fps: float) -> None:
        """FPSを設定"""
        self._fps = fps if fps > 0 else 30.0
        self.update()

    def set_zoom_view(self, zoom: float, offset: float) -> None:
        """ズームとスクロール位置を設定"""
        self._zoom = max(1.0, zoom)
        self._zoom_offset = max(0.0, offset)
        self.update()

    def set_selected_track(self, frames: list[int], thumbnails: list[tuple[int, QImage]] = None) -> None:
        """選択中のトラック情報を設定"""
        self._track_frames = frames
        self._track_thumbnails = thumbnails or []
        self.update()

    def set_annotations(self, annotations: dict[int, list]) -> None:
        """アノテーション位置を設定（タイムライン上にマーカー表示用）"""
        self._annotations = annotations
        self.update()

    def set_selection(self, start: int | None, end: int | None) -> None:
        """選択範囲を設定"""
        self._selection_start = start
        self._selection_end = end
        self.update()

    def get_selection(self) -> tuple[int, int] | None:
        """選択範囲を取得 (start, end) 、選択なしなら None"""
        if self._selection_start is None or self._selection_end is None:
            return None
        return (
            min(self._selection_start, self._selection_end),
            max(self._selection_start, self._selection_end),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """マウス押下イベント"""
        if event.modifiers() & Qt.ShiftModifier:
            # Shiftキーが押されている場合は範囲選択開始
            self._is_selecting = True
            value = self._pixel_to_value(event.x())
            self._selection_start = value
            self._selection_end = value
            self.update()
            # イベントを親に伝播させない（スライダー移動を防ぐ）
            event.accept()
        elif event.button() == Qt.RightButton:
            # 右クリックの場合は何もしない（コンテキストメニュー用）
            # ただし、選択範囲外での右クリックなら選択解除してもいいかもしれないが、
            # ユーザー体験的には保持したほうが安全
            event.accept()
        else:
            # 通常の動作
            if self.get_selection() is not None:
                # 選択範囲外をクリックしたら選択解除
                # ただし、今回はシンプルに「Shiftなしクリック」で解除とする
                self.set_selection(None, None)
                self.selection_cleared.emit()
            
            if event.button() == Qt.LeftButton:
                # クリック位置へ即座に移動
                value = self._pixel_to_value(event.x())
                self.setValue(value)
                # ドラッグ操作開始
                self.setSliderDown(True)
                event.accept()
            else:
                super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """マウス移動イベント"""
        if self._is_selecting:
            value = self._pixel_to_value(event.x())
            self._selection_end = value
            self.update()
            event.accept()
        elif event.buttons() & Qt.LeftButton:
            # 通常のドラッグ（シーク）
            value = self._pixel_to_value(event.x())
            self.setValue(value)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """マウスリリースイベント"""
        if self._is_selecting:
            self._is_selecting = False
            selection = self.get_selection()
            if selection:
                self.selection_changed.emit(*selection)
            event.accept()
        else:
            # ドラッグ終了（手動でsetSliderDown(True)した可能性があるため明示的にFalseにする）
            self.setSliderDown(False)
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """ホイールイベントは親（TimelineWidget）に委譲"""
        event.ignore()

    def paintEvent(self, event) -> None:
        """カスタム描画"""
        # 親のpaintEventは呼ばない（完全カスタム描画）
        # super().paintEvent(event)

        if self.maximum() == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 描画領域全体
        rect = self.rect()
        
        # 1. 背景描画
        painter.fillRect(rect, QColor(40, 40, 40))
        
        # 値の範囲
        val_range = self.maximum() - self.minimum()
        if val_range <= 0:
            return

        # 表示範囲（フレーム数）
        visible_range = val_range / self._zoom
        
        # 座標変換ヘルパー (値をピクセルへ)
        def val_to_x(val):
            # (val - offset) / visible_range * width
            normalized = (val - self._zoom_offset) / visible_range
            return int(normalized * rect.width())

        # 1.1. 目盛りの描画（ルーラースタイル）
        if self._fps > 0:
            # ルーラー領域の背景（上部20px）
            ruler_height = 20
            painter.fillRect(0, 0, rect.width(), ruler_height, QColor(50, 50, 50))
            
            # フォント設定
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            
            # 表示範囲
            start_frame = self._zoom_offset
            end_frame = start_frame + visible_range
            start_time = start_frame / self._fps
            end_time = end_frame / self._fps
            duration = end_time - start_time
            
            if duration > 0:
                pixels_per_sec = rect.width() / duration
                
                # ベース間隔候補
                intervals = [
                    0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 
                    300.0, 600.0, 1800.0, 3600.0, 7200.0
                ]
                
                # メインのラベル間隔を決定（ラベル幅を考慮して最低60pxくらい空ける）
                major_step = intervals[-1]
                min_label_dist = 80
                for interval in intervals:
                    if interval * pixels_per_sec >= min_label_dist:
                        major_step = interval
                        break
                        
                # サブ目盛り間隔（メインの1/5 または 1/2）
                minor_step = major_step / 5.0
                if minor_step * pixels_per_sec < 10:
                    minor_step = major_step / 2.0
                if minor_step * pixels_per_sec < 10:
                    minor_step = major_step  # サブなし
                
                # 描画ループ
                # start_time の少し前から描画して端切れを防ぐ
                t = (int(start_time / minor_step) * minor_step)
                
                while t <= end_time:
                    f = t * self._fps
                    x = val_to_x(f)
                    
                    if 0 <= x <= rect.width():
                        # メジャーかマイナーか判定 (浮動小数点の誤差を考慮)
                        # 許容誤差は minor_step の 10% 程度
                        epsilon = minor_step * 0.1
                        dist = abs(t % major_step)
                        # 周期境界（割り切れ）の判定: 0に近い or major_stepに近い
                        is_major = dist < epsilon or abs(major_step - dist) < epsilon
                        
                        if is_major:
                            # メジャー目盛り
                            painter.setPen(QPen(QColor(200, 200, 200), 1))
                            painter.drawLine(x, 0, x, ruler_height)
                            
                            # ラベル描画
                            time_str = format_time(t)
                            # 簡略化
                            if major_step < 1.0:
                                time_str = f"{t % 60:04.1f}"
                            elif major_step < 60:
                                m = int(t // 60)
                                s = int(t % 60)
                                time_str = f"{m:02d}:{s:02d}"
                            else:
                                m = int(t // 60)
                                h = int(m // 60)
                                m = m % 60
                                if h > 0:
                                    time_str = f"{h}h{m:02d}m"
                                else:
                                    time_str = f"{m}m"

                            # テキストの位置
                            text_rect = painter.fontMetrics().boundingRect(time_str)
                            text_x = x + 4 # 線の右側に配置
                            
                            # 右端で見切れる場合は左側に寄せる？などは一旦省略
                            painter.setPen(QColor(220, 220, 220))
                            painter.drawText(text_x, 14, time_str)
                            
                        else:
                            # マイナー目盛り
                            painter.setPen(QPen(QColor(120, 120, 120), 1))
                            painter.drawLine(x, ruler_height - 5, x, ruler_height)
                    
                    t += minor_step

        # 1.5. 選択中トラックの強調表示（背景）
        if self._track_frames:
            painter.setPen(Qt.NoPen)
            # 薄いオレンジ色で範囲を表示
            start_f = self._track_frames[0]
            end_f = self._track_frames[-1]
            x1 = val_to_x(start_f)
            x2 = val_to_x(end_f)
            ruler_height = 20
            painter.fillRect(x1, ruler_height, x2 - x1, rect.height() - ruler_height, QColor(255, 100, 0, 30))
            
            # 各フレームにもマーカーを付ける
            painter.setBrush(QColor(255, 150, 0, 150))
            for f in self._track_frames:
                x = val_to_x(f)
                if x < rect.width():
                     painter.drawRect(x, rect.height() - 5, 2, 5)



        # 2. アノテーションマーカーの描画
        if self._annotations:
            # 緑色の少し幅のある線
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 200, 100, 180))
            
            for frame_num in self._annotations.keys():
                if self._annotations[frame_num]:
                    x = val_to_x(frame_num)
                    # 以前は画面外のアノテーションを端に寄せていたが、ズーム機能追加により
                    # 画面外は描画しないように修正（緑線が張り付く問題を解消）
                    if 0 <= x < rect.width():
                        ruler_height = 20
                        painter.drawRect(x, ruler_height, 2, rect.height() - ruler_height)



        # 3. 選択範囲の描画
        selection = self.get_selection()
        if selection:
            start, end = selection
            start_x = val_to_x(start)
            end_x = val_to_x(end)
            
            if start_x == end_x:
                end_x += 1
                
            # 半透明の青色で塗りつぶし (前面に表示)
            fill_color = QColor(0, 120, 255, 100)
            ruler_height = 20
            painter.fillRect(
                start_x, 
                ruler_height, 
                end_x - start_x, 
                rect.height() - ruler_height, 
                fill_color
            )
            
            # 枠線
            border_pen = QPen(QColor(0, 180, 255), 1)
            painter.setPen(border_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                start_x,
                ruler_height,
                end_x - start_x,
                rect.height() - ruler_height - 1 
            )

        # 3.5. サムネイルの描画（アノテーション、選択範囲の上に描画）
        for frame, img in self._track_thumbnails:
            x = val_to_x(frame)
            if 0 <= x < rect.width():
                # アスペクト比を維持して高さ30pxくらいにリサイズ（下半分）
                thumbs_h = 30
                scaled_img = img.scaledToHeight(thumbs_h, Qt.SmoothTransformation)
                w = scaled_img.width()
                # タイムラインの下端に描画
                y = rect.height() - scaled_img.height()
                
                # 画像が重ならないように調整...は難しいので単純描画
                # 左側にはみ出さないように
                draw_x = x - w // 2
                painter.drawImage(draw_x, y, scaled_img)
                
                # 枠線を描画（白）
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(draw_x, y, w, thumbs_h - 1)
                
                # 出展元を示す線
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
                painter.drawLine(x, rect.height() - thumbs_h, x, rect.height())

        # 4. 現在位置（プレイヘッド）の描画
        current_val = self.value()
        cx = val_to_x(current_val)
        
        # 赤い縦線
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawLine(cx, 0, cx, rect.height())
        
        # ヘッド（三角形）
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 50, 50))
        
        # 上の三角形
        painter.drawPolygon([
             QPoint(cx - 6, 0),
             QPoint(cx + 6, 0),
             QPoint(cx, 10)
        ])

        painter.end()

    def _pixel_to_value(self, x: int) -> int:
        """ピクセル座標を値（フレーム番号）に変換"""
        rect = self.rect()
        if rect.width() <= 0:
            return 0
            
        x = max(0, min(x, rect.width()))
        normalized_pos = x / rect.width()
        
        val_range = self.maximum() - self.minimum()
        visible_range = val_range / self._zoom
        
        # offset + normalized * visible_range
        return int(max(0, min(self._zoom_offset + normalized_pos * visible_range, val_range)))


class TimelineWidget(QWidget):
    """タイムラインコントロールウィジェット"""

    frame_changed = pyqtSignal(int)  # スライダーからのフレーム変更
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    # 範囲選択関連シグナル
    range_selected = pyqtSignal(int, int)
    delete_range_triggered = pyqtSignal(int, int)
    detect_range_triggered = pyqtSignal(int, int)
    
    # 表示範囲変更シグナル（サムネイル生成用）
    visible_range_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_count = 0
        self._fps = 30.0
        self._is_playing = False
        
        # ズーム状態
        self._zoom = 1.0
        
        # 自動スクロール判定用
        self._last_frame = 0  # 直前のフレーム番号
        
        self._setup_ui()

    def set_duration(self, duration: int) -> None:
        """期間設定（フレーム数を更新）"""
        # 互換性のため（使用箇所があるか確認）
        # self._frame_count = duration
        pass

    def _setup_ui(self) -> None:
        """UIを構築"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)

        # コントロール行
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        # 再生コントロールボタン
        self._play_button = QPushButton("▶")
        self._play_button.setFixedSize(40, 30)
        self._play_button.clicked.connect(self._on_play_clicked)
        controls_layout.addWidget(self._play_button)

        self._stop_button = QPushButton("■")
        self._stop_button.setFixedSize(40, 30)
        self._stop_button.clicked.connect(self.stop_clicked.emit)
        controls_layout.addWidget(self._stop_button)

        # フレーム移動ボタン
        self._prev_button = QPushButton("◀◀")
        self._prev_button.setFixedSize(40, 30)
        self._prev_button.setToolTip("10フレーム戻る")
        self._prev_button.clicked.connect(lambda: self._step(-10))
        controls_layout.addWidget(self._prev_button)

        self._prev_frame_button = QPushButton("◀")
        self._prev_frame_button.setFixedSize(30, 30)
        self._prev_frame_button.setToolTip("1フレーム戻る")
        self._prev_frame_button.clicked.connect(lambda: self._step(-1))
        controls_layout.addWidget(self._prev_frame_button)

        self._next_frame_button = QPushButton("▶")
        self._next_frame_button.setFixedSize(30, 30)
        self._next_frame_button.setToolTip("1フレーム進む")
        self._next_frame_button.clicked.connect(lambda: self._step(1))
        controls_layout.addWidget(self._next_frame_button)

        self._next_button = QPushButton("▶▶")
        self._next_button.setFixedSize(40, 30)
        self._next_button.setToolTip("10フレーム進む")
        self._next_button.clicked.connect(lambda: self._step(10))
        controls_layout.addWidget(self._next_button)

        controls_layout.addStretch()

        # 時間表示
        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        controls_layout.addWidget(self._time_label)

        # フレーム番号表示
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        controls_layout.addWidget(self._frame_label)

        layout.addLayout(controls_layout)

        # タイムラインスライダー
        self._slider = TimelineSlider()
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.selection_changed.connect(self.range_selected.emit)
        
        # コンテキストメニューポリシーを設定
        self._slider.setContextMenuPolicy(Qt.CustomContextMenu)
        self._slider.customContextMenuRequested.connect(self._show_context_menu)
        
        layout.addWidget(self._slider)
        
        # 水平スクロールバー
        self._scrollbar = QScrollBar(Qt.Horizontal)
        self._scrollbar.setRange(0, 0)
        self._scrollbar.setPageStep(100)
        self._scrollbar.valueChanged.connect(self._on_scroll_changed)
        
        # スクロールバーのスタイル
        self._scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                border: none;
                background: #2b2b2b;
                height: 14px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:horizontal {
                background: #555;
                min-width: 20px;
                border-radius: 7px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #666;
            }
            QScrollBar::add-line:horizontal {
                width: 0px;
            }
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)
        layout.addWidget(self._scrollbar)
        
        # ズーム倍率表示
        self._zoom_label = QLabel("x1.0")
        self._zoom_label.setStyleSheet("color: gray; font-size: 10px;")
        # 時間表示の横に追加したいが、layout構成変えるのは手間なので
        # とりあえずスクロールバーの下か横に... ここではスクロールバーと同じ行には入らないので
        # コントロール行に追加するのが自然か。
        controls_layout.addWidget(self._zoom_label) # addStretchの後に追加される


        self._slider_dragging = False

    def set_video_info(self, frame_count: int, fps: float) -> None:
        """動画情報を設定"""
        self._frame_count = frame_count
        self._fps = fps if fps > 0 else 30.0
        self._slider.set_fps(self._fps)
        self._slider.setMaximum(max(0, frame_count - 1))
        self._slider.set_selection(None, None)  # 選択をリセット
        
        # ズームリセット
        self._zoom = 1.0
        self._last_frame = 0
        self._update_zoom_view()
        
        self._update_labels(0)

    def set_frame(self, frame_number: int) -> None:
        """現在のフレーム番号を設定（外部からの更新用）"""
        if not self._slider_dragging:
            self._slider.blockSignals(True)
            self._slider.setValue(frame_number)
            self._slider.blockSignals(False)
        
        # 自動スクロール判定
        self._check_auto_scroll(frame_number)
        self._last_frame = frame_number

        self._update_labels(frame_number)

    def set_playing(self, is_playing: bool) -> None:
        """再生状態を設定"""
        self._is_playing = is_playing
        self._play_button.setText("⏸" if is_playing else "▶")

    def set_annotations(self, annotations: dict[int, list]) -> None:
        """アノテーション位置を設定"""
        self._slider.set_annotations(annotations)

    def set_selected_track(self, frames: list[int], thumbnails: list[tuple[int, QImage]] = None) -> None:
        """選択中のトラック情報を設定"""
        self._slider.set_selected_track(frames, thumbnails)

    def _update_labels(self, frame_number: int) -> None:
        """ラベルを更新"""
        current_time = frame_number / self._fps if self._fps > 0 else 0
        total_time = self._frame_count / self._fps if self._fps > 0 else 0

        self._time_label.setText(
            f"{format_time(current_time)} / {format_time(total_time)}"
        )
        self._frame_label.setText(f"Frame: {frame_number} / {self._frame_count}")

    def _on_play_clicked(self) -> None:
        """再生ボタンクリック"""
        if self._is_playing:
            self.pause_clicked.emit()
        else:
            self.play_clicked.emit()

    def _on_slider_changed(self, value: int) -> None:
        """スライダー値変更"""
        self._update_labels(value)
        
        # 自動スクロール判定
        self._check_auto_scroll(value)
        self._last_frame = value
        
        # 以前はドラッグ中のみ発火していたが、キー操作やホイール操作（有効な場合）でも
        # 反応するように変更。プログラマティックな変更は blockSignals で防ぐこと。
        self.frame_changed.emit(value)

    def _on_slider_pressed(self) -> None:
        """スライダードラッグ開始"""
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        """スライダードラッグ終了"""
        self._slider_dragging = False
        self.frame_changed.emit(self._slider.value())

    def _step(self, frames: int) -> None:
        """フレームを移動"""
        new_frame = max(0, min(self._slider.value() + frames, self._frame_count - 1))
        self.frame_changed.emit(new_frame)

    def _show_context_menu(self, pos) -> None:
        """コンテキストメニューを表示"""
        selection = self._slider.get_selection()
        if not selection:
            return

        start, end = selection
        menu = QMenu(self)
        
        label_action = QAction(f"選択範囲: {start} - {end}", self)
        label_action.setEnabled(False)
        menu.addAction(label_action)
        menu.addSeparator()

        delete_action = QAction("この範囲のアノテーションを削除", self)
        delete_action.triggered.connect(lambda: self.delete_range_triggered.emit(start, end))
        menu.addAction(delete_action)

        detect_action = QAction("この範囲で顔検出を実行...", self)
        detect_action.triggered.connect(lambda: self.detect_range_triggered.emit(start, end))
        menu.addAction(detect_action)
        
        # スライダー座標系での表示
        menu.exec_(self._slider.mapToGlobal(pos))

    def _update_zoom_view(self) -> None:
        """ズームとスクロールバーの状態を更新"""
        if self._frame_count <= 0:
            return

        visible_range = self._frame_count / self._zoom
        
        # スクロールバーの設定
        # 最大値 = 全体 - 表示範囲
        max_scroll = max(0, int(self._frame_count - visible_range))
        page_step = int(visible_range)
        
        self._scrollbar.setRange(0, max_scroll)
        self._scrollbar.setPageStep(page_step)
        
        # 現在のオフセットを取得（範囲内に収める）
        offset = min(self._scrollbar.value(), max_scroll)
        self._scrollbar.setValue(offset)
        
        # スライダーに反映
        self._slider.set_zoom_view(self._zoom, offset)
        self._zoom_label.setText(f"x{self._zoom:.1f}")
        
        # シグナル発火
        self.visible_range_changed.emit()

    def _check_auto_scroll(self, new_frame: int) -> None:
        """自動スクロール判定と実行"""
        if self._frame_count <= 0:
            return

        visible_range = self._frame_count / self._zoom
        start_visible = self._scrollbar.value()
        end_visible = start_visible + visible_range
        
        # 直前のフレームが表示範囲内にあったか？
        was_visible = (start_visible <= self._last_frame <= end_visible)
        
        if not was_visible:
            return
            
        # 連続的な追従（スムーズスクロール）
        # 右端の20%エリアに入ったらスクロールを開始して、その相対位置をキープする
        margin_ratio = 0.2
        margin = visible_range * margin_ratio
        
        # 閾値（これを超えたらスクロール）
        right_threshold = end_visible - margin
        left_threshold = start_visible + margin
        
        if new_frame > right_threshold:
            # 右側の閾値を超えた -> 超えた分だけスクロールして追従
            # つまり、new_frame がちょうど right_threshold の位置に来るようにする
            # new_frame = (new_start + visible_range) - margin
            # new_start = new_frame - visible_range + margin
            target = new_frame - visible_range + margin
            self._scrollbar.setValue(int(target))
            
        elif new_frame < left_threshold:
            # 左側の閾値を超えた
            # new_frame = new_start + margin
            # new_start = new_frame - margin
            target = new_frame - margin
            self._scrollbar.setValue(int(target))

    def _on_scroll_changed(self, value: int) -> None:
        """スクロールバーの値変更"""
        self._slider.set_zoom_view(self._zoom, value)
        self.visible_range_changed.emit()

    def wheelEvent(self, event) -> None:
        """マウスホイールイベント（ズーム・スクロール）"""
        modifiers = event.modifiers()
        
        if modifiers & Qt.ControlModifier:
            # ズーム (Ctrl + Wheel)
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom *= 1.1
            else:
                self._zoom /= 1.1
            
            # 制限
            self._zoom = max(1.0, min(self._zoom, 50.0))
            
            # マウス位置を中心にズームしたいが、簡易実装として中心ズームまたは左端維持
            # ここではシンプルに更新
            self._update_zoom_view()
            event.accept()
            
        elif modifiers & Qt.ShiftModifier:
            # Shift+Wheelは通常水平スクロールだが、ここではスクロールバー操作に割り当て
             delta = event.angleDelta().y()
             step = self._scrollbar.pageStep() // 10
             if delta > 0:
                 self._scrollbar.setValue(self._scrollbar.value() - step)
             else:
                 self._scrollbar.setValue(self._scrollbar.value() + step)
             event.accept()
        else:
             # 通常スクロールも許可
             delta = event.angleDelta().y()
             step = self._scrollbar.pageStep() // 5
             if delta > 0:
                 self._scrollbar.setValue(self._scrollbar.value() - step)
             else:
                 self._scrollbar.setValue(self._scrollbar.value() + step)
             event.accept()

    def get_visible_range(self) -> tuple[float, float]:
        """現在の表示範囲（開始フレーム、終了フレーム）を取得"""
        if self._frame_count <= 0:
            return (0.0, 0.0)
        
        visible_range = self._frame_count / self._zoom
        start = self._scrollbar.value()
        end = start + visible_range
        return (start, end)
    
    def resizeEvent(self, event) -> None:
        """リサイズイベント"""
        super().resizeEvent(event)
        self._update_zoom_view()
        self.visible_range_changed.emit()

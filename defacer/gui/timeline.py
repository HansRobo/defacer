"""タイムラインウィジェット"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
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
)


def format_time(seconds: float) -> str:
    """秒数を MM:SS.mmm 形式に変換"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


class TimelineSlider(QSlider):
    """カスタムタイムラインスライダー"""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(0)
        self._annotations: dict[int, list] = {}  # frame_number -> annotations

    def set_annotations(self, annotations: dict[int, list]) -> None:
        """アノテーション位置を設定（タイムライン上にマーカー表示用）"""
        self._annotations = annotations
        self.update()

    def paintEvent(self, event) -> None:
        """カスタム描画"""
        super().paintEvent(event)

        if not self._annotations or self.maximum() == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # スライダーの有効範囲を取得
        # QStyleOptionSliderインスタンスを作成
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # 正しい引数でsubControlRectを呼び出し
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        # アノテーションがあるフレームにマーカーを描画
        pen = QPen(QColor(0, 200, 0), 2)
        painter.setPen(pen)

        for frame_num in self._annotations.keys():
            if self._annotations[frame_num]:  # 空でないアノテーションのみ
                x = int(
                    groove_rect.x()
                    + (frame_num / self.maximum()) * groove_rect.width()
                )
                painter.drawLine(x, groove_rect.y(), x, groove_rect.y() + groove_rect.height())

        painter.end()


class TimelineWidget(QWidget):
    """タイムラインコントロールウィジェット"""

    frame_changed = pyqtSignal(int)  # スライダーからのフレーム変更
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_count = 0
        self._fps = 30.0
        self._is_playing = False
        self._setup_ui()

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
        layout.addWidget(self._slider)

        self._slider_dragging = False

    def set_video_info(self, frame_count: int, fps: float) -> None:
        """動画情報を設定"""
        self._frame_count = frame_count
        self._fps = fps if fps > 0 else 30.0
        self._slider.setMaximum(max(0, frame_count - 1))
        self._update_labels(0)

    def set_frame(self, frame_number: int) -> None:
        """現在のフレーム番号を設定（外部からの更新用）"""
        if not self._slider_dragging:
            self._slider.blockSignals(True)
            self._slider.setValue(frame_number)
            self._slider.blockSignals(False)
        self._update_labels(frame_number)

    def set_playing(self, is_playing: bool) -> None:
        """再生状態を設定"""
        self._is_playing = is_playing
        self._play_button.setText("⏸" if is_playing else "▶")

    def set_annotations(self, annotations: dict[int, list]) -> None:
        """アノテーション位置を設定"""
        self._slider.set_annotations(annotations)

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
        if self._slider_dragging:
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

"""フレーム範囲選択ウィジェット"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QRadioButton,
)


class FrameRangeSelector(QGroupBox):
    """フレーム範囲選択ウィジェット（全フレーム / カスタム範囲）

    Args:
        frame_count: 総フレーム数
        current_frame: 現在のフレーム番号
        include_current_frame_option: 「現在のフレームのみ」オプションを表示するか
        initial_range: カスタム範囲の初期値 (start, end)。指定した場合はカスタム範囲を選択
        title: グループボックスのタイトル
    """

    range_changed = pyqtSignal()

    def __init__(
        self,
        frame_count: int,
        current_frame: int = 0,
        include_current_frame_option: bool = False,
        initial_range: tuple[int, int] | None = None,
        title: str = "範囲",
        parent=None,
    ):
        super().__init__(title, parent)
        self._frame_count = frame_count
        self._current_frame = current_frame
        self._include_current = include_current_frame_option
        self._range_current: QRadioButton | None = None

        self._setup_ui(initial_range)

    def _setup_ui(self, initial_range: tuple[int, int] | None) -> None:
        layout = QVBoxLayout(self)

        self._range_all = QRadioButton("全フレーム")
        if self._include_current:
            self._range_current = QRadioButton("現在のフレームのみ")
        self._range_custom = QRadioButton("範囲を指定")

        if initial_range:
            self._range_custom.setChecked(True)
        else:
            self._range_all.setChecked(True)

        layout.addWidget(self._range_all)
        if self._range_current is not None:
            layout.addWidget(self._range_current)
        layout.addWidget(self._range_custom)

        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("開始:"))
        self._start_spin = QSpinBox()
        self._start_spin.setRange(0, self._frame_count - 1)
        self._start_spin.setValue(initial_range[0] if initial_range else 0)
        custom_layout.addWidget(self._start_spin)

        custom_layout.addWidget(QLabel("終了:"))
        self._end_spin = QSpinBox()
        self._end_spin.setRange(0, self._frame_count - 1)
        self._end_spin.setValue(initial_range[1] if initial_range else self._frame_count - 1)
        custom_layout.addWidget(self._end_spin)
        custom_layout.addStretch()
        layout.addLayout(custom_layout)

        self._range_all.toggled.connect(self._on_radio_changed)
        self._range_custom.toggled.connect(self._on_radio_changed)
        if self._range_current is not None:
            self._range_current.toggled.connect(self._on_radio_changed)

        self._on_radio_changed()

    def _on_radio_changed(self) -> None:
        is_custom = self._range_custom.isChecked()
        self._start_spin.setEnabled(is_custom)
        self._end_spin.setEnabled(is_custom)

        if self._range_current is not None and self._range_current.isChecked():
            self._start_spin.setValue(self._current_frame)
            self._end_spin.setValue(self._current_frame)
        elif self._range_all.isChecked():
            self._start_spin.setValue(0)
            self._end_spin.setValue(self._frame_count - 1)

        self.range_changed.emit()

    @property
    def start_frame(self) -> int:
        return self._start_spin.value()

    @property
    def end_frame(self) -> int:
        return self._end_spin.value()

    def validate(self) -> str | None:
        """バリデーション。エラーがあればメッセージ文字列、なければNoneを返す"""
        if self.start_frame > self.end_frame:
            return "開始フレームが終了フレームより大きいです"
        return None

    def is_current_frame_only(self) -> bool:
        """「現在のフレームのみ」が選択されているか"""
        return self._range_current is not None and self._range_current.isChecked()

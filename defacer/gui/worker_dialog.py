"""ワーカースレッドダイアログ基底クラス"""

from PyQt5.QtCore import QThread
from defacer.gui.styles import ERROR_STYLE
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
)


class WorkerDialog(QDialog):
    """バックグラウンドワーカーを持つダイアログの基底クラス

    サブクラスの責務:
    - _setup_ui() の中で _add_progress_widgets() と _add_button_row() を呼ぶ
    - ワーカーの finished シグナルを接続し、ハンドラ内で _finish_worker() を呼ぶ
    - ワーカーを起動するときは _start_worker() を呼ぶ
    """

    def __init__(self, parent, title: str, min_width: int = 450):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        self.setModal(True)
        self._worker: QThread | None = None
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._status_label = QLabel("")
        self._action_btn: QPushButton | None = None

    def _add_progress_widgets(self, layout: QVBoxLayout) -> None:
        """進捗バーとステータスラベルをレイアウトに追加"""
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._status_label)

    def _add_button_row(
        self, layout: QVBoxLayout, action_text: str, action_callback
    ) -> None:
        """キャンセル + アクションボタン行をレイアウトに追加"""
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_btn = QPushButton("キャンセル")
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)

        self._action_btn = QPushButton(action_text)
        self._action_btn.clicked.connect(action_callback)
        button_layout.addWidget(self._action_btn)

        layout.addLayout(button_layout)

    def _start_worker(self, worker: QThread) -> None:
        """ワーカーを起動（ボタン無効化・進捗表示）"""
        self._worker = worker
        if self._action_btn:
            self._action_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        worker.progress.connect(self._on_progress)
        worker.start()

    def _finish_worker(self, success: bool) -> None:
        """ワーカー完了処理（ボタン再有効化・進捗非表示）"""
        if self._action_btn:
            self._action_btn.setEnabled(True)
        self._progress_bar.setVisible(False)

    def _on_progress(self, current: int, total: int) -> None:
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _show_error(self, message: str) -> None:
        self._status_label.setText(f"エラー: {message}")
        self._status_label.setStyleSheet(ERROR_STYLE)

    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            if hasattr(self._worker, "cancel"):
                self._worker.cancel()
            else:
                self._worker.terminate()
            self._worker.wait()
        self.reject()

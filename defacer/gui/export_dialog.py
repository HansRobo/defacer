"""エクスポートダイアログ"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QProgressBar,
    QFileDialog,
    QGroupBox,
    QMessageBox,
)

from defacer.gui.annotation import AnnotationStore
from defacer.anonymization import create_anonymizer
from defacer.pipeline.processor import export_processed_video, ExportConfig
from defacer.video.writer import check_ffmpeg_available
from defacer.gui.worker_dialog import WorkerDialog


class ExportWorker(QThread):
    """エクスポート処理を行うワーカースレッド"""

    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        annotations: AnnotationStore,
        config: ExportConfig,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.annotations = annotations
        self.config = config

    def run(self):
        try:
            success = export_processed_video(
                self.input_path,
                self.output_path,
                self.annotations,
                self.config,
                lambda current, total: self.progress.emit(current, total),
            )
            if success:
                self.finished.emit(True, "エクスポートが完了しました")
            else:
                self.finished.emit(False, "エクスポートに失敗しました")
        except Exception as e:
            self.finished.emit(False, str(e))


class ExportDialog(WorkerDialog):
    """エクスポート設定ダイアログ"""

    def __init__(
        self,
        parent,
        input_path: Path,
        annotations: AnnotationStore,
    ):
        super().__init__(parent, "動画をエクスポート", min_width=500)
        self.input_path = input_path
        self.annotations = annotations

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 出力ファイル
        output_group = QGroupBox("出力ファイル")
        output_layout = QHBoxLayout(output_group)

        self._output_path = QLineEdit()
        default_output = self.input_path.with_stem(self.input_path.stem + "_defaced")
        self._output_path.setText(str(default_output))

        browse_btn = QPushButton("参照...")
        browse_btn.clicked.connect(self._browse_output)

        output_layout.addWidget(self._output_path)
        output_layout.addWidget(browse_btn)
        layout.addWidget(output_group)

        # モザイク設定
        mosaic_group = QGroupBox("モザイク設定")
        mosaic_layout = QVBoxLayout(mosaic_group)

        # モザイクタイプ
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("タイプ:"))
        self._mosaic_type = QComboBox()
        self._mosaic_type.addItems(["モザイク", "ぼかし", "塗りつぶし"])
        self._mosaic_type.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self._mosaic_type)
        type_layout.addStretch()
        mosaic_layout.addLayout(type_layout)

        # モザイクブロックサイズ
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("ブロックサイズ:"))
        self._block_size = QSpinBox()
        self._block_size.setRange(5, 50)
        self._block_size.setValue(10)
        block_layout.addWidget(self._block_size)
        block_layout.addStretch()
        mosaic_layout.addLayout(block_layout)
        self._block_size_label = block_layout.itemAt(0).widget()

        # ぼかしカーネルサイズ
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("ぼかし強度:"))
        self._blur_kernel = QSpinBox()
        self._blur_kernel.setRange(11, 199)
        self._blur_kernel.setSingleStep(2)
        self._blur_kernel.setValue(99)
        blur_layout.addWidget(self._blur_kernel)
        blur_layout.addStretch()
        mosaic_layout.addLayout(blur_layout)
        self._blur_kernel_label = blur_layout.itemAt(0).widget()
        self._blur_kernel.setVisible(False)
        self._blur_kernel_label.setVisible(False)

        # 楕円形マスク
        self._ellipse_check = QCheckBox("楕円形マスクを使用")
        self._ellipse_check.setChecked(True)
        mosaic_layout.addWidget(self._ellipse_check)

        # バウンディングボックス拡大
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("領域拡大率:"))
        self._bbox_scale = QSpinBox()
        self._bbox_scale.setRange(100, 150)
        self._bbox_scale.setSuffix("%")
        self._bbox_scale.setValue(110)
        scale_layout.addWidget(self._bbox_scale)
        scale_layout.addStretch()
        mosaic_layout.addLayout(scale_layout)

        # フレーム間自動補間
        self._auto_interpolate = QCheckBox("フレーム間を自動補間")
        self._auto_interpolate.setChecked(True)
        self._auto_interpolate.setToolTip(
            "エクスポート時にアノテーション間を自動的に線形補間します。\n"
            "これにより、手動アノテーションがないフレームでも顔が匿名化されます。"
        )
        mosaic_layout.addWidget(self._auto_interpolate)

        layout.addWidget(mosaic_group)

        # エンコード設定
        encode_group = QGroupBox("エンコード設定")
        encode_layout = QVBoxLayout(encode_group)

        # 品質
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("品質 (CRF):"))
        self._crf = QSpinBox()
        self._crf.setRange(0, 51)
        self._crf.setValue(18)
        self._crf.setToolTip("0=最高品質/最大サイズ, 51=最低品質/最小サイズ, 推奨: 18-23")
        quality_layout.addWidget(self._crf)
        quality_layout.addStretch()
        encode_layout.addLayout(quality_layout)

        # プリセット
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("速度:"))
        self._preset = QComboBox()
        self._preset.addItems(["ultrafast", "fast", "medium", "slow", "veryslow"])
        self._preset.setCurrentText("medium")
        self._preset.setToolTip("遅いほど圧縮効率が良い")
        preset_layout.addWidget(self._preset)
        preset_layout.addStretch()
        encode_layout.addLayout(preset_layout)

        layout.addWidget(encode_group)

        self._add_progress_widgets(layout)
        self._add_button_row(layout, "エクスポート", self._start_export)

        if not check_ffmpeg_available():
            self._action_btn.setEnabled(False)
            self._show_error("FFmpegが見つかりません")

    def _browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "出力ファイルを選択",
            self._output_path.text(),
            "MP4ファイル (*.mp4);;すべてのファイル (*)",
        )
        if file_path:
            self._output_path.setText(file_path)

    def _on_type_changed(self, index):
        is_mosaic = index == 0
        is_blur = index == 1

        self._block_size.setVisible(is_mosaic)
        self._block_size_label.setVisible(is_mosaic)
        self._blur_kernel.setVisible(is_blur)
        self._blur_kernel_label.setVisible(is_blur)

    def _create_anonymizer(self):
        types = ["mosaic", "blur", "solid"]
        return create_anonymizer(
            types[self._mosaic_type.currentIndex()],
            block_size=self._block_size.value(),
            kernel_size=self._blur_kernel.value(),
        )

    def _start_export(self):
        output_path = Path(self._output_path.text())

        if output_path.exists():
            reply = QMessageBox.question(
                self,
                "確認",
                f"ファイルが既に存在します:\n{output_path}\n\n上書きしますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._status_label.setText("エクスポート中...")

        config = ExportConfig(
            anonymizer=self._create_anonymizer(),
            ellipse=self._ellipse_check.isChecked(),
            bbox_scale=self._bbox_scale.value() / 100.0,
            interpolate=self._auto_interpolate.isChecked(),
            crf=self._crf.value(),
            preset=self._preset.currentText(),
        )
        worker = ExportWorker(self.input_path, output_path, self.annotations, config)
        worker.finished.connect(self._on_finished)
        self._start_worker(worker)

    def _on_progress(self, current, total):
        super()._on_progress(current, total)
        self._status_label.setText(f"処理中: {current}/{total} フレーム")

    def _on_finished(self, success, message):
        self._finish_worker(success)
        if success:
            self._status_label.setText(message)
            QMessageBox.information(self, "完了", message)
            self.accept()
        else:
            self._show_error(message)

    def _on_cancel(self):
        """エクスポート中は確認ダイアログを表示"""
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "確認", "エクスポートを中止しますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        super()._on_cancel()

    def reject(self):
        self._on_cancel()

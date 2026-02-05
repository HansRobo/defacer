"""自動検出ダイアログ"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
)

from defacer.detection import get_available_detectors, create_detector
from defacer.video.reader import VideoReader
from defacer.gui.annotation import AnnotationStore, Annotation, BoundingBox


class DetectionWorker(QThread):
    """検出処理を行うワーカースレッド

    ROI (Region of Interest) に対応しており、roi パラメータを指定すると
    その領域内のみで検出を行い、結果を元のフレーム座標系に変換します。
    """

    progress = pyqtSignal(int, int)  # (current, total)
    detection_found = pyqtSignal(int, list)  # (frame_number, detections)
    finished = pyqtSignal(bool, str, int)  # (success, message, detection_count)

    def __init__(
        self,
        video_path: Path,
        detector_type: str,
        confidence_threshold: float,
        start_frame: int,
        end_frame: int,
        frame_skip: int,
        bbox_scale: float,
        roi: BoundingBox | None = None,
    ):
        super().__init__()
        self.video_path = video_path
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = frame_skip
        self.bbox_scale = bbox_scale
        self.roi = roi
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            detector = create_detector(
                self.detector_type,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as e:
            self.finished.emit(False, f"検出器の初期化に失敗: {e}", 0)
            return

        try:
            reader = VideoReader(self.video_path)
        except Exception as e:
            self.finished.emit(False, f"動画の読み込みに失敗: {e}", 0)
            return

        total_frames = self.end_frame - self.start_frame + 1
        frames_to_process = total_frames // (self.frame_skip + 1)
        detection_count = 0

        try:
            current = 0
            for frame_num in range(self.start_frame, self.end_frame + 1, self.frame_skip + 1):
                if self._cancelled:
                    self.finished.emit(False, "キャンセルされました", detection_count)
                    return

                frame = reader.read_frame(frame_num)
                if frame is None:
                    continue

                # ROI領域のクロップ処理
                if self.roi is not None:
                    roi_x1 = max(0, int(self.roi.x1))
                    roi_y1 = max(0, int(self.roi.y1))
                    roi_x2 = min(frame.shape[1], int(self.roi.x2))
                    roi_y2 = min(frame.shape[0], int(self.roi.y2))

                    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
                        continue

                    cropped_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    detection_frame = cropped_frame
                else:
                    detection_frame = frame
                    roi_x1 = roi_y1 = 0

                # 純粋な検出のみ（トラッキングなし）
                detections = detector.detect(detection_frame)

                # ROI使用時は座標を補正
                if self.roi is not None and detections:
                    from defacer.detection.base import Detection
                    adjusted_detections = []
                    for det in detections:
                        adjusted_bbox = (
                            det.x1 + roi_x1,
                            det.y1 + roi_y1,
                            det.x2 + roi_x1,
                            det.y2 + roi_y1,
                        )
                        adjusted_det = Detection(
                            bbox=adjusted_bbox,
                            confidence=det.confidence,
                            landmarks=det.landmarks,
                        )
                        adjusted_detections.append(adjusted_det)
                    detections = adjusted_detections

                if detections:
                    self.detection_found.emit(frame_num, detections)
                    detection_count += len(detections)

                current += 1
                self.progress.emit(current, frames_to_process)

            self.finished.emit(True, "検出が完了しました", detection_count)

        except Exception as e:
            self.finished.emit(False, f"検出中にエラー: {e}", detection_count)
        finally:
            reader.release()


class DetectionDialog(QDialog):
    """自動検出設定ダイアログ"""

    detections_ready = pyqtSignal(object)  # AnnotationStore


    def __init__(
        self,
        parent,
        video_path: Path,
        frame_count: int,
        current_frame: int,
        initial_range: tuple[int, int] | None = None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.frame_count = frame_count
        self.current_frame = current_frame
        self.initial_range = initial_range
        self._worker: DetectionWorker | None = None
        self._temp_store = AnnotationStore()
        self._next_track_id = 1

        self.setWindowTitle("自動顔検出")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 検出器設定
        detector_group = QGroupBox("検出器設定")
        detector_layout = QVBoxLayout(detector_group)

        # 検出器選択
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("検出器:"))
        self._detector_type = QComboBox()

        available = get_available_detectors()
        if available:
            self._detector_type.addItems(available)
            if "yolo11-face" in available:
                self._detector_type.setCurrentText("yolo11-face")
        else:
            self._detector_type.addItem("(利用可能な検出器がありません)")
            self._detector_type.setEnabled(False)

        type_layout.addWidget(self._detector_type)
        type_layout.addStretch()
        detector_layout.addLayout(type_layout)

        # 信頼度閾値
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("信頼度閾値:"))
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.1, 1.0)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(0.5)
        self._threshold.setToolTip("高いほど確実な顔のみ検出（推奨: 0.5）")
        threshold_layout.addWidget(self._threshold)
        threshold_layout.addStretch()
        detector_layout.addLayout(threshold_layout)

        # 領域拡大率
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("領域拡大率:"))
        self._bbox_scale = QSpinBox()
        self._bbox_scale.setRange(100, 150)
        self._bbox_scale.setSuffix("%")
        self._bbox_scale.setValue(110)
        self._bbox_scale.setToolTip("検出領域を拡大してモザイク漏れを防止")
        scale_layout.addWidget(self._bbox_scale)
        scale_layout.addStretch()
        detector_layout.addLayout(scale_layout)

        layout.addWidget(detector_group)

        # 範囲設定
        range_group = QGroupBox("検出範囲")
        range_layout = QVBoxLayout(range_group)

        # 範囲選択
        self._range_all = QRadioButton("全フレーム")
        self._range_current = QRadioButton("現在のフレームのみ")
        self._range_custom = QRadioButton("範囲を指定")

        if self.initial_range:
            self._range_custom.setChecked(True)
        else:
            self._range_all.setChecked(True)

        range_layout.addWidget(self._range_all)
        range_layout.addWidget(self._range_current)
        range_layout.addWidget(self._range_custom)

        # カスタム範囲
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("開始:"))
        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, self.frame_count - 1)
        
        start_val = 0
        if self.initial_range:
            start_val = self.initial_range[0]
        self._start_frame.setValue(start_val)
        
        custom_layout.addWidget(self._start_frame)

        custom_layout.addWidget(QLabel("終了:"))
        self._end_frame = QSpinBox()
        self._end_frame.setRange(0, self.frame_count - 1)
        
        end_val = self.frame_count - 1
        if self.initial_range:
            end_val = self.initial_range[1]
        self._end_frame.setValue(end_val)
        
        custom_layout.addWidget(self._end_frame)

        custom_layout.addStretch()
        range_layout.addLayout(custom_layout)

        # フレームスキップ
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("フレームスキップ:"))
        self._frame_skip = QSpinBox()
        self._frame_skip.setRange(0, 30)
        self._frame_skip.setValue(0)
        self._frame_skip.setToolTip("N=0で全フレーム、N=1で1フレームおきに検出")
        skip_layout.addWidget(self._frame_skip)
        skip_layout.addStretch()
        range_layout.addLayout(skip_layout)

        layout.addWidget(range_group)

        # 範囲選択の連動
        self._range_all.toggled.connect(self._on_range_changed)
        self._range_current.toggled.connect(self._on_range_changed)
        self._range_custom.toggled.connect(self._on_range_changed)
        self._on_range_changed()

        # 進捗
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # ボタン
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_btn = QPushButton("キャンセル")
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)

        self._detect_btn = QPushButton("検出開始")
        self._detect_btn.clicked.connect(self._start_detection)
        self._detect_btn.setEnabled(bool(available))
        button_layout.addWidget(self._detect_btn)

        layout.addLayout(button_layout)

    def _on_range_changed(self):
        """範囲選択が変更された時"""
        is_custom = self._range_custom.isChecked()
        self._start_frame.setEnabled(is_custom)
        self._end_frame.setEnabled(is_custom)
        self._frame_skip.setEnabled(not self._range_current.isChecked())

        if self._range_current.isChecked():
            self._start_frame.setValue(self.current_frame)
            self._end_frame.setValue(self.current_frame)
        elif self._range_all.isChecked():
            self._start_frame.setValue(0)
            self._end_frame.setValue(self.frame_count - 1)

    def _start_detection(self):
        """検出を開始"""
        if self._range_current.isChecked():
            start_frame = self.current_frame
            end_frame = self.current_frame
        elif self._range_custom.isChecked():
            start_frame = self._start_frame.value()
            end_frame = self._end_frame.value()
        else:
            start_frame = 0
            end_frame = self.frame_count - 1

        if start_frame > end_frame:
            QMessageBox.warning(self, "エラー", "開始フレームが終了フレームより大きいです")
            return

        self._detect_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("検出中...")
        self._temp_store = AnnotationStore()

        self._worker = DetectionWorker(
            self.video_path,
            self._detector_type.currentText(),
            self._threshold.value(),
            start_frame,
            end_frame,
            self._frame_skip.value(),
            self._bbox_scale.value() / 100.0,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.detection_found.connect(self._on_detection_found)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current, total):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _on_detection_found(self, frame_number, detections):
        """検出結果を受け取り、track_idを連番で割り当てる"""
        bbox_scale = self._bbox_scale.value() / 100.0

        for det in detections:
            # バウンディングボックスを拡大
            x1, y1, x2, y2 = det.bbox
            if bbox_scale != 1.0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                new_w = int((x2 - x1) * bbox_scale)
                new_h = int((y2 - y1) * bbox_scale)
                x1 = max(0, cx - new_w // 2)
                y1 = max(0, cy - new_h // 2)
                x2 = cx + new_w // 2
                y2 = cy + new_h // 2

            ann = Annotation(
                frame=frame_number,
                bbox=BoundingBox(x1, y1, x2, y2),
                track_id=self._next_track_id,
                is_manual=False,
                confidence=det.confidence,
            )
            self._temp_store.add(ann, save_undo=False)
            self._next_track_id += 1

    def _on_finished(self, success, message, detection_count):
        self._detect_btn.setEnabled(True)
        self._progress_bar.setVisible(False)

        if success:
            self._status_label.setText(f"完了: {detection_count}件の顔を検出")
            if detection_count > 0:
                self.detections_ready.emit(self._temp_store)
                QMessageBox.information(
                    self,
                    "検出完了",
                    f"{detection_count}件の顔を検出しました。\n"
                    "アノテーションとして追加されます。",
                )
            else:
                QMessageBox.information(
                    self,
                    "検出完了",
                    "顔は検出されませんでした。",
                )
            self.accept()
        else:
            self._status_label.setText(f"エラー: {message}")
            self._status_label.setStyleSheet("color: red;")

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self.reject()

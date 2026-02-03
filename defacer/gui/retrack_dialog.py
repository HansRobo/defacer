"""既存アノテーションの再トラッキングダイアログ"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QRadioButton,
)

from defacer.video.reader import VideoReader
from defacer.gui.annotation import AnnotationStore, Annotation
from defacer.detection.base import Detection


class RetrackWorker(QThread):
    """再トラッキング処理を行うワーカースレッド"""

    progress = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(bool, str, dict)  # (success, message, track_id_mapping)

    def __init__(
        self,
        video_path: Path,
        annotations: AnnotationStore,
        start_frame: int,
        end_frame: int,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        super().__init__()
        self.video_path = video_path
        self.annotations = annotations
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.max_age = max_age
        self.min_hits = min_hits
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _compute_iou(self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
        """IoU（Intersection over Union）を計算"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 交差領域
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # 結合領域
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _build_track_id_mapping(self, anns: list[Annotation], tracked: list) -> dict:
        """
        bbox位置でマッチングしてtrack_idマッピングを構築

        Args:
            anns: 現在のフレームのアノテーションリスト
            tracked: トラッキング結果リスト

        Returns:
            {(frame, ann_index): new_track_id} のマッピング
        """
        mapping = {}

        for idx, ann in enumerate(anns):
            best_match = None
            best_iou = 0.0

            for t in tracked:
                iou = self._compute_iou(ann.bbox.to_tuple(), t.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = t

            # IoUが0.5以上で一致とみなす
            if best_match and best_iou > 0.5:
                mapping[idx] = best_match.track_id

        return mapping

    def run(self):
        try:
            # トラッカー初期化
            from defacer.tracking import create_tracker
            tracker = create_tracker(max_age=self.max_age, min_hits=self.min_hits)
        except ModuleNotFoundError as e:
            if "torch" in str(e):
                self.finished.emit(
                    False,
                    "PyTorchがインストールされていません。\n\n"
                    "以下のコマンドでインストールしてください:\n"
                    "pip install torch torchvision\n\n"
                    "または、ROCm環境の場合:\n"
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2",
                    {}
                )
            else:
                self.finished.emit(False, f"トラッカーの初期化に失敗: {e}", {})
            return
        except Exception as e:
            self.finished.emit(False, f"トラッカーの初期化に失敗: {e}", {})
            return

        try:
            reader = VideoReader(self.video_path)
        except Exception as e:
            self.finished.emit(False, f"動画の読み込みに失敗: {e}", {})
            return

        # 処理対象フレームを取得
        all_frames = sorted(self.annotations.get_all_frames())
        frames_to_process = [f for f in all_frames if self.start_frame <= f <= self.end_frame]

        if not frames_to_process:
            self.finished.emit(True, "処理対象のフレームがありません", {})
            reader.release()
            return

        # 全体のマッピングを構築 {(frame, ann_index): new_track_id}
        global_mapping = {}

        try:
            for idx, frame_num in enumerate(frames_to_process):
                if self._cancelled:
                    self.finished.emit(False, "キャンセルされました", {})
                    return

                # フレーム読み込み
                frame = reader.read_frame(frame_num)
                if frame is None:
                    continue

                # 現在のフレームのアノテーションを取得
                anns = self.annotations.get_frame_annotations(frame_num)
                if not anns:
                    continue

                # Annotation -> Detection 変換
                detections = [
                    Detection(bbox=a.bbox.to_tuple(), confidence=a.confidence if a.confidence else 1.0)
                    for a in anns
                ]

                # DeepSORTでトラッキング
                tracked = tracker.update(detections, frame)

                # track_idマッピングを構築
                frame_mapping = self._build_track_id_mapping(anns, tracked)

                # グローバルマッピングに追加
                for ann_idx, new_track_id in frame_mapping.items():
                    key = (frame_num, ann_idx)
                    global_mapping[key] = new_track_id

                # 進捗を通知
                self.progress.emit(idx + 1, len(frames_to_process))

            self.finished.emit(True, "再トラッキングが完了しました", global_mapping)

        except Exception as e:
            self.finished.emit(False, f"再トラッキング中にエラー: {e}", {})
        finally:
            reader.release()


class RetrackDialog(QDialog):
    """再トラッキング設定ダイアログ"""

    retrack_completed = pyqtSignal()  # 完了通知

    def __init__(
        self,
        parent,
        video_path: Path,
        annotations: AnnotationStore,
        frame_count: int,
        current_frame: int,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.annotations = annotations
        self.frame_count = frame_count
        self.current_frame = current_frame
        self._worker: RetrackWorker | None = None

        self.setWindowTitle("既存アノテーションの再トラッキング")
        self.setMinimumWidth(450)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 範囲設定
        range_group = QGroupBox("再トラッキング範囲")
        range_layout = QVBoxLayout(range_group)

        # 範囲選択
        self._range_all = QRadioButton("全フレーム")
        self._range_all.setChecked(True)
        self._range_custom = QRadioButton("範囲を指定")

        range_layout.addWidget(self._range_all)
        range_layout.addWidget(self._range_custom)

        # カスタム範囲
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("開始:"))
        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, self.frame_count - 1)
        self._start_frame.setValue(0)
        custom_layout.addWidget(self._start_frame)

        custom_layout.addWidget(QLabel("終了:"))
        self._end_frame = QSpinBox()
        self._end_frame.setRange(0, self.frame_count - 1)
        self._end_frame.setValue(self.frame_count - 1)
        custom_layout.addWidget(self._end_frame)

        custom_layout.addStretch()
        range_layout.addLayout(custom_layout)

        layout.addWidget(range_group)

        # 範囲選択の連動
        self._range_all.toggled.connect(self._on_range_changed)
        self._range_custom.toggled.connect(self._on_range_changed)
        self._on_range_changed()

        # トラッキング設定
        tracking_group = QGroupBox("トラッキング設定")
        tracking_layout = QVBoxLayout(tracking_group)

        # max_age設定
        max_age_layout = QHBoxLayout()
        max_age_layout.addWidget(QLabel("最大追跡フレーム数:"))
        self._tracking_max_age = QSpinBox()
        self._tracking_max_age.setRange(10, 100)
        self._tracking_max_age.setValue(30)
        self._tracking_max_age.setToolTip("顔が見えなくなってから何フレーム追跡を続けるか")
        max_age_layout.addWidget(self._tracking_max_age)
        max_age_layout.addStretch()
        tracking_layout.addLayout(max_age_layout)

        # min_hits設定
        min_hits_layout = QHBoxLayout()
        min_hits_layout.addWidget(QLabel("確定までの検出回数:"))
        self._tracking_min_hits = QSpinBox()
        self._tracking_min_hits.setRange(1, 10)
        self._tracking_min_hits.setValue(3)
        self._tracking_min_hits.setToolTip("トラックを確定するまでに必要な連続検出回数")
        min_hits_layout.addWidget(self._tracking_min_hits)
        min_hits_layout.addStretch()
        tracking_layout.addLayout(min_hits_layout)

        layout.addWidget(tracking_group)

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

        self._retrack_btn = QPushButton("再トラッキング")
        self._retrack_btn.clicked.connect(self._start_retracking)
        button_layout.addWidget(self._retrack_btn)

        layout.addLayout(button_layout)

    def _on_range_changed(self):
        """範囲選択が変更された時"""
        is_custom = self._range_custom.isChecked()
        self._start_frame.setEnabled(is_custom)
        self._end_frame.setEnabled(is_custom)

        if self._range_all.isChecked():
            self._start_frame.setValue(0)
            self._end_frame.setValue(self.frame_count - 1)

    def _start_retracking(self):
        """再トラッキングを開始"""
        if self._range_custom.isChecked():
            start_frame = self._start_frame.value()
            end_frame = self._end_frame.value()
        else:
            start_frame = 0
            end_frame = self.frame_count - 1

        if start_frame > end_frame:
            QMessageBox.warning(self, "エラー", "開始フレームが終了フレームより大きいです")
            return

        # Undo状態を保存
        self.annotations._save_undo_state()

        self._retrack_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("再トラッキング中...")

        self._worker = RetrackWorker(
            self.video_path,
            self.annotations,
            start_frame,
            end_frame,
            max_age=self._tracking_max_age.value(),
            min_hits=self._tracking_min_hits.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current, total):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _on_finished(self, success, message, track_id_mapping):
        self._retrack_btn.setEnabled(True)
        self._progress_bar.setVisible(False)

        if success:
            # メインスレッドでtrack_idを更新
            updated_count = 0
            for (frame_num, ann_idx), new_track_id in track_id_mapping.items():
                anns = self.annotations.get_frame_annotations(frame_num)
                if ann_idx < len(anns):
                    ann = anns[ann_idx]
                    if ann.track_id != new_track_id:
                        ann.track_id = new_track_id
                        updated_count += 1

            self._status_label.setText(f"完了: {updated_count}件のtrack_idを更新")
            if updated_count > 0:
                self.retrack_completed.emit()
                QMessageBox.information(
                    self,
                    "再トラッキング完了",
                    f"{updated_count}件のアノテーションのtrack_idを更新しました。",
                )
            else:
                QMessageBox.information(
                    self,
                    "再トラッキング完了",
                    "更新されたtrack_idはありませんでした。",
                )
            self.accept()
        else:
            self._status_label.setText(f"エラー: {message}")
            self._status_label.setStyleSheet("color: red;")
            QMessageBox.critical(
                self,
                "再トラッキングエラー",
                message,
            )

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        self.reject()

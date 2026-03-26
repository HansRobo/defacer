"""既存アノテーションの再トラッキングダイアログ"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QComboBox,
)

from defacer.video.reader import VideoReader
from defacer.gui.annotation import AnnotationStore, Annotation
from defacer.detection.base import Detection, find_best_iou_match
from defacer.gui.worker_dialog import WorkerDialog
from defacer.gui.frame_range_selector import FrameRangeSelector


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
        tracker_type: str = "botsort",
        max_age: int = 30,
        min_hits: int = 3,
    ):
        super().__init__()
        self.video_path = video_path
        self.annotations = annotations
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.tracker_type = tracker_type
        self.max_age = max_age
        self.min_hits = min_hits
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _build_track_id_mapping(self, anns: list[Annotation], tracked: list) -> dict:
        """bbox位置でマッチングしてtrack_idマッピングを構築"""
        mapping = {}
        for idx, ann in enumerate(anns):
            best_match = find_best_iou_match(ann.bbox.to_tuple(), tracked, threshold=0.5)
            if best_match is not None:
                mapping[idx] = best_match.track_id
        return mapping

    def run(self):
        try:
            # トラッカー初期化
            from defacer.tracking import create_tracker
            tracker = create_tracker(
                tracker_type=self.tracker_type,
                max_age=self.max_age,
                min_hits=self.min_hits
            )
        except ModuleNotFoundError as e:
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

                # トラッキング実行
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


class RetrackDialog(WorkerDialog):
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
        super().__init__(parent, "既存アノテーションの再トラッキング")
        self.video_path = video_path
        self.annotations = annotations
        self.frame_count = frame_count
        self.current_frame = current_frame

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._range_selector = FrameRangeSelector(
            self.frame_count,
            self.current_frame,
            title="再トラッキング範囲",
        )
        layout.addWidget(self._range_selector)

        # トラッキング設定
        tracking_group = QGroupBox("トラッキング設定")
        tracking_layout = QVBoxLayout(tracking_group)

        # トラッカー選択
        tracker_layout = QHBoxLayout()
        tracker_layout.addWidget(QLabel("トラッカー:"))
        self._tracker_type = QComboBox()

        from defacer.tracking import get_available_trackers
        available_trackers = get_available_trackers()
        if available_trackers:
            self._tracker_type.addItems(available_trackers)
            if "botsort" in available_trackers:
                self._tracker_type.setCurrentText("botsort")
        else:
            self._tracker_type.addItem("(利用可能なトラッカーがありません)")
            self._tracker_type.setEnabled(False)

        tracker_layout.addWidget(self._tracker_type)
        tracker_layout.addStretch()
        tracking_layout.addLayout(tracker_layout)

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

        self._add_progress_widgets(layout)
        self._add_button_row(layout, "再トラッキング", self._start_retracking)

    def _start_retracking(self):
        """再トラッキングを開始"""
        error = self._range_selector.validate()
        if error:
            QMessageBox.warning(self, "エラー", error)
            return

        start_frame = self._range_selector.start_frame
        end_frame = self._range_selector.end_frame

        self.annotations._save_undo_state()
        self._status_label.setText("再トラッキング中...")

        worker = RetrackWorker(
            self.video_path,
            self.annotations,
            start_frame,
            end_frame,
            tracker_type=self._tracker_type.currentText(),
            max_age=self._tracking_max_age.value(),
            min_hits=self._tracking_min_hits.value(),
        )
        worker.finished.connect(self._on_finished)
        self._start_worker(worker)

    def _on_finished(self, success, message, track_id_mapping):
        self._finish_worker(success)

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

            # track_idを直接変更したのでキャッシュを再構築
            if updated_count > 0:
                self.annotations._rebuild_cache()

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
            self._show_error(message)
            QMessageBox.critical(self, "再トラッキングエラー", message)

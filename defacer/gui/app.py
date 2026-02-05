"""GUIメインアプリケーション"""

import os
import sys
from pathlib import Path

# OpenCVとPyQt5のQtプラグイン競合を回避
# PyQt5を先にインポートしてプラグインパスを確立
from PyQt5.QtCore import Qt, QCoreApplication

# cv2のQtプラグインパスを完全に除外
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ.pop("QT_PLUGIN_PATH", None)

# PyQt5のプラグインパスのみを使用
import cv2
cv2_dir = os.path.dirname(cv2.__file__)
cv2_plugins = os.path.join(cv2_dir, "qt", "plugins")

# cv2のpluginsディレクトリを無効化（PyQt5との競合を回避）
if os.path.exists(cv2_plugins):
    disabled_path = cv2_plugins + ".disabled"
    if not os.path.exists(disabled_path):
        try:
            os.rename(cv2_plugins, disabled_path)
        except (OSError, PermissionError):
            # 権限エラーの場合は警告のみ
            pass

# QCoreApplicationのライブラリパスからcv2を除外
current_paths = QCoreApplication.libraryPaths()
filtered_paths = [p for p in current_paths if "cv2" not in p]
QCoreApplication.setLibraryPaths(filtered_paths)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QAction,
    QActionGroup,
    QToolBar,
    QStatusBar,
    QSplitter,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QInputDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialog,
    QProgressDialog,
    QProgressBar,
)

from PyQt5.QtCore import QThread, pyqtSignal, QEventLoop

from defacer.gui.video_player import VideoPlayerWidget
from defacer.gui.timeline import TimelineWidget
from defacer.gui.annotation import Annotation, AnnotationStore

from defacer.gui.export_dialog import ExportDialog
from defacer.gui.detection_dialog import DetectionDialog
from defacer.gui.retrack_dialog import RetrackDialog
from defacer.tracking.interpolation import interpolate_track
from defacer.detection import get_available_detectors


class SimpleProgressDialog(QDialog):
    """シンプルで確実に動作するプログレスダイアログ"""

    def __init__(self, title: str, message: str, maximum: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setMinimumHeight(120)

        # レイアウト
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # メッセージラベル
        self._label = QLabel(message)
        layout.addWidget(self._label)

        # プログレスバー
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(maximum)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # キャンセルボタン
        self._cancel_button = QPushButton("キャンセル")
        self._cancel_button.clicked.connect(self.reject)
        layout.addWidget(self._cancel_button)

        self._cancelled = False

    def setValue(self, value: int) -> None:
        """進捗値を設定"""
        self._progress_bar.setValue(value)

    def wasCanceled(self) -> bool:
        """キャンセルされたか確認"""
        return self._cancelled

    def setMessage(self, message: str) -> None:
        """メッセージを更新"""
        self._label.setText(message)

    def reject(self) -> None:
        """キャンセル処理"""
        self._cancelled = True
        super().reject()


class MainWindow(QMainWindow):
    """メインウィンドウ"""

    def __init__(self, initial_video: Path | None = None):
        super().__init__()
        self.setWindowTitle("Defacer - 顔モザイクツール")
        self.setMinimumSize(1280, 720)

        self._current_video_path: Path | None = None
        self._annotation_file_path: Path | None = None
        self._unsaved_changes = False

        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_shortcuts()

        if initial_video:
            self._open_video(initial_video)

    def _setup_ui(self) -> None:
        """UIを構築"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # メインスプリッター（動画プレーヤー | サイドパネル）
        splitter = QSplitter(Qt.Horizontal)

        # 左側: 動画プレーヤー
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(5)

        self._video_player = VideoPlayerWidget()
        video_layout.addWidget(self._video_player, stretch=1)

        # タイムライン
        self._timeline = TimelineWidget()
        video_layout.addWidget(self._timeline)

        splitter.addWidget(video_container)

        # 右側: サイドパネル
        side_panel = self._create_side_panel()
        splitter.addWidget(side_panel)

        # スプリッターの初期サイズ比率（動画:サイドパネル = 3:1）
        splitter.setSizes([900, 300])

        main_layout.addWidget(splitter)

        # シグナル接続
        self._video_player.frame_changed.connect(self._on_frame_changed)
        self._video_player.playback_state_changed.connect(self._timeline.set_playing)
        self._video_player.annotations_changed.connect(self._on_annotations_changed)
        self._video_player.annotation_selected.connect(self._on_annotation_selected)
        self._video_player.status_message.connect(
            lambda msg, timeout: self._status_bar.showMessage(msg, timeout)
        )

        self._timeline.frame_changed.connect(self._video_player.seek)
        self._timeline.play_clicked.connect(self._video_player.play)
        self._timeline.pause_clicked.connect(self._video_player.pause)
        self._timeline.stop_clicked.connect(self._video_player.stop)
        self._timeline.delete_range_triggered.connect(self._on_delete_range)
        self._timeline.detect_range_triggered.connect(self._on_detect_range)
        self._timeline.visible_range_changed.connect(self._on_timeline_view_changed)

        # ステータスバー
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("動画ファイルを開いてください (Ctrl+O)")

        # ステータスバー用プログレスバー（初期は非表示）
        self._status_progress_bar = QProgressBar()
        self._status_progress_bar.setMaximumWidth(300)
        self._status_progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._status_progress_bar)

    def _create_side_panel(self) -> QWidget:
        """サイドパネルを作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 動画情報グループ
        info_group = QGroupBox("動画情報")
        info_layout = QVBoxLayout(info_group)

        self._info_resolution = QLabel("解像度: -")
        self._info_fps = QLabel("FPS: -")
        self._info_frames = QLabel("フレーム数: -")
        self._info_duration = QLabel("長さ: -")

        info_layout.addWidget(self._info_resolution)
        info_layout.addWidget(self._info_fps)
        info_layout.addWidget(self._info_frames)
        info_layout.addWidget(self._info_duration)

        layout.addWidget(info_group)

        # アノテーション操作グループ
        annotation_group = QGroupBox("アノテーション")
        annotation_layout = QVBoxLayout(annotation_group)

        self._annotation_count = QLabel("領域数: 0 (現フレーム: 0)")
        annotation_layout.addWidget(self._annotation_count)

        # アノテーション操作ボタン
        self._delete_btn = QPushButton("選択を削除")
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_btn.setEnabled(False)
        annotation_layout.addWidget(self._delete_btn)

        self._copy_next_btn = QPushButton("次フレームにコピー (F)")
        self._copy_next_btn.clicked.connect(self._copy_to_next_frame)
        self._copy_next_btn.setEnabled(False)
        annotation_layout.addWidget(self._copy_next_btn)

        # 補間機能
        interpolate_layout = QHBoxLayout()
        interpolate_layout.addWidget(QLabel("補間:"))
        self._interpolate_btn = QPushButton("選択を補間")
        self._interpolate_btn.clicked.connect(self._interpolate_selected)
        self._interpolate_btn.setEnabled(False)
        self._interpolate_btn.setToolTip("選択中のトラックをフレーム間で補間")
        interpolate_layout.addWidget(self._interpolate_btn)
        annotation_layout.addLayout(interpolate_layout)

        # 自動補間モード
        self._auto_interpolate_btn = QPushButton("自動補間")
        self._auto_interpolate_btn.setCheckable(True)
        self._auto_interpolate_btn.setChecked(False)
        self._auto_interpolate_btn.clicked.connect(self._toggle_auto_interpolate)
        self._auto_interpolate_btn.setToolTip(
            "ONの場合、フレームスキップ時に選択中のアノテーションを自動的に補間します"
        )
        annotation_layout.addWidget(self._auto_interpolate_btn)

        layout.addWidget(annotation_group)

        # 検出設定グループ
        detection_group = QGroupBox("自動検出")
        detection_layout = QVBoxLayout(detection_group)

        available_detectors = get_available_detectors()
        if available_detectors:
            self._detect_btn = QPushButton("自動顔検出を実行...")
            self._detect_btn.clicked.connect(self._run_auto_detection)
            detection_layout.addWidget(self._detect_btn)

            detector_info = ", ".join(available_detectors)
            detection_layout.addWidget(QLabel(f"利用可能: {detector_info}"))
        else:
            detection_layout.addWidget(QLabel("検出器がインストールされていません"))
            detection_layout.addWidget(QLabel("pip install ultralytics huggingface-hub"))

        layout.addWidget(detection_group)

        # 出力設定グループ
        output_group = QGroupBox("出力")
        output_layout = QVBoxLayout(output_group)

        self._export_btn = QPushButton("動画をエクスポート...")
        self._export_btn.clicked.connect(self._export_video)
        output_layout.addWidget(self._export_btn)

        layout.addWidget(output_group)

        layout.addStretch()

        return panel

    def _setup_menubar(self) -> None:
        """メニューバーを設定"""
        menubar = self.menuBar()

        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル(&F)")

        open_action = QAction("動画を開く(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # アノテーション保存/読み込み
        self._save_action = QAction("アノテーションを保存(&S)", self)
        self._save_action.setShortcut("Ctrl+S")
        self._save_action.triggered.connect(self._save_annotations)
        self._save_action.setEnabled(False)
        file_menu.addAction(self._save_action)

        save_as_action = QAction("名前を付けて保存(&A)...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_annotations_as)
        file_menu.addAction(save_as_action)

        load_action = QAction("アノテーションを読み込み(&L)...", self)
        load_action.triggered.connect(self._load_annotations)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_action = QAction("動画をエクスポート(&E)...", self)
        export_action.setShortcut("Ctrl+Shift+E")
        export_action.triggered.connect(self._export_video)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("終了(&Q)", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # 編集メニュー
        edit_menu = menubar.addMenu("編集(&E)")

        self._undo_action = QAction("元に戻す(&U)", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.triggered.connect(self._undo)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("やり直し(&R)", self)
        self._redo_action.setShortcut("Ctrl+Y")
        self._redo_action.triggered.connect(self._redo)
        edit_menu.addAction(self._redo_action)

        edit_menu.addSeparator()

        delete_action = QAction("選択を削除(&D)", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self._delete_selected)
        edit_menu.addAction(delete_action)

        edit_menu.addSeparator()

        # 自動検出
        auto_detect_action = QAction("自動顔検出(&A)...", self)
        auto_detect_action.setShortcut("Ctrl+D")
        auto_detect_action.triggered.connect(self._run_auto_detection)
        edit_menu.addAction(auto_detect_action)

        # 再トラッキング
        retrack_action = QAction("既存アノテーションを再トラッキング(&R)...", self)
        retrack_action.triggered.connect(self._run_retracking)
        edit_menu.addAction(retrack_action)

        # 再生メニュー
        playback_menu = menubar.addMenu("再生(&P)")

        play_action = QAction("再生/一時停止", self)
        play_action.setShortcut("Space")
        play_action.triggered.connect(self._video_player.toggle_playback)
        playback_menu.addAction(play_action)

        stop_action = QAction("停止", self)
        stop_action.triggered.connect(self._video_player.stop)
        playback_menu.addAction(stop_action)

        playback_menu.addSeparator()

        prev_frame_action = QAction("前のフレーム", self)
        prev_frame_action.setShortcut("Left")
        prev_frame_action.triggered.connect(lambda: self._video_player.step_backward(1))
        playback_menu.addAction(prev_frame_action)

        next_frame_action = QAction("次のフレーム", self)
        next_frame_action.setShortcut("Right")
        next_frame_action.triggered.connect(lambda: self._video_player.step_forward(1))
        playback_menu.addAction(next_frame_action)

        prev_10_action = QAction("10フレーム戻る", self)
        prev_10_action.setShortcut("Shift+Left")
        prev_10_action.triggered.connect(lambda: self._video_player.step_backward(10))
        playback_menu.addAction(prev_10_action)

        next_10_action = QAction("10フレーム進む", self)
        next_10_action.setShortcut("Shift+Right")
        next_10_action.triggered.connect(lambda: self._video_player.step_forward(10))
        playback_menu.addAction(next_10_action)

        # A/Dキーショートカット（次フレーム移動専用）
        prev_frame_a = QAction("前のフレーム (A)", self)
        prev_frame_a.setShortcut("A")
        prev_frame_a.triggered.connect(lambda: self._video_player.step_backward(1))
        playback_menu.addAction(prev_frame_a)

        next_frame_d = QAction("次のフレーム (D)", self)
        next_frame_d.setShortcut("D")
        next_frame_d.triggered.connect(lambda: self._video_player.step_forward(1))
        playback_menu.addAction(next_frame_d)

        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ(&H)")

        about_action = QAction("Defacerについて(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        shortcuts_action = QAction("ショートカット一覧(&K)", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def _setup_toolbar(self) -> None:
        """ツールバーを設定"""
        toolbar = QToolBar("メインツールバー")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("開く", self)
        open_action.triggered.connect(self._on_open)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        save_action = QAction("保存", self)
        save_action.triggered.connect(self._save_annotations)
        toolbar.addAction(save_action)

    def _setup_shortcuts(self) -> None:
        """追加のキーボードショートカットを設定"""
        # Fキーで次フレームにコピー
        copy_action = QAction("次フレームにコピー", self)
        copy_action.setShortcut("F")
        copy_action.triggered.connect(self._copy_to_next_frame)
        self.addAction(copy_action)

    def _on_open(self) -> None:
        """動画ファイルを開くダイアログ"""
        if self._unsaved_changes:
            reply = QMessageBox.question(
                self,
                "未保存の変更",
                "未保存の変更があります。保存しますか？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_annotations()
            elif reply == QMessageBox.Cancel:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "動画ファイルを開く",
            "",
            "動画ファイル (*.mp4 *.avi *.mov *.mkv *.webm);;すべてのファイル (*)",
        )
        if file_path:
            self._open_video(Path(file_path))

    def _open_video(self, path: Path) -> None:
        """動画を開く"""
        if not path.exists():
            QMessageBox.critical(
                self,
                "エラー",
                f"ファイルが見つかりません:\n{path}",
            )
            return

        if self._video_player.load_video(str(path)):
            self._current_video_path = path
            self._annotation_file_path = None
            self._unsaved_changes = False
            self.setWindowTitle(f"Defacer - {path.name}")

            # 動画情報を更新
            self._timeline.set_video_info(
                self._video_player.frame_count,
                self._video_player.fps,
            )
            self._update_video_info()
            self._update_annotation_info()

            # 同名のアノテーションファイルがあれば読み込み
            annotation_path = path.with_suffix(".defacer.json")
            if annotation_path.exists():
                reply = QMessageBox.question(
                    self,
                    "アノテーション読み込み",
                    f"アノテーションファイルが見つかりました:\n{annotation_path.name}\n\n読み込みますか？",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._load_annotations_from_path(annotation_path)

            self._status_bar.showMessage(f"読み込み完了: {path.name}")
        else:
            QMessageBox.critical(
                self,
                "エラー",
                f"動画を開けませんでした:\n{path}",
            )

    def _update_video_info(self) -> None:
        """動画情報表示を更新"""
        if self._video_player.frame_count == 0:
            return

        self._info_resolution.setText(
            f"解像度: {self._video_player.video_width} x {self._video_player.video_height}"
        )
        self._info_fps.setText(f"FPS: {self._video_player.fps:.2f}")
        self._info_frames.setText(f"フレーム数: {self._video_player.frame_count}")

        duration = self._video_player.frame_count / self._video_player.fps
        minutes = int(duration // 60)
        seconds = duration % 60
        self._info_duration.setText(f"長さ: {minutes:02d}:{seconds:05.2f}")

    def _update_annotation_info(self) -> None:
        """アノテーション情報を更新"""
        store = self._video_player.annotation_store
        total = len(store)
        current_frame = self._video_player.current_frame_number
        current = len(store.get_frame_annotations(current_frame))

        self._annotation_count.setText(f"領域数: {total} (現フレーム: {current})")

        # タイムラインのアノテーションマーカーを更新
        self._timeline.set_annotations(store.annotations)

    def _on_frame_changed(self, frame_number: int) -> None:
        """フレーム番号が変わった時"""
        self._timeline.set_frame(frame_number)
        self._update_annotation_info()

    def _on_annotations_changed(self, track_structure_changed: bool = True) -> None:
        """アノテーションが変更された時

        Args:
            track_structure_changed: トラック構造が変更されたか（追加/削除/統合）
        """
        self._unsaved_changes = True
        self._save_action.setEnabled(True)
        self._update_annotation_info()
        self._update_window_title()

    def _on_annotation_selected(self, annotation) -> None:
        """アノテーションが選択された時"""
        has_selection = annotation is not None
        self._delete_btn.setEnabled(has_selection)
        self._copy_next_btn.setEnabled(has_selection)
        self._interpolate_btn.setEnabled(has_selection)
        
        # タイムラインのトラック表示更新
        self._update_timeline_track_viz(annotation)

    def _on_timeline_view_changed(self) -> None:
        """タイムラインの表示範囲（ズーム・スクロール）が変更された時"""
        # 選択中のアノテーションがあればサムネイルを更新
        if self._video_player.selected_annotation:
            self._update_timeline_track_viz(self._video_player.selected_annotation)

    def _update_timeline_track_viz(self, annotation) -> None:
        """タイムライン上のトラック可視化を更新"""
        if annotation is None or annotation.track_id is None:
            self._timeline.set_selected_track([], [])
            return

        store = self._video_player.annotation_store
        frames = store.get_track_frames(annotation.track_id)
        
        if not frames:
            self._timeline.set_selected_track([], [])
            return
            
        # 表示範囲を取得
        start_view, end_view = self._timeline.get_visible_range()
        
        # タイムラインの幅からサムネイルの推奨数を計算（動的密度）
        timeline_width = self._timeline.width()
        if timeline_width <= 0:
            timeline_width = 800
            
        # 150pxに1つくらいの密度
        max_thumbnails = max(2, timeline_width // 150)
        # 上限を設定（パフォーマンス考慮）
        max_thumbnails = min(50, max_thumbnails)
        
        # 表示範囲内のフレームを抽出
        visible_frames = [f for f in frames if start_view <= f <= end_view]
        
        target_frames = []
        if not visible_frames:
             # 表示範囲内にない場合はサムネイルなし
             pass
        elif len(visible_frames) <= max_thumbnails:
             target_frames = visible_frames
        else:
             # 均等に間引く
             step = max(1.0, (len(visible_frames) - 1) / (max_thumbnails - 1))
             target_frames = [visible_frames[int(i * step)] for i in range(max_thumbnails)]
             
             # 最後のフレームが含まれないことがあるので調整（必須ではないが）
        
        # プレーヤーの状態を保存
        current_frame = self._video_player._current_frame_number
        
        thumbnails = []
        from PyQt5.QtCore import QRect

        for frame_num in target_frames:
            
            # アノテーション情報を取得（座標用）
            ann_list = store.get_frame_annotations(frame_num)
            target_ann = next((a for a in ann_list if a.track_id == annotation.track_id), None)
            
            img = self._video_player.get_thumbnail(frame_num)
            if img:
                # 30pxの高さにリサイズ（描画最適化）
                scaled_h = 30
                
                if target_ann:
                    # バウンディングボックスに合わせて切り抜き
                    bbox = target_ann.bbox
                    rect = QRect(bbox.x1, bbox.y1, bbox.width, bbox.height)
                    cropped_img = img.copy(rect)
                    final_img = cropped_img.scaledToHeight(scaled_h, Qt.SmoothTransformation)
                else:
                    final_img = img.scaledToHeight(scaled_h, Qt.SmoothTransformation)
                    
                thumbnails.append((frame_num, final_img))
                
        # 元のフレームに戻す（表示用バッファを復元）
        self._video_player._show_frame(current_frame)
        
        self._timeline.set_selected_track(frames, thumbnails)

    def _update_window_title(self) -> None:
        """ウィンドウタイトルを更新"""
        if self._current_video_path:
            title = f"Defacer - {self._current_video_path.name}"
            if self._unsaved_changes:
                title += " *"
            self.setWindowTitle(title)

    def _delete_selected(self) -> None:
        """選択中のアノテーションを削除"""
        # video_player.delete_selected_annotation() が annotations_changed を emit するので
        # ここでは追加の呼び出しは不要
        self._video_player.delete_selected_annotation()

    def _copy_to_next_frame(self) -> None:
        """選択中のアノテーションを次のフレームにコピー"""
        # video_player.copy_to_next_frame() が annotations_changed を emit するので
        # ここでは追加の呼び出しは不要
        self._video_player.copy_to_next_frame()

    def _interpolate_selected(self) -> None:
        """選択中のトラックを補間"""
        ann = self._video_player.selected_annotation
        if ann is None or ann.track_id is None:
            return

        store = self._video_player.annotation_store
        count = interpolate_track(store, ann.track_id)

        if count > 0:
            self._on_annotations_changed()
            self._status_bar.showMessage(f"{count}フレームを補間しました")
        else:
            self._status_bar.showMessage("補間するフレームがありません")

    def _on_delete_range(self, start: int, end: int) -> None:
        """指定範囲のアノテーションを削除"""
        count = self._video_player.annotation_store.remove_range(start, end)
        if count > 0:
            self._on_annotations_changed()
            self._status_bar.showMessage(f"{count}個のアノテーションを削除しました（範囲: {start}-{end}）")
        else:
            self._status_bar.showMessage(f"削除対象のアノテーションがありませんでした（範囲: {start}-{end}）")

    def _on_detect_range(self, start: int, end: int) -> None:
        """指定範囲で顔検出を実行"""
        if self._current_video_path is None:
            return

        dialog = DetectionDialog(
            self,
            self._current_video_path,
            self._video_player.frame_count,
            self._video_player.current_frame_number,
            initial_range=(start, end),
        )
        dialog.detections_ready.connect(self._on_detections_ready)
        dialog.exec_()

    def _toggle_auto_interpolate(self) -> None:
        """自動補間モードを切り替え"""
        enabled = self._auto_interpolate_btn.isChecked()
        self._video_player.set_auto_interpolate(enabled)
        status = "有効" if enabled else "無効"
        self._status_bar.showMessage(f"自動補間モード: {status}")

    def _update_progress(self, current: int, total: int) -> None:
        """進捗バーを更新"""
        if total > 100:  # 少量なら表示しない
            self._status_progress_bar.setVisible(True)
            self._status_progress_bar.setMaximum(total)
            self._status_progress_bar.setValue(current)
            QApplication.processEvents()  # UIを更新

    def _undo(self) -> None:
        """アンドゥ"""
        if self._video_player.annotation_store.undo():
            self._on_annotations_changed()

    def _redo(self) -> None:
        """リドゥ"""
        if self._video_player.annotation_store.redo():
            self._on_annotations_changed()

    def _save_annotations(self) -> None:
        """アノテーションを保存"""
        if self._annotation_file_path:
            self._save_annotations_to_path(self._annotation_file_path)
        elif self._current_video_path:
            default_path = self._current_video_path.with_suffix(".defacer.json")
            self._save_annotations_to_path(default_path)
        else:
            self._save_annotations_as()

    def _save_annotations_as(self) -> None:
        """名前を付けてアノテーションを保存"""
        default_name = ""
        if self._current_video_path:
            default_name = str(self._current_video_path.with_suffix(".defacer.json"))

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "アノテーションを保存",
            default_name,
            "Defacerアノテーション (*.defacer.json);;JSONファイル (*.json)",
        )
        if file_path:
            self._save_annotations_to_path(Path(file_path))

    def _save_annotations_to_path(self, path: Path) -> None:
        """指定パスにアノテーションを保存"""
        try:
            self._video_player.annotation_store.save(path)
            self._annotation_file_path = path
            self._unsaved_changes = False
            self._update_window_title()
            self._status_bar.showMessage(f"保存しました: {path.name}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "保存エラー",
                f"アノテーションの保存に失敗しました:\n{e}",
            )

    def _load_annotations(self) -> None:
        """アノテーションを読み込み"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "アノテーションを読み込み",
            "",
            "Defacerアノテーション (*.defacer.json);;JSONファイル (*.json)",
        )
        if file_path:
            self._load_annotations_from_path(Path(file_path))

    def _load_annotations_from_path(self, path: Path) -> None:
        """指定パスからアノテーションを読み込み"""
        try:
            store = AnnotationStore.load(path)
            self._video_player.set_annotation_store(store)
            self._annotation_file_path = path
            self._unsaved_changes = False
            self._update_annotation_info()
            self._update_window_title()
            self._status_bar.showMessage(f"読み込みました: {path.name}")
        except Exception as e:
            QMessageBox.critical(
                self,
                "読み込みエラー",
                f"アノテーションの読み込みに失敗しました:\n{e}",
            )

    def _run_auto_detection(self) -> None:
        """自動顔検出を実行"""
        if self._current_video_path is None:
            self._status_bar.showMessage("動画ファイルを開いてください", 5000)
            return

        dialog = DetectionDialog(
            self,
            self._current_video_path,
            self._video_player.frame_count,
            self._video_player.current_frame_number,
        )
        dialog.detections_ready.connect(self._on_detections_ready)
        dialog.exec_()

    def _on_detections_ready(self, new_store: AnnotationStore) -> None:
        """検出結果を受け取ってマージ"""
        current_store = self._video_player.annotation_store
        total_count = len(new_store)

        # Undo状態を保存（バッチ処理の前に1回だけ）
        current_store._save_undo_state()

        # 大量のアノテーションがある場合はステータスバーに進捗を表示
        show_progress = total_count > 100
        if show_progress:
            self._status_progress_bar.setMaximum(total_count)
            self._status_progress_bar.setValue(0)
            self._status_progress_bar.setVisible(True)
            self._status_bar.showMessage("検出結果をマージ中...")
            QApplication.processEvents()

        # 新しい検出結果をマージ（save_undo=Falseで高速化）
        for idx, ann in enumerate(new_store):
            # プログレスバーの更新（10件ごとに更新してUIフリーズ防止）
            if show_progress and idx % 10 == 0:
                self._status_progress_bar.setValue(idx)
                # パーセント表示も更新
                percent = int((idx / total_count) * 100)
                self._status_bar.showMessage(f"検出結果をマージ中... {percent}% ({idx}/{total_count})")
                QApplication.processEvents()

            # 既存のトラックIDと衝突しないように調整
            ann.track_id = current_store.new_track_id()
            current_store.add(ann, save_undo=False)

        # プログレスバーを隠す
        if show_progress:
            self._status_progress_bar.setValue(total_count)
            self._status_progress_bar.setVisible(False)

        # 処理完了後に一度だけUI更新
        self._on_annotations_changed()
        self._status_bar.showMessage(f"{len(new_store)}件の検出結果を追加しました")

    def _run_retracking(self) -> None:
        """既存アノテーションの再トラッキングを実行"""
        if self._current_video_path is None:
            self._status_bar.showMessage("動画ファイルを開いてください", 5000)
            return

        if len(self._video_player.annotation_store) == 0:
            self._status_bar.showMessage("アノテーションがありません。先にアノテーションを追加してください", 5000)
            return

        dialog = RetrackDialog(
            self,
            self._current_video_path,
            self._video_player.annotation_store,
            self._video_player.frame_count,
            self._video_player.current_frame_number,
        )
        dialog.retrack_completed.connect(self._on_retrack_completed)
        dialog.exec_()

    def _on_retrack_completed(self) -> None:
        """再トラッキング完了時の処理"""
        self._on_annotations_changed()
        self._video_player.update()
        self._status_bar.showMessage("再トラッキングが完了しました")

    def _export_video(self) -> None:
        """動画をエクスポート"""
        if self._current_video_path is None:
            self._status_bar.showMessage("動画ファイルを開いてください", 5000)
            return

        if len(self._video_player.annotation_store) == 0:
            reply = QMessageBox.question(
                self,
                "確認",
                "アノテーションがありません。そのままエクスポートしますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        dialog = ExportDialog(
            self,
            self._current_video_path,
            self._video_player.annotation_store,
        )
        dialog.exec_()

    def _show_about(self) -> None:
        """Aboutダイアログ"""
        QMessageBox.about(
            self,
            "Defacerについて",
            "Defacer v0.3.2\n\n"
            "動画内の顔を自動検知してモザイク処理を行うソフトウェア\n\n"
            "検知漏れがある場合は手動で顔領域を指定できます。\n\n"
            "v0.3.2の新機能:\n"
            "- アノテーション削除処理を O(n) から O(1) に最適化（参照カウント方式）\n"
            "- 位置編集時の無駄なトラック一覧更新を削減\n"
            "- 大量データ処理時に進捗バーを表示",
        )

    def _show_shortcuts(self) -> None:
        """ショートカット一覧ダイアログ"""
        shortcuts = """
キーボードショートカット:

ファイル操作:
  Ctrl+O        動画を開く
  Ctrl+S        アノテーションを保存
  Ctrl+Shift+S  名前を付けて保存
  Ctrl+Q        終了

再生操作:
  Space         再生/一時停止
  ←/→ または A/D  1フレーム移動
  Shift+←/→     10フレーム移動

アノテーション:
  マウスドラッグ   描画モードで領域を描画
  Delete         選択領域を削除
  F              選択を次フレームにコピー
  Ctrl+Z         元に戻す
  Ctrl+Y         やり直し
  E              編集モードに切替
        """
        QMessageBox.information(self, "ショートカット一覧", shortcuts.strip())

    def closeEvent(self, event) -> None:
        """ウィンドウを閉じる前の処理"""
        if self._unsaved_changes:
            reply = QMessageBox.question(
                self,
                "未保存の変更",
                "未保存の変更があります。保存しますか？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_annotations()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        self._video_player.release()
        event.accept()


def main(initial_video: Path | None = None) -> int:
    """GUIアプリケーションを起動"""
    app = QApplication(sys.argv)
    app.setApplicationName("Defacer")
    app.setOrganizationName("Defacer")

    # ダークテーマ風のスタイル
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2d2d2d;
            color: #e0e0e0;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #606060;
        }
        QPushButton:disabled {
            background-color: #353535;
            color: #707070;
        }
        QPushButton:checked {
            background-color: #0078d4;
            border-color: #0078d4;
        }
        QSlider::groove:horizontal {
            border: 1px solid #444;
            height: 8px;
            background: #333;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #0078d4;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        QSlider::sub-page:horizontal {
            background: #0078d4;
            border-radius: 4px;
        }
        QMenuBar {
            background-color: #2d2d2d;
        }
        QMenuBar::item:selected {
            background-color: #404040;
        }
        QMenu {
            background-color: #2d2d2d;
            border: 1px solid #444;
        }
        QMenu::item:selected {
            background-color: #0078d4;
        }
        QStatusBar {
            background-color: #252525;
        }
        QToolBar {
            background-color: #2d2d2d;
            border: none;
            spacing: 5px;
        }
        QSpinBox {
            background-color: #404040;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 2px;
        }
    """)

    window = MainWindow(initial_video)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    main()

"""動画出力クラス"""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np


class VideoWriter:
    """FFmpegを使用した動画出力クラス"""

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
    ):
        """
        Args:
            output_path: 出力ファイルパス
            width: 動画幅
            height: 動画高さ
            fps: フレームレート
            codec: 使用するコーデック（デフォルト: libx264）
            crf: 品質（0-51、低いほど高品質、デフォルト: 18）
            preset: エンコード速度プリセット（ultrafast, fast, medium, slow, veryslow）
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset

        self._process: subprocess.Popen | None = None
        self._frame_count = 0

    def open(self) -> None:
        """FFmpegプロセスを開始"""
        cmd = [
            "ffmpeg",
            "-y",  # 上書き確認なし
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",  # stdin から入力
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        """フレームを書き込み"""
        if self._process is None:
            raise RuntimeError("VideoWriterが開かれていません")

        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def close(self) -> None:
        """FFmpegプロセスを終了"""
        if self._process is not None:
            self._process.stdin.close()
            self._process.wait()
            self._process = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def export_video_with_audio(
    input_path: str | Path,
    output_path: str | Path,
    frame_generator: Iterator[np.ndarray],
    total_frames: int,
    fps: float,
    width: int,
    height: int,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    progress_callback: Callable[[int, int], None] | None = None,
) -> bool:
    """
    音声付きで動画をエクスポート

    Args:
        input_path: 入力動画パス（音声ソース）
        output_path: 出力動画パス
        frame_generator: フレームを生成するイテレータ
        total_frames: 総フレーム数
        fps: フレームレート
        width: 動画幅
        height: 動画高さ
        codec: 使用するコーデック
        crf: 品質
        preset: エンコード速度プリセット
        progress_callback: 進捗コールバック(current, total)

    Returns:
        成功した場合True
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 一時ファイルに映像を出力
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = Path(tmp.name)

    try:
        # 映像を一時ファイルに書き込み
        with VideoWriter(
            temp_video_path,
            width,
            height,
            fps,
            codec,
            crf,
            preset,
        ) as writer:
            for i, frame in enumerate(frame_generator):
                writer.write(frame)
                if progress_callback:
                    progress_callback(i + 1, total_frames)

        # FFmpegで音声と結合
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_video_path),  # 映像
            "-i", str(input_path),  # 音声ソース
            "-c:v", "copy",  # 映像はコピー
            "-c:a", "aac",  # 音声はAACでエンコード
            "-map", "0:v:0",  # 映像は1番目の入力から
            "-map", "1:a:0?",  # 音声は2番目の入力から（存在すれば）
            "-shortest",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return result.returncode == 0

    finally:
        # 一時ファイルを削除
        if temp_video_path.exists():
            temp_video_path.unlink()


def check_ffmpeg_available() -> bool:
    """FFmpegが利用可能か確認"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

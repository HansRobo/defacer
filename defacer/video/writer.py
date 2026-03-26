"""動画出力クラス"""

import logging
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        self._stderr_output: str = ""

    def open(self) -> None:
        """FFmpegプロセスを開始"""
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
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
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """フレームを書き込み"""
        if self._process is None:
            raise RuntimeError("VideoWriterが開かれていません")

        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        try:
            self._process.stdin.write(memoryview(frame))
        except BrokenPipeError:
            stderr = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            raise RuntimeError(f"FFmpegプロセスが異常終了しました: {stderr}") from None
        self._frame_count += 1

    def close(self) -> None:
        """FFmpegプロセスを終了"""
        if self._process is not None:
            self._process.stdin.close()
            self._stderr_output = self._process.stderr.read().decode(errors="replace") if self._process.stderr else ""
            returncode = self._process.wait()
            self._process = None
            if returncode != 0:
                logger.error("FFmpegエンコードエラー: %s", self._stderr_output)
                raise RuntimeError(f"FFmpegエンコードがエラーコード {returncode} で終了しました")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # 既に例外が発生している場合はclose()のエラーを抑制
            try:
                self.close()
            except RuntimeError:
                pass
        else:
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
    音声付きで動画をエクスポート（単一パス）

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

    # 単一パス: rawフレームをstdinからパイプし、元動画から音声を取得
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for i, frame in enumerate(frame_generator):
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            try:
                process.stdin.write(memoryview(frame))
            except BrokenPipeError:
                break
            if progress_callback:
                progress_callback(i + 1, total_frames)

        process.stdin.close()
        stderr_output = process.stderr.read().decode(errors="replace") if process.stderr else ""
        returncode = process.wait()

        if returncode != 0:
            logger.error("FFmpegエクスポートエラー: %s", stderr_output)
            return False
        return True

    except Exception:
        process.kill()
        process.wait()
        raise


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

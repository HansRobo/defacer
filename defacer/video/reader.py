"""動画読み取りクラス"""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """OpenCVを使用した動画読み取りクラス"""

    def __init__(self, path: str | Path):
        """
        Args:
            path: 動画ファイルのパス
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"動画ファイルを開けません: {self.path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0

    @property
    def frame_count(self) -> int:
        """総フレーム数"""
        return self._frame_count

    @property
    def fps(self) -> float:
        """フレームレート"""
        return self._fps

    @property
    def width(self) -> int:
        """フレーム幅"""
        return self._width

    @property
    def height(self) -> int:
        """フレーム高さ"""
        return self._height

    @property
    def duration(self) -> float:
        """動画の長さ（秒）"""
        if self._fps > 0:
            return self._frame_count / self._fps
        return 0.0

    @property
    def current_frame(self) -> int:
        """現在のフレーム位置"""
        return self._current_frame

    def seek(self, frame_number: int) -> bool:
        """
        指定フレームにシーク

        Args:
            frame_number: 移動先のフレーム番号

        Returns:
            成功した場合True
        """
        if 0 <= frame_number < self._frame_count:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame = frame_number
            return True
        return False

    def read(self) -> np.ndarray | None:
        """
        現在位置のフレームを読み取り

        Returns:
            BGR画像、または読み取り失敗時はNone
        """
        ret, frame = self._cap.read()
        if ret:
            self._current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            return frame
        return None

    def read_frame(self, frame_number: int) -> np.ndarray | None:
        """
        指定フレームを読み取り

        Args:
            frame_number: 読み取るフレーム番号

        Returns:
            BGR画像、または読み取り失敗時はNone
        """
        if self.seek(frame_number):
            return self.read()
        return None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """フレームをイテレート"""
        self.seek(0)
        while True:
            frame = self.read()
            if frame is None:
                break
            yield self._current_frame - 1, frame

    def __len__(self) -> int:
        return self._frame_count

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """リソースを解放"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        self.release()

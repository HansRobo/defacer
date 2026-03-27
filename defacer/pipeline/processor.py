"""メイン処理パイプライン"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from defacer.video.reader import VideoReader
from defacer.video.writer import export_video_with_audio, check_ffmpeg_available
from defacer.anonymization import Anonymizer, MosaicAnonymizer
from defacer.annotation import AnnotationStore


@dataclass
class ExportConfig:
    """エクスポート設定"""
    anonymizer: Anonymizer | None = None
    ellipse: bool = True
    bbox_scale: float = 1.0
    interpolate: bool = True
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"


def process_frame(
    frame: np.ndarray,
    frame_number: int,
    annotations: AnnotationStore,
    anonymizer: Anonymizer,
    ellipse: bool = True,
    bbox_scale: float = 1.0,
) -> np.ndarray:
    """
    単一フレームを処理

    Args:
        frame: 入力フレーム
        frame_number: フレーム番号
        annotations: アノテーションストア
        anonymizer: 使用するAnonymizer
        ellipse: 楕円形マスクを使用するか
        bbox_scale: バウンディングボックスの拡大率

    Returns:
        処理後のフレーム
    """
    frame_annotations = annotations.get_frame_annotations(frame_number)

    if not frame_annotations:
        return frame

    result = frame.copy()
    h, w = frame.shape[:2]

    for ann in frame_annotations:
        bbox = ann.bbox

        if bbox_scale != 1.0:
            bbox = bbox.scale_from_center(bbox_scale, w, h)
        result = anonymizer.apply(result, bbox, ellipse)

    return result


def generate_processed_frames(
    reader: VideoReader,
    annotations: AnnotationStore,
    anonymizer: Anonymizer,
    ellipse: bool = True,
    bbox_scale: float = 1.0,
) -> Iterator[np.ndarray]:
    """
    処理済みフレームを生成するイテレータ

    Args:
        reader: VideoReader
        annotations: アノテーションストア
        anonymizer: 使用するAnonymizer
        ellipse: 楕円形マスクを使用するか
        bbox_scale: バウンディングボックスの拡大率

    Yields:
        処理後のフレーム
    """
    for frame_number, frame in reader:
        processed = process_frame(
            frame,
            frame_number,
            annotations,
            anonymizer,
            ellipse,
            bbox_scale,
        )
        yield processed


def export_processed_video(
    input_path: str | Path,
    output_path: str | Path,
    annotations: AnnotationStore,
    config: ExportConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> bool:
    """
    処理済み動画をエクスポート

    Args:
        input_path: 入力動画パス
        output_path: 出力動画パス
        annotations: アノテーションストア
        config: エクスポート設定（Noneの場合はデフォルト値）
        progress_callback: 進捗コールバック(current, total)

    Returns:
        成功した場合True
    """
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpegが見つかりません。インストールしてください。")

    if config is None:
        config = ExportConfig()
    anonymizer = config.anonymizer or MosaicAnonymizer()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if config.interpolate:
        from defacer.tracking.interpolation import interpolate_sequential_annotations
        interpolate_sequential_annotations(annotations)

    with VideoReader(input_path) as reader:
        frame_generator = generate_processed_frames(
            reader,
            annotations,
            anonymizer,
            config.ellipse,
            config.bbox_scale,
        )

        return export_video_with_audio(
            input_path,
            output_path,
            frame_generator,
            reader.frame_count,
            reader.fps,
            reader.width,
            reader.height,
            config.codec,
            config.crf,
            config.preset,
            progress_callback,
        )

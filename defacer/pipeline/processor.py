"""メイン処理パイプライン"""

from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from defacer.video.reader import VideoReader
from defacer.video.writer import export_video_with_audio, check_ffmpeg_available
from defacer.anonymization.base import Anonymizer
from defacer.anonymization.mosaic import MosaicAnonymizer
from defacer.anonymization.blur import GaussianBlurAnonymizer, SolidFillAnonymizer
from defacer.gui.annotation import AnnotationStore
from defacer.config import AnonymizationType, AnonymizationConfig


def create_anonymizer(config: AnonymizationConfig) -> Anonymizer:
    """設定に基づいてAnonymizerを作成"""
    if config.anonymization_type == AnonymizationType.MOSAIC:
        return MosaicAnonymizer(block_size=config.mosaic_block_size)
    elif config.anonymization_type == AnonymizationType.BLUR:
        return GaussianBlurAnonymizer(kernel_size=config.blur_kernel_size)
    elif config.anonymization_type == AnonymizationType.SOLID:
        return SolidFillAnonymizer(color=config.solid_color)
    else:
        return MosaicAnonymizer()


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

        # バウンディングボックスを拡大
        if bbox_scale != 1.0:
            cx, cy = bbox.center
            new_w = int(bbox.width * bbox_scale)
            new_h = int(bbox.height * bbox_scale)
            x1 = max(0, cx - new_w // 2)
            y1 = max(0, cy - new_h // 2)
            x2 = min(w, cx + new_w // 2)
            y2 = min(h, cy + new_h // 2)
            scaled_bbox = (x1, y1, x2, y2)
        else:
            scaled_bbox = bbox.to_tuple()

        result = anonymizer.apply(result, scaled_bbox, ellipse)

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
    anonymizer: Anonymizer | None = None,
    ellipse: bool = True,
    bbox_scale: float = 1.0,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    progress_callback: Callable[[int, int], None] | None = None,
    interpolate: bool = True,
) -> bool:
    """
    処理済み動画をエクスポート

    Args:
        input_path: 入力動画パス
        output_path: 出力動画パス
        annotations: アノテーションストア
        anonymizer: 使用するAnonymizer（Noneの場合はMosaicAnonymizer）
        ellipse: 楕円形マスクを使用するか
        bbox_scale: バウンディングボックスの拡大率
        codec: 使用するコーデック
        crf: 品質
        preset: エンコード速度プリセット
        progress_callback: 進捗コールバック(current, total)
        interpolate: フレーム間を自動補間するか（デフォルト: True）

    Returns:
        成功した場合True
    """
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpegが見つかりません。インストールしてください。")

    if anonymizer is None:
        anonymizer = MosaicAnonymizer()

    input_path = Path(input_path)
    output_path = Path(output_path)

    # エクスポート前に補間を実行
    if interpolate:
        from defacer.tracking.interpolation import interpolate_sequential_annotations
        interpolate_sequential_annotations(annotations)

    with VideoReader(input_path) as reader:
        frame_generator = generate_processed_frames(
            reader,
            annotations,
            anonymizer,
            ellipse,
            bbox_scale,
        )

        return export_video_with_audio(
            input_path,
            output_path,
            frame_generator,
            reader.frame_count,
            reader.fps,
            reader.width,
            reader.height,
            codec,
            crf,
            preset,
            progress_callback,
        )

"""設定クラス"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DetectorType(Enum):
    """顔検知器の種類"""

    YOLO11_FACE = "yolo11-face"


class AnonymizationType(Enum):
    """匿名化の種類"""

    MOSAIC = "mosaic"
    BLUR = "blur"
    SOLID = "solid"


@dataclass
class DetectionConfig:
    """顔検知設定"""

    detector_type: DetectorType = DetectorType.YOLO11_FACE
    confidence_threshold: float = 0.5
    bbox_scale: float = 1.1  # バウンディングボックス拡大率


@dataclass
class TrackingConfig:
    """トラッキング設定"""

    enabled: bool = True
    max_age: int = 30  # 見失ってから保持するフレーム数
    min_hits: int = 3  # 確定するまでの検出回数


@dataclass
class AnonymizationConfig:
    """匿名化設定"""

    anonymization_type: AnonymizationType = AnonymizationType.MOSAIC
    mosaic_block_size: int = 10  # モザイクのブロックサイズ
    blur_kernel_size: int = 99  # ぼかしのカーネルサイズ
    solid_color: tuple[int, int, int] = (0, 0, 0)  # 塗りつぶし色


@dataclass
class OutputConfig:
    """出力設定"""

    output_path: Path | None = None
    codec: str = "libx264"
    crf: int = 18  # 品質（低いほど高品質）
    preserve_audio: bool = True


@dataclass
class Config:
    """全体設定"""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    anonymization: AnonymizationConfig = field(default_factory=AnonymizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

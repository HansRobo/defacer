"""匿名化（モザイク処理）モジュール"""

from defacer.anonymization.base import Anonymizer
from defacer.anonymization.mosaic import MosaicAnonymizer
from defacer.anonymization.blur import GaussianBlurAnonymizer, SolidFillAnonymizer


def create_anonymizer(
    anonymizer_type: str,
    block_size: int = 10,
    kernel_size: int = 99,
    color: tuple[int, int, int] = (0, 0, 0),
) -> Anonymizer:
    """
    文字列で指定した種類のAnonymizerを作成

    Args:
        anonymizer_type: "mosaic" / "blur" / "solid"
        block_size: モザイク用ブロックサイズ
        kernel_size: ぼかし用カーネルサイズ
        color: 塗りつぶし色 (B, G, R)

    Returns:
        Anonymizerインスタンス
    """
    if anonymizer_type == "mosaic":
        return MosaicAnonymizer(block_size=block_size)
    elif anonymizer_type == "blur":
        return GaussianBlurAnonymizer(kernel_size=kernel_size)
    elif anonymizer_type == "solid":
        return SolidFillAnonymizer(color=color)
    else:
        return MosaicAnonymizer(block_size=block_size)


__all__ = [
    "Anonymizer",
    "MosaicAnonymizer",
    "GaussianBlurAnonymizer",
    "SolidFillAnonymizer",
    "create_anonymizer",
]

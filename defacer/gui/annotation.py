"""後方互換のための再エクスポート（本体は defacer.annotation）"""

from defacer.annotation import AnnotationStore  # noqa: F401
from defacer.models import BoundingBox, Annotation  # noqa: F401

__all__ = ["AnnotationStore", "BoundingBox", "Annotation"]

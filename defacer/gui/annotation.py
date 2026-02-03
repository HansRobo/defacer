"""手動アノテーション機能"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class BoundingBox:
    """バウンディングボックス"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def contains_point(self, x: int, y: int, margin: int = 0) -> bool:
        """点がボックス内にあるか判定"""
        return (
            self.x1 - margin <= x <= self.x2 + margin
            and self.y1 - margin <= y <= self.y2 + margin
        )

    def get_resize_handle(self, x: int, y: int, handle_size: int = 10) -> str | None:
        """リサイズハンドルを取得（nw, ne, sw, se, n, s, e, w）"""
        hs = handle_size

        # 角
        if abs(x - self.x1) < hs and abs(y - self.y1) < hs:
            return "nw"
        if abs(x - self.x2) < hs and abs(y - self.y1) < hs:
            return "ne"
        if abs(x - self.x1) < hs and abs(y - self.y2) < hs:
            return "sw"
        if abs(x - self.x2) < hs and abs(y - self.y2) < hs:
            return "se"

        # 辺
        if abs(y - self.y1) < hs and self.x1 < x < self.x2:
            return "n"
        if abs(y - self.y2) < hs and self.x1 < x < self.x2:
            return "s"
        if abs(x - self.x1) < hs and self.y1 < y < self.y2:
            return "w"
        if abs(x - self.x2) < hs and self.y1 < y < self.y2:
            return "e"

        return None

    def normalize(self) -> "BoundingBox":
        """座標を正規化（x1 < x2, y1 < y2 にする）"""
        return BoundingBox(
            x1=min(self.x1, self.x2),
            y1=min(self.y1, self.y2),
            x2=max(self.x1, self.x2),
            y2=max(self.y1, self.y2),
        )

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """画像境界内に収める"""
        return BoundingBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
        )

    @classmethod
    def interpolate(cls, box1: "BoundingBox", box2: "BoundingBox", t: float) -> "BoundingBox":
        """2つのボックス間を線形補間"""
        return cls(
            x1=int(box1.x1 + (box2.x1 - box1.x1) * t),
            y1=int(box1.y1 + (box2.y1 - box1.y1) * t),
            x2=int(box1.x2 + (box2.x2 - box1.x2) * t),
            y2=int(box1.y2 + (box2.y2 - box1.y2) * t),
        )


@dataclass
class Annotation:
    """単一のアノテーション（1フレーム、1領域）"""

    frame: int
    bbox: BoundingBox
    track_id: int | None = None  # トラッキングID（同一人物を識別）
    is_manual: bool = True  # 手動で追加されたか
    confidence: float = 1.0  # 信頼度（自動検出の場合）

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "bbox": asdict(self.bbox),
            "track_id": self.track_id,
            "is_manual": self.is_manual,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        return cls(
            frame=data["frame"],
            bbox=BoundingBox(**data["bbox"]),
            track_id=data.get("track_id"),
            is_manual=data.get("is_manual", True),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class AnnotationStore:
    """アノテーションの保存・管理"""

    annotations: dict[int, list[Annotation]] = field(default_factory=dict)
    _next_track_id: int = 1
    _undo_stack: list[dict] = field(default_factory=list)
    _redo_stack: list[dict] = field(default_factory=list)

    def add(self, annotation: Annotation, save_undo: bool = True) -> None:
        """アノテーションを追加"""
        if save_undo:
            self._save_undo_state()

        frame = annotation.frame
        if frame not in self.annotations:
            self.annotations[frame] = []
        self.annotations[frame].append(annotation)

    def remove(self, frame: int, index: int, save_undo: bool = True) -> Annotation | None:
        """アノテーションを削除"""
        if frame not in self.annotations:
            return None
        if index < 0 or index >= len(self.annotations[frame]):
            return None

        if save_undo:
            self._save_undo_state()

        removed = self.annotations[frame].pop(index)
        if not self.annotations[frame]:
            del self.annotations[frame]
        return removed

    def remove_annotation(self, annotation: Annotation, save_undo: bool = True) -> bool:
        """指定のアノテーションを削除"""
        frame = annotation.frame
        if frame not in self.annotations:
            return False

        for i, ann in enumerate(self.annotations[frame]):
            if ann is annotation:
                self.remove(frame, i, save_undo)
                return True
        return False

    def get_frame_annotations(self, frame: int) -> list[Annotation]:
        """指定フレームのアノテーションを取得"""
        return self.annotations.get(frame, [])

    def get_all_frames(self) -> list[int]:
        """アノテーションがあるフレームのリスト"""
        return sorted(self.annotations.keys())

    def get_annotation_at_point(
        self, frame: int, x: int, y: int, margin: int = 5
    ) -> tuple[Annotation, int] | None:
        """指定位置のアノテーションを取得（最前面のもの）"""
        annotations = self.get_frame_annotations(frame)
        for i in range(len(annotations) - 1, -1, -1):
            if annotations[i].bbox.contains_point(x, y, margin):
                return (annotations[i], i)
        return None

    def new_track_id(self) -> int:
        """新しいトラッキングIDを生成"""
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def get_all_track_ids(self) -> set[int]:
        """すべてのトラックIDを取得（None除く）"""
        track_ids = set()
        for ann in self:
            if ann.track_id is not None:
                track_ids.add(ann.track_id)
        return track_ids

    def get_track_info(self, track_id: int) -> dict:
        """トラックの情報を取得（フレーム範囲、アノテーション数）"""
        frames = []
        count = 0
        for ann in self:
            if ann.track_id == track_id:
                frames.append(ann.frame)
                count += 1

        if not frames:
            return {"exists": False}

        return {
            "exists": True,
            "frame_min": min(frames),
            "frame_max": max(frames),
            "frame_count": len(set(frames)),
            "annotation_count": count,
        }

    def merge_tracks(
        self,
        source_track_id: int,
        target_track_id: int,
        save_undo: bool = True,
    ) -> int:
        """
        source_track_idのすべてのアノテーションをtarget_track_idに統合

        Args:
            source_track_id: 統合元のトラックID
            target_track_id: 統合先のトラックID
            save_undo: Undoスタックに保存するか

        Returns:
            変更されたアノテーション数
        """
        if source_track_id == target_track_id:
            return 0

        if save_undo:
            self._save_undo_state()

        count = 0
        for frame, anns in self.annotations.items():
            for ann in anns:
                if ann.track_id == source_track_id:
                    ann.track_id = target_track_id
                    count += 1

        return count

    def interpolate_frames(
        self,
        track_id: int,
        start_frame: int,
        end_frame: int,
        save_undo: bool = True,
    ) -> int:
        """指定トラックIDの開始/終了フレーム間を補間"""
        # 開始と終了のアノテーションを探す
        start_ann = None
        end_ann = None

        for ann in self.get_frame_annotations(start_frame):
            if ann.track_id == track_id:
                start_ann = ann
                break

        for ann in self.get_frame_annotations(end_frame):
            if ann.track_id == track_id:
                end_ann = ann
                break

        if start_ann is None or end_ann is None:
            return 0

        if save_undo:
            self._save_undo_state()

        # 中間フレームを生成
        count = 0
        for frame in range(start_frame + 1, end_frame):
            t = (frame - start_frame) / (end_frame - start_frame)
            interpolated_bbox = BoundingBox.interpolate(start_ann.bbox, end_ann.bbox, t)

            # 既存のアノテーションがあれば更新、なければ追加
            existing = None
            for ann in self.get_frame_annotations(frame):
                if ann.track_id == track_id:
                    existing = ann
                    break

            if existing:
                existing.bbox = interpolated_bbox
            else:
                new_ann = Annotation(
                    frame=frame,
                    bbox=interpolated_bbox,
                    track_id=track_id,
                    is_manual=True,
                    confidence=1.0,
                )
                self.add(new_ann, save_undo=False)
                count += 1

        return count

    def clear(self, save_undo: bool = True) -> None:
        """全アノテーションをクリア"""
        if save_undo:
            self._save_undo_state()
        self.annotations.clear()

    def _save_undo_state(self) -> None:
        """現在の状態をUndoスタックに保存"""
        state = self.to_dict()
        self._undo_stack.append(state)
        self._redo_stack.clear()
        # スタックサイズ制限
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        """アンドゥ"""
        if not self._undo_stack:
            return False

        # 現在の状態をRedoに保存
        self._redo_stack.append(self.to_dict())

        # 前の状態を復元
        state = self._undo_stack.pop()
        self._restore_state(state)
        return True

    def redo(self) -> bool:
        """リドゥ"""
        if not self._redo_stack:
            return False

        # 現在の状態をUndoに保存
        self._undo_stack.append(self.to_dict())

        # 次の状態を復元
        state = self._redo_stack.pop()
        self._restore_state(state)
        return True

    def _restore_state(self, state: dict) -> None:
        """状態を復元"""
        self.annotations.clear()
        for frame_str, annotations in state.get("annotations", {}).items():
            frame = int(frame_str)
            self.annotations[frame] = [Annotation.from_dict(a) for a in annotations]
        self._next_track_id = state.get("next_track_id", 1)

    def to_dict(self) -> dict:
        """JSON用の辞書に変換"""
        return {
            "annotations": {
                str(frame): [a.to_dict() for a in anns]
                for frame, anns in self.annotations.items()
            },
            "next_track_id": self._next_track_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationStore":
        """辞書から復元"""
        store = cls()
        for frame_str, annotations in data.get("annotations", {}).items():
            frame = int(frame_str)
            store.annotations[frame] = [Annotation.from_dict(a) for a in annotations]
        store._next_track_id = data.get("next_track_id", 1)
        return store

    def save(self, path: Path | str) -> None:
        """JSONファイルに保存"""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "AnnotationStore":
        """JSONファイルから読み込み"""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return sum(len(anns) for anns in self.annotations.values())

    def __iter__(self) -> Iterator[Annotation]:
        for frame in sorted(self.annotations.keys()):
            yield from self.annotations[frame]

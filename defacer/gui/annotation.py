"""手動アノテーション機能"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Callable


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

    # パフォーマンス最適化用キャッシュ
    _total_count: int = 0
    _track_ids: set[int] = field(default_factory=set)
    _track_count: dict[int, int] = field(default_factory=dict)  # track_id → アノテーション数

    # 高速アクセス用インデックス
    _track_annotations: dict[int, dict[int, Annotation]] = field(default_factory=dict)  # track_id → {id(ann): ann}
    _frame_track_index: dict[tuple[int, int], Annotation] = field(default_factory=dict)  # (frame, track_id) → ann

    # 進捗通知コールバック
    progress_callback: Callable[[int, int], None] | None = None

    def _rebuild_cache(self) -> None:
        """キャッシュを再構築"""
        self._total_count = 0
        self._track_ids.clear()
        self._track_count.clear()
        self._track_annotations.clear()
        self._frame_track_index.clear()

        frames = list(self.annotations.items())
        total = len(frames)

        for i, (frame, anns) in enumerate(frames):
            # 進捗通知（100フレームごと）
            if self.progress_callback and total > 100 and i % 100 == 0:
                self.progress_callback(i, total)

            self._total_count += len(anns)
            for ann in anns:
                if ann.track_id is not None:
                    self._track_ids.add(ann.track_id)
                    self._track_count[ann.track_id] = self._track_count.get(ann.track_id, 0) + 1

                    # 高速インデックスを構築
                    if ann.track_id not in self._track_annotations:
                        self._track_annotations[ann.track_id] = {}
                    self._track_annotations[ann.track_id][id(ann)] = ann
                    self._frame_track_index[(frame, ann.track_id)] = ann

        # 完了通知
        if self.progress_callback and total > 100:
            self.progress_callback(total, total)

    def add(self, annotation: Annotation, save_undo: bool = True) -> None:
        """アノテーションを追加"""
        if save_undo:
            self._save_undo_state()

        frame = annotation.frame
        if frame not in self.annotations:
            self.annotations[frame] = []
        self.annotations[frame].append(annotation)

        # キャッシュ更新
        self._total_count += 1
        if annotation.track_id is not None:
            self._track_ids.add(annotation.track_id)
            self._track_count[annotation.track_id] = self._track_count.get(annotation.track_id, 0) + 1

            # インデックス更新
            if annotation.track_id not in self._track_annotations:
                self._track_annotations[annotation.track_id] = {}
            self._track_annotations[annotation.track_id][id(annotation)] = annotation
            self._frame_track_index[(frame, annotation.track_id)] = annotation

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

        # キャッシュ更新
        self._total_count -= 1
        # 参照カウント方式で O(1) 削除
        if removed.track_id is not None:
            count = self._track_count.get(removed.track_id, 0)
            if count <= 1:
                # 最後の1個なので削除
                self._track_ids.discard(removed.track_id)
                self._track_count.pop(removed.track_id, None)
                self._track_annotations.pop(removed.track_id, None)
            else:
                # まだ残っているので減らす
                self._track_count[removed.track_id] = count - 1
                # インデックスから削除
                if removed.track_id in self._track_annotations:
                    self._track_annotations[removed.track_id].pop(id(removed), None)

            # フレーム×トラックインデックスから削除
            self._frame_track_index.pop((frame, removed.track_id), None)

        return removed

    def remove_annotation(self, annotation: Annotation, save_undo: bool = True) -> bool:
        """指定のアノテーションを削除"""
        frame = annotation.frame
        if frame not in self.annotations:
            return False

        # インデックスで存在確認（O(1)）
        if annotation.track_id is not None:
            key = (frame, annotation.track_id)
            if key not in self._frame_track_index:
                return False

        # リストから削除位置を特定
        try:
            i = self.annotations[frame].index(annotation)
            self.remove(frame, i, save_undo)
            return True
        except ValueError:
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
        return self._track_ids.copy()

    def get_track_info(self, track_id: int) -> dict:
        """トラックの情報を取得（フレーム範囲、アノテーション数）"""
        # インデックスから直接取得（O(トラック内アノテーション数)）
        if track_id not in self._track_annotations:
            return {"exists": False}

        anns = self._track_annotations[track_id].values()
        if not anns:
            return {"exists": False}

        frames = [ann.frame for ann in anns]
        return {
            "exists": True,
            "frame_min": min(frames),
            "frame_max": max(frames),
            "frame_count": len(set(frames)),
            "annotation_count": len(frames),
        }

    def remove_track(self, track_id: int, save_undo: bool = True) -> int:
        """
        指定トラックIDのアノテーションを全削除

        Args:
            track_id: 削除対象のトラックID
            save_undo: Undoスタックに保存するか

        Returns:
            削除されたアノテーション数
        """
        if save_undo:
            self._save_undo_state()

        # インデックスから対象アノテーションを直接取得（O(トラック内アノテーション数)）
        if track_id not in self._track_annotations:
            return 0

        target_anns = list(self._track_annotations[track_id].values())
        count = len(target_anns)

        # フレームごとにグループ化
        frame_to_anns: dict[int, list[Annotation]] = {}
        for ann in target_anns:
            if ann.frame not in frame_to_anns:
                frame_to_anns[ann.frame] = []
            frame_to_anns[ann.frame].append(ann)

        frames_list = list(frame_to_anns.items())
        total = len(frames_list)

        # フレームごとにまとめて削除
        for i, (frame, anns_to_remove) in enumerate(frames_list):
            # 進捗通知（100フレームごと）
            if self.progress_callback and total > 100 and i % 100 == 0:
                self.progress_callback(i, total)

            if frame not in self.annotations:
                continue

            # 該当アノテーションを削除
            for ann in anns_to_remove:
                try:
                    self.annotations[frame].remove(ann)
                except ValueError:
                    pass

            # 空になったフレームを削除
            if not self.annotations[frame]:
                del self.annotations[frame]

        # 完了通知
        if self.progress_callback and total > 100:
            self.progress_callback(total, total)

        # キャッシュ更新
        self._total_count -= count
        self._track_ids.discard(track_id)
        self._track_count.pop(track_id, None)
        self._track_annotations.pop(track_id, None)

        # フレーム×トラックインデックスから削除
        for ann in target_anns:
            self._frame_track_index.pop((ann.frame, track_id), None)

        return count

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

        # インデックスから対象アノテーションを直接取得
        if source_track_id not in self._track_annotations:
            return 0

        source_anns = list(self._track_annotations[source_track_id].values())
        count = len(source_anns)

        # 進捗通知の準備
        total = count
        for i, ann in enumerate(source_anns):
            # 進捗通知（100個ごと）
            if self.progress_callback and total > 100 and i % 100 == 0:
                self.progress_callback(i, total)

            # track_idを変更
            old_frame = ann.frame
            ann.track_id = target_track_id

            # インデックスを更新
            self._frame_track_index.pop((old_frame, source_track_id), None)
            self._frame_track_index[(old_frame, target_track_id)] = ann

        # 完了通知
        if self.progress_callback and total > 100:
            self.progress_callback(total, total)

        # キャッシュ更新（参照カウント移動）
        source_count = self._track_count.pop(source_track_id, 0)
        self._track_ids.discard(source_track_id)
        self._track_ids.add(target_track_id)
        self._track_count[target_track_id] = self._track_count.get(target_track_id, 0) + source_count

        # トラックアノテーションインデックスを移動
        if target_track_id not in self._track_annotations:
            self._track_annotations[target_track_id] = {}
        for ann_id, ann in self._track_annotations[source_track_id].items():
            self._track_annotations[target_track_id][ann_id] = ann
        self._track_annotations.pop(source_track_id, None)

        return count

    def split_track(
        self,
        track_id: int,
        split_frame: int,
        save_undo: bool = True,
    ) -> int | None:
        """
        指定トラックを指定フレーム位置で分割

        Args:
            track_id: 分割対象のトラックID
            split_frame: 分割位置（このフレーム以降が新トラックになる）
            save_undo: Undoスタックに保存するか

        Returns:
            新しいトラックID（成功時）、None（失敗時）
        """
        # トラック存在チェック
        if track_id not in self._track_annotations:
            return None

        # split_frame以降のアノテーションを収集
        annotations_to_move = []
        for ann in self._track_annotations[track_id].values():
            # ann.frameをスカラー値に変換（numpy配列の場合に対応）
            frame_num = int(np.asarray(ann.frame).item())
            if frame_num >= split_frame:
                annotations_to_move.append(ann)

        # 移動対象が0件または全件の場合は分割不可
        total_count = len(self._track_annotations[track_id])
        if len(annotations_to_move) == 0 or len(annotations_to_move) == total_count:
            return None

        # Undo状態を保存
        if save_undo:
            self._save_undo_state()

        # 新トラックIDを生成
        new_track_id = self.new_track_id()

        # アノテーションを新トラックに移動
        if new_track_id not in self._track_annotations:
            self._track_annotations[new_track_id] = {}

        for ann in annotations_to_move:
            # track_idを変更
            old_frame = ann.frame
            ann.track_id = new_track_id

            # インデックスを更新
            self._frame_track_index.pop((old_frame, track_id), None)
            self._frame_track_index[(old_frame, new_track_id)] = ann

            # トラックアノテーションインデックスを更新
            ann_id = id(ann)
            self._track_annotations[track_id].pop(ann_id, None)
            self._track_annotations[new_track_id][ann_id] = ann

        # キャッシュ更新
        moved_count = len(annotations_to_move)
        self._track_ids.add(new_track_id)
        self._track_count[new_track_id] = moved_count
        self._track_count[track_id] -= moved_count

        return new_track_id

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

        # キャッシュリセット
        self._total_count = 0
        self._track_ids.clear()
        self._track_count.clear()
        self._track_annotations.clear()
        self._frame_track_index.clear()

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

        # キャッシュ再構築
        self._rebuild_cache()

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

        # キャッシュ初期化
        store._rebuild_cache()

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
        return self._total_count

    def __iter__(self) -> Iterator[Annotation]:
        for frame in sorted(self.annotations.keys()):
            yield from self.annotations[frame]

    def get_annotation_by_frame_track(self, frame: int, track_id: int) -> Annotation | None:
        """指定フレーム・トラックIDのアノテーションを取得（O(1)）"""
        return self._frame_track_index.get((frame, track_id))

    def get_all_track_stats(self) -> dict[int, dict]:
        """全トラックの統計情報を取得（インデックス活用でO(トラック数 × 平均アノテーション数)）

        Returns:
            {track_id: {"frame_min": int, "frame_max": int, "count": int}}
        """
        result = {}
        for track_id, anns_dict in self._track_annotations.items():
            if not anns_dict:
                continue
            frames = [ann.frame for ann in anns_dict.values()]
            result[track_id] = {
                "frame_min": min(frames),
                "frame_max": max(frames),
                "count": len(frames),
            }
        return result

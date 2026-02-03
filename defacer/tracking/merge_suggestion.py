"""トラック統合サジェスト機能"""

from dataclasses import dataclass
from typing import Dict, List, Set
from defacer.gui.annotation import AnnotationStore, Annotation


class UnionFind:
    """Union-Find（素集合データ構造）"""

    def __init__(self, elements: list):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}

    def find(self, x):
        """ルートを探索（経路圧縮あり）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """2つの集合を統合"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # ランクによる最適化
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self) -> dict:
        """グループごとに要素をまとめる"""
        groups = {}
        for elem in self.parent:
            root = self.find(elem)
            if root not in groups:
                groups[root] = []
            groups[root].append(elem)
        return groups


@dataclass
class TrackInfo:
    """トラック情報"""

    track_id: int
    frame_min: int
    frame_max: int
    last_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    first_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class MergeSuggestion:
    """統合サジェスト（複数トラック対応）"""

    track_ids: list[int]  # 統合対象トラックのリスト（時系列順）
    confidence: float  # 0.0 - 1.0（グループ全体の平均信頼度）
    time_gaps: list[int]  # 各ペア間のフレーム差
    position_distances: list[float]  # 各ペア間のピクセル距離

    @property
    def track_count(self) -> int:
        """トラック数"""
        return len(self.track_ids)

    @property
    def is_multi_track(self) -> bool:
        """3つ以上のトラックを含むか"""
        return len(self.track_ids) >= 3


def collect_track_infos(store: AnnotationStore) -> list[TrackInfo]:
    """
    各トラックの情報を収集

    Args:
        store: アノテーションストア

    Returns:
        トラック情報のリスト
    """
    track_data: dict[int, list[Annotation]] = {}

    # トラックごとにアノテーションを収集
    for ann in store:
        if ann.track_id is None:
            continue
        if ann.track_id not in track_data:
            track_data[ann.track_id] = []
        track_data[ann.track_id].append(ann)

    # TrackInfoに変換
    track_infos = []
    for track_id, annotations in track_data.items():
        # フレーム順にソート
        annotations.sort(key=lambda a: a.frame)

        first_ann = annotations[0]
        last_ann = annotations[-1]

        track_info = TrackInfo(
            track_id=track_id,
            frame_min=first_ann.frame,
            frame_max=last_ann.frame,
            first_bbox=first_ann.bbox.to_tuple(),
            last_bbox=last_ann.bbox.to_tuple(),
        )
        track_infos.append(track_info)

    # フレーム順にソート
    track_infos.sort(key=lambda t: t.frame_min)

    return track_infos


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """バウンディングボックスの中心座標を計算"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """バウンディングボックスのサイズを計算"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """2点間の距離を計算"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def compute_merge_suggestions(
    store: AnnotationStore,
    max_time_gap: int = 60,
    max_position_distance: float = 200.0,
    min_confidence: float = 0.5,
) -> list[MergeSuggestion]:
    """
    統合候補を自動検出（複数トラック対応・高速化版）

    時間軸インデックスを使った効率的な候補検索により、
    O(n²)からO(n*k)に計算量を削減（kは時間窓内の平均トラック数）

    連鎖的なトラック（A→B→C→D）を自動的にグループ化して、
    複数トラックの一括統合候補として提案します。

    Args:
        store: アノテーションストア
        max_time_gap: 最大時間差（フレーム数）
        max_position_distance: 最大位置差（ピクセル）
        min_confidence: 最小信頼度

    Returns:
        統合サジェストのリスト（信頼度の高い順、複数トラック含む）
    """
    track_infos = collect_track_infos(store)

    if len(track_infos) < 2:
        return []

    # ステップ1: ペアワイズの統合候補を検出
    pairwise_candidates = []  # (track_id_a, track_id_b, confidence, time_gap, pos_distance)

    for i in range(len(track_infos)):
        track_a = track_infos[i]

        for j in range(i + 1, len(track_infos)):
            track_b = track_infos[j]

            # トラックBの開始がトラックAの終了より前なら次へ
            if track_b.frame_min <= track_a.frame_max:
                continue

            time_gap = track_b.frame_min - track_a.frame_max

            # 時間差がしきい値を超えたら早期終了
            if time_gap > max_time_gap:
                break

            # 粗い位置チェック
            ax1, ay1, ax2, ay2 = track_a.last_bbox
            bx1, by1, bx2, by2 = track_b.first_bbox
            a_center_x, a_center_y = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            b_center_x, b_center_y = (bx1 + bx2) / 2, (by1 + by2) / 2

            manhattan_distance = abs(a_center_x - b_center_x) + abs(a_center_y - b_center_y)
            if manhattan_distance > max_position_distance * 1.5:
                continue

            position_distance = _distance((a_center_x, a_center_y), (b_center_x, b_center_y))
            if position_distance > max_position_distance:
                continue

            # スコア計算
            time_score = max(0.0, 1.0 - time_gap / max_time_gap) * 0.4
            position_score = max(0.0, 1.0 - position_distance / max_position_distance) * 0.4
            size_a = _bbox_size(track_a.last_bbox)
            size_b = _bbox_size(track_b.first_bbox)

            # サイズが0の場合はスキップ（無効なバウンディングボックス）
            max_width = max(size_a[0], size_b[0])
            if max_width == 0:
                continue

            size_ratio = min(size_a[0], size_b[0]) / max_width
            size_score = size_ratio * 0.15
            movement_score = 0.05 if abs(size_a[0] - size_b[0]) < 20 else 0.0
            confidence = time_score + position_score + size_score + movement_score

            if confidence < min_confidence:
                continue

            pairwise_candidates.append(
                (track_a.track_id, track_b.track_id, confidence, time_gap, position_distance)
            )

    if not pairwise_candidates:
        return []

    # ステップ2: Union-Findでトラックをグループ化
    all_track_ids = [info.track_id for info in track_infos]
    uf = UnionFind(all_track_ids)

    # 信頼度の高いペアから順に統合
    pairwise_candidates.sort(key=lambda x: x[2], reverse=True)
    pair_info = {}  # (track_a, track_b) -> (confidence, time_gap, pos_distance)

    for track_a, track_b, conf, tg, pd in pairwise_candidates:
        uf.union(track_a, track_b)
        pair_info[(track_a, track_b)] = (conf, tg, pd)

    # ステップ3: グループごとに統合サジェストを作成
    groups = uf.get_groups()
    suggestions = []

    for root, group_track_ids in groups.items():
        # 単一トラックのグループはスキップ
        if len(group_track_ids) < 2:
            continue

        # トラックを時系列でソート
        group_track_ids.sort(key=lambda tid: next(info.frame_min for info in track_infos if info.track_id == tid))

        # グループ内のペア情報を収集
        confidences = []
        time_gaps = []
        position_distances = []

        for i in range(len(group_track_ids) - 1):
            track_a = group_track_ids[i]
            track_b = group_track_ids[i + 1]

            # ペア情報を取得
            if (track_a, track_b) in pair_info:
                conf, tg, pd = pair_info[(track_a, track_b)]
                confidences.append(conf)
                time_gaps.append(tg)
                position_distances.append(pd)
            else:
                # 直接接続されていないペア（間接的にグループ化された）
                # デフォルト値を使用
                confidences.append(min_confidence)
                time_gaps.append(0)
                position_distances.append(0.0)

        # グループ全体の平均信頼度
        avg_confidence = sum(confidences) / len(confidences) if confidences else min_confidence

        suggestion = MergeSuggestion(
            track_ids=group_track_ids,
            confidence=avg_confidence,
            time_gaps=time_gaps,
            position_distances=position_distances,
        )
        suggestions.append(suggestion)

    # 信頼度の高い順にソート
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    return suggestions

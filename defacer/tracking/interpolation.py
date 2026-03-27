"""フレーム間補間機能"""

from defacer.models import Annotation, BoundingBox
from defacer.gui.annotation import AnnotationStore


def _interpolate_between(
    store: AnnotationStore,
    ann1: Annotation,
    ann2: Annotation,
    track_id: int | None,
    is_manual: bool,
) -> int:
    """2つのアノテーション間のフレームを線形補間で埋める"""
    f1, f2 = ann1.frame, ann2.frame
    count = 0
    for frame in range(f1 + 1, f2):
        t = (frame - f1) / (f2 - f1)
        new_ann = Annotation(
            frame=frame,
            bbox=BoundingBox.interpolate(ann1.bbox, ann2.bbox, t),
            track_id=track_id,
            is_manual=is_manual,
            confidence=1.0,
        )
        store.add(new_ann, save_undo=False)
        count += 1
    return count


def interpolate_sequential_annotations(
    store: AnnotationStore,
) -> int:
    """
    連続するすべてのアノテーションを補間（track_idに関係なく）

    エクスポート用の補間関数。各フレームの最初のアノテーションを
    時系列で補間することで、track_idが異なっていても補間を実行する。

    Args:
        store: アノテーションストア

    Returns:
        追加されたアノテーション数
    """
    # 全フレームを取得してソート
    all_frames = sorted(store.get_all_frames())

    if len(all_frames) < 2:
        return 0

    count = 0

    # 連続するフレームペア間を補間
    for i in range(len(all_frames) - 1):
        f1 = all_frames[i]
        f2 = all_frames[i + 1]

        if f2 - f1 <= 1:
            continue  # 隣接フレームは補間不要

        # 各フレームの最初のアノテーションを使用
        anns1 = store.get_frame_annotations(f1)
        anns2 = store.get_frame_annotations(f2)

        if not anns1 or not anns2:
            continue

        ann1 = anns1[0]
        ann2 = anns2[0]

        # track_idは元のアノテーションから継承（なければNone）
        track_id = ann1.track_id if ann1.track_id is not None else ann2.track_id
        count += _interpolate_between(store, ann1, ann2, track_id, is_manual=False)

    return count


def interpolate_track(
    store: AnnotationStore,
    track_id: int,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> int:
    """
    指定トラックIDのアノテーションをフレーム間で線形補間

    Args:
        store: アノテーションストア
        track_id: 補間するトラックID
        start_frame: 開始フレーム（Noneの場合は最初のアノテーションから）
        end_frame: 終了フレーム（Noneの場合は最後のアノテーションまで）

    Returns:
        追加されたアノテーション数
    """
    # 指定トラックIDのアノテーションをインデックスから直接取得（O(1)）
    sorted_anns = store.get_track_annotations(track_id)

    if len(sorted_anns) < 2:
        return 0

    if start_frame is not None:
        sorted_anns = [a for a in sorted_anns if a.frame >= start_frame]
    if end_frame is not None:
        sorted_anns = [a for a in sorted_anns if a.frame <= end_frame]

    if len(sorted_anns) < 2:
        return 0

    count = 0

    # 連続するフレームペア間を補間
    for ann1, ann2 in zip(sorted_anns, sorted_anns[1:]):
        if ann2.frame - ann1.frame <= 1:
            continue  # 隣接フレームは補間不要
        count += _interpolate_between(store, ann1, ann2, track_id, is_manual=True)

    return count


def interpolate_all_tracks(
    store: AnnotationStore,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> int:
    """
    全トラックを補間

    Args:
        store: アノテーションストア
        start_frame: 開始フレーム
        end_frame: 終了フレーム

    Returns:
        追加されたアノテーション総数
    """
    track_ids = store.get_all_track_ids()

    total_count = 0
    for track_id in track_ids:
        count = interpolate_track(store, track_id, start_frame, end_frame)
        total_count += count

    return total_count

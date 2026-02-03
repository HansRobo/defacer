"""フレーム間補間機能"""

from defacer.gui.annotation import AnnotationStore, Annotation, BoundingBox


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

        # 中間フレームを生成
        for frame in range(f1 + 1, f2):
            t = (frame - f1) / (f2 - f1)
            interpolated_bbox = BoundingBox.interpolate(ann1.bbox, ann2.bbox, t)

            # track_idは元のアノテーションから継承（なければNone）
            track_id = ann1.track_id if ann1.track_id is not None else ann2.track_id

            new_ann = Annotation(
                frame=frame,
                bbox=interpolated_bbox,
                track_id=track_id,
                is_manual=False,
                confidence=1.0,
            )
            store.add(new_ann, save_undo=False)
            count += 1

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
    # 指定トラックIDのアノテーションを収集
    track_annotations: dict[int, Annotation] = {}
    for frame in store.get_all_frames():
        for ann in store.get_frame_annotations(frame):
            if ann.track_id == track_id:
                track_annotations[frame] = ann
                break

    if len(track_annotations) < 2:
        return 0

    frames = sorted(track_annotations.keys())

    if start_frame is not None:
        frames = [f for f in frames if f >= start_frame]
    if end_frame is not None:
        frames = [f for f in frames if f <= end_frame]

    if len(frames) < 2:
        return 0

    count = 0

    # 連続するフレームペア間を補間
    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]

        if f2 - f1 <= 1:
            continue  # 隣接フレームは補間不要

        ann1 = track_annotations[f1]
        ann2 = track_annotations[f2]

        # 中間フレームを生成
        for frame in range(f1 + 1, f2):
            t = (frame - f1) / (f2 - f1)
            interpolated_bbox = BoundingBox.interpolate(ann1.bbox, ann2.bbox, t)

            new_ann = Annotation(
                frame=frame,
                bbox=interpolated_bbox,
                track_id=track_id,
                is_manual=True,
                confidence=1.0,
            )
            store.add(new_ann, save_undo=False)
            count += 1

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
    # 全トラックIDを収集
    track_ids = set()
    for ann in store:
        if ann.track_id is not None:
            track_ids.add(ann.track_id)

    total_count = 0
    for track_id in track_ids:
        count = interpolate_track(store, track_id, start_frame, end_frame)
        total_count += count

    return total_count

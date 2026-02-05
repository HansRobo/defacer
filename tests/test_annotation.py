"""AnnotationStoreの重複防止機能テスト"""

import pytest
from defacer.gui.annotation import AnnotationStore, Annotation, BoundingBox


class TestDuplicatePrevention:
    """重複防止機能のテスト"""

    def test_add_duplicate_updates_existing(self):
        """同一(frame, track_id)のaddで既存が更新されること"""
        store = AnnotationStore()

        # 最初のアノテーションを追加
        ann1 = Annotation(
            frame=10,
            bbox=BoundingBox(10, 10, 50, 50),
            track_id=1,
            is_manual=True,
            confidence=0.9,
        )
        store.add(ann1, save_undo=False)

        # 同じ(frame, track_id)で2つ目を追加 → 既存が更新されるべき
        ann2 = Annotation(
            frame=10,
            bbox=BoundingBox(20, 20, 60, 60),
            track_id=1,
            is_manual=False,
            confidence=0.95,
        )
        store.add(ann2, save_undo=False)

        # フレーム10には1つだけ存在すべき
        anns = store.get_frame_annotations(10)
        assert len(anns) == 1

        # 既存オブジェクト(ann1)が更新されているべき
        assert anns[0] is ann1
        assert anns[0].bbox.x1 == 20
        assert anns[0].bbox.y1 == 20
        assert anns[0].is_manual is False
        assert anns[0].confidence == 0.95

        # 総カウントも1であるべき
        assert len(store) == 1

    def test_add_none_track_id_allows_duplicates(self):
        """track_id=Noneは重複チェック対象外"""
        store = AnnotationStore()

        # track_id=Noneのアノテーションを複数追加
        ann1 = Annotation(
            frame=10,
            bbox=BoundingBox(10, 10, 50, 50),
            track_id=None,
            is_manual=True,
        )
        ann2 = Annotation(
            frame=10,
            bbox=BoundingBox(20, 20, 60, 60),
            track_id=None,
            is_manual=True,
        )

        store.add(ann1, save_undo=False)
        store.add(ann2, save_undo=False)

        # 両方追加されるべき
        anns = store.get_frame_annotations(10)
        assert len(anns) == 2
        assert len(store) == 2

    def test_rebuild_cache_removes_duplicates(self):
        """_rebuild_cache()で重複が除去されること"""
        store = AnnotationStore()

        # 内部データに直接重複を作成
        ann1 = Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1)
        ann2 = Annotation(frame=10, bbox=BoundingBox(20, 20, 60, 60), track_id=1)
        ann3 = Annotation(frame=10, bbox=BoundingBox(30, 30, 70, 70), track_id=2)

        store.annotations[10] = [ann1, ann2, ann3]

        # キャッシュ再構築
        store._rebuild_cache()

        # 重複が除去され、先勝ちでann1が残るべき
        anns = store.get_frame_annotations(10)
        assert len(anns) == 2
        assert ann1 in anns
        assert ann2 not in anns
        assert ann3 in anns

        # 総カウントも正しく更新されるべき
        assert len(store) == 2

    def test_merge_tracks_handles_conflicts(self):
        """merge_tracks()で衝突フレームが処理されること"""
        store = AnnotationStore()

        # source_track_id=1のアノテーション
        store.add(
            Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1),
            save_undo=False,
        )
        store.add(
            Annotation(frame=20, bbox=BoundingBox(10, 10, 50, 50), track_id=1),
            save_undo=False,
        )
        store.add(
            Annotation(frame=30, bbox=BoundingBox(10, 10, 50, 50), track_id=1),
            save_undo=False,
        )

        # target_track_id=2のアノテーション（フレーム20で衝突）
        store.add(
            Annotation(frame=20, bbox=BoundingBox(20, 20, 60, 60), track_id=2),
            save_undo=False,
        )
        store.add(
            Annotation(frame=40, bbox=BoundingBox(20, 20, 60, 60), track_id=2),
            save_undo=False,
        )

        # マージ実行
        moved_count = store.merge_tracks(source_track_id=1, target_track_id=2, save_undo=False)

        # フレーム20は衝突するので削除され、10と30のみが移動される
        assert moved_count == 2

        # フレーム20にはtarget_track_id=2のみが残る
        anns_20 = store.get_frame_annotations(20)
        assert len(anns_20) == 1
        assert anns_20[0].track_id == 2
        assert anns_20[0].bbox.x1 == 20  # targetのbbox

        # フレーム10と30はtrack_id=2に変更される
        assert store.get_frame_annotations(10)[0].track_id == 2
        assert store.get_frame_annotations(30)[0].track_id == 2

        # フレーム40はそのまま
        assert store.get_frame_annotations(40)[0].track_id == 2

        # track_id=1は消滅
        assert 1 not in store.get_all_track_ids()

    def test_from_dict_cleans_duplicates(self):
        """JSONに重複がある場合のクリーニング"""
        # 重複を含むJSONデータ
        data = {
            "annotations": {
                "10": [
                    {
                        "frame": 10,
                        "bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                        "track_id": 1,
                        "is_manual": True,
                        "confidence": 0.9,
                    },
                    {
                        "frame": 10,
                        "bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60},
                        "track_id": 1,  # 重複
                        "is_manual": False,
                        "confidence": 0.95,
                    },
                    {
                        "frame": 10,
                        "bbox": {"x1": 30, "y1": 30, "x2": 70, "y2": 70},
                        "track_id": 2,
                        "is_manual": True,
                        "confidence": 1.0,
                    },
                ],
            },
            "next_track_id": 3,
        }

        # from_dictで読み込み（_rebuild_cacheが呼ばれる）
        store = AnnotationStore.from_dict(data)

        # 重複が除去され、先勝ちで最初のtrack_id=1が残るべき
        anns = store.get_frame_annotations(10)
        assert len(anns) == 2

        # track_id=1のアノテーションが1つだけ
        track1_anns = [ann for ann in anns if ann.track_id == 1]
        assert len(track1_anns) == 1
        assert track1_anns[0].bbox.x1 == 10  # 最初のものが残る

    def test_different_tracks_same_frame_allowed(self):
        """同じフレームに異なるtrack_idは追加可能"""
        store = AnnotationStore()

        ann1 = Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1)
        ann2 = Annotation(frame=10, bbox=BoundingBox(20, 20, 60, 60), track_id=2)
        ann3 = Annotation(frame=10, bbox=BoundingBox(30, 30, 70, 70), track_id=3)

        store.add(ann1, save_undo=False)
        store.add(ann2, save_undo=False)
        store.add(ann3, save_undo=False)

        anns = store.get_frame_annotations(10)
        assert len(anns) == 3
        assert len(store) == 3

    def test_same_track_different_frames_allowed(self):
        """同じtrack_idの異なるフレームは追加可能"""
        store = AnnotationStore()

        ann1 = Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1)
        ann2 = Annotation(frame=20, bbox=BoundingBox(20, 20, 60, 60), track_id=1)
        ann3 = Annotation(frame=30, bbox=BoundingBox(30, 30, 70, 70), track_id=1)

        store.add(ann1, save_undo=False)
        store.add(ann2, save_undo=False)
        store.add(ann3, save_undo=False)

        assert len(store.get_frame_annotations(10)) == 1
        assert len(store.get_frame_annotations(20)) == 1
        assert len(store.get_frame_annotations(30)) == 1
        assert len(store) == 3

    def test_integration_retrack_scenario(self):
        """再トラッキングシナリオの統合テスト"""
        store = AnnotationStore()

        # 初期状態: track_id=1のアノテーション
        ann1 = Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1)
        ann2 = Annotation(frame=20, bbox=BoundingBox(20, 20, 60, 60), track_id=1)
        store.add(ann1, save_undo=False)
        store.add(ann2, save_undo=False)

        # 再トラッキングでtrack_idを変更（retrack_dialogがやること）
        ann1.track_id = 10
        ann2.track_id = 10

        # キャッシュ再構築
        store._rebuild_cache()

        # track_id=10として正しくインデックスされている
        assert store.get_annotation_by_frame_track(10, 10) is ann1
        assert store.get_annotation_by_frame_track(20, 10) is ann2
        assert 10 in store.get_all_track_ids()
        assert 1 not in store.get_all_track_ids()

    def test_integration_interpolation_with_duplicates(self):
        """補間操作で重複が発生しないことを確認"""
        from defacer.tracking.interpolation import interpolate_track

        store = AnnotationStore()

        # キーフレーム
        ann1 = Annotation(frame=10, bbox=BoundingBox(10, 10, 50, 50), track_id=1)
        ann2 = Annotation(frame=20, bbox=BoundingBox(30, 30, 70, 70), track_id=1)
        store.add(ann1, save_undo=False)
        store.add(ann2, save_undo=False)

        # 補間実行
        interpolate_track(store, track_id=1, start_frame=10, end_frame=20)

        # 各フレームにtrack_id=1が1つずつ存在するべき
        for frame in range(10, 21):
            anns = store.get_frame_annotations(frame)
            track1_anns = [ann for ann in anns if ann.track_id == 1]
            assert len(track1_anns) == 1, f"フレーム{frame}でtrack_id=1の重複発生"

        # 2回補間しても重複しない（既存を更新）
        interpolate_track(store, track_id=1, start_frame=10, end_frame=20)

        for frame in range(10, 21):
            anns = store.get_frame_annotations(frame)
            track1_anns = [ann for ann in anns if ann.track_id == 1]
            assert len(track1_anns) == 1, f"2回目補間後、フレーム{frame}でtrack_id=1の重複発生"

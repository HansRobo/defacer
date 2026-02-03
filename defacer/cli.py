"""CLIエントリーポイント"""

# OpenCVとPyQt5のQtプラグイン競合を回避（他のインポートより先に実行）
import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ.pop("QT_PLUGIN_PATH", None)

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="defacer",
        description="動画内の顔を自動検知してモザイク処理を行うソフトウェア",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.2.2",
    )

    subparsers = parser.add_subparsers(dest="command", help="サブコマンド")

    # GUIコマンド
    gui_parser = subparsers.add_parser("gui", help="GUIモードで起動")
    gui_parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="入力動画ファイル（オプション）",
    )

    # 自動処理コマンド
    auto_parser = subparsers.add_parser("auto", help="自動処理モード")
    auto_parser.add_argument(
        "input",
        type=Path,
        help="入力動画ファイル",
    )
    auto_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="出力動画ファイル（デフォルト: input_defaced.mp4）",
    )
    auto_parser.add_argument(
        "--detector",
        choices=["yolo11-face"],
        default="yolo11-face",
        help="使用する顔検知器（デフォルト: yolo11-face）",
    )
    auto_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="検出信頼度の閾値（デフォルト: 0.5）",
    )
    auto_parser.add_argument(
        "--mosaic-type",
        choices=["mosaic", "blur", "solid"],
        default="mosaic",
        help="モザイクの種類（デフォルト: mosaic）",
    )
    auto_parser.add_argument(
        "--block-size",
        type=int,
        default=10,
        help="モザイクのブロックサイズ（デフォルト: 10）",
    )
    auto_parser.add_argument(
        "--bbox-scale",
        type=float,
        default=1.1,
        help="バウンディングボックスの拡大率（デフォルト: 1.1）",
    )
    auto_parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="トラッキングを無効化",
    )
    auto_parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="出力品質 CRF値（0-51、低いほど高品質、デフォルト: 18）",
    )
    auto_parser.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
        default="medium",
        help="エンコード速度プリセット（デフォルト: medium）",
    )

    return parser


def run_gui(args: argparse.Namespace) -> int:
    """GUIモードを実行"""
    try:
        from defacer.gui.app import main as gui_main
        return gui_main(args.input)
    except ImportError as e:
        print(f"エラー: GUI依存関係が不足しています: {e}", file=sys.stderr)
        print("pip install defacer[all] でインストールしてください", file=sys.stderr)
        return 1


def run_auto(args: argparse.Namespace) -> int:
    """自動処理モードを実行"""
    if not args.input.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input}", file=sys.stderr)
        return 1

    # 出力パスのデフォルト設定
    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_defaced")

    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"検知器: {args.detector}")
    print(f"閾値: {args.threshold}")
    print(f"モザイク: {args.mosaic_type}")
    print(f"トラッキング: {'無効' if args.no_tracking else '有効'}")
    print()

    # 依存関係のチェック
    try:
        from defacer.detection import get_available_detectors, create_detector
        from defacer.video.reader import VideoReader
        from defacer.video.writer import check_ffmpeg_available
        from defacer.gui.annotation import AnnotationStore, Annotation, BoundingBox
        from defacer.anonymization.mosaic import MosaicAnonymizer
        from defacer.anonymization.blur import GaussianBlurAnonymizer, SolidFillAnonymizer
        from defacer.pipeline.processor import export_processed_video
    except ImportError as e:
        print(f"エラー: 依存関係が不足しています: {e}", file=sys.stderr)
        return 1

    # FFmpegチェック
    if not check_ffmpeg_available():
        print("エラー: FFmpegが見つかりません。インストールしてください。", file=sys.stderr)
        return 1

    # 検出器のチェック
    available = get_available_detectors()
    if args.detector not in available:
        if not available:
            print(f"エラー: 利用可能な検出器がありません。", file=sys.stderr)
            print("pip install ultralytics huggingface-hub でYOLOv11をインストールしてください", file=sys.stderr)
            return 1
        print(f"警告: {args.detector}が利用できません。{available[0]}を使用します。", file=sys.stderr)
        args.detector = available[0]

    # 検出器を作成
    try:
        detector = create_detector(args.detector, confidence_threshold=args.threshold)
    except Exception as e:
        print(f"エラー: 検出器の初期化に失敗: {e}", file=sys.stderr)
        return 1

    # 匿名化器を作成
    if args.mosaic_type == "mosaic":
        anonymizer = MosaicAnonymizer(block_size=args.block_size)
    elif args.mosaic_type == "blur":
        anonymizer = GaussianBlurAnonymizer(kernel_size=99)
    else:
        anonymizer = SolidFillAnonymizer()

    # トラッカーを作成（オプション）
    tracker = None
    if not args.no_tracking:
        try:
            from defacer.tracking import create_tracker
            tracker = create_tracker(max_age=30, min_hits=3)
            print("DeepSORTトラッキングを使用")
        except ImportError as e:
            print(f"エラー: トラッキングが利用できません: {e}", file=sys.stderr)
            print("pip install deep-sort-realtime でインストールしてください", file=sys.stderr)
            return 1

    # 動画を読み込み
    try:
        reader = VideoReader(args.input)
    except Exception as e:
        print(f"エラー: 動画の読み込みに失敗: {e}", file=sys.stderr)
        return 1

    print(f"解像度: {reader.width}x{reader.height}")
    print(f"フレーム数: {reader.frame_count}")
    print(f"FPS: {reader.fps:.2f}")
    print()

    # 顔検出を実行
    print("顔を検出中...")
    store = AnnotationStore()
    next_track_id = 1

    with tqdm(total=reader.frame_count, desc="検出") as pbar:
        for frame_num, frame in reader:
            detections = detector.detect(frame)

            if tracker:
                tracked = tracker.update(detections, frame)
                for t in tracked:
                    x1, y1, x2, y2 = t.bbox
                    # バウンディングボックスを拡大
                    if args.bbox_scale != 1.0:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        w = int((x2 - x1) * args.bbox_scale)
                        h = int((y2 - y1) * args.bbox_scale)
                        x1 = max(0, cx - w // 2)
                        y1 = max(0, cy - h // 2)
                        x2 = min(reader.width, cx + w // 2)
                        y2 = min(reader.height, cy + h // 2)

                    ann = Annotation(
                        frame=frame_num,
                        bbox=BoundingBox(x1, y1, x2, y2),
                        track_id=t.track_id,
                        is_manual=False,
                        confidence=t.confidence,
                    )
                    store.add(ann, save_undo=False)
            else:
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    # バウンディングボックスを拡大
                    if args.bbox_scale != 1.0:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        w = int((x2 - x1) * args.bbox_scale)
                        h = int((y2 - y1) * args.bbox_scale)
                        x1 = max(0, cx - w // 2)
                        y1 = max(0, cy - h // 2)
                        x2 = min(reader.width, cx + w // 2)
                        y2 = min(reader.height, cy + h // 2)

                    ann = Annotation(
                        frame=frame_num,
                        bbox=BoundingBox(x1, y1, x2, y2),
                        track_id=next_track_id,
                        is_manual=False,
                        confidence=det.confidence,
                    )
                    store.add(ann, save_undo=False)
                    next_track_id += 1

            pbar.update(1)

    reader.release()
    print(f"\n検出完了: {len(store)}件の顔領域")
    print()

    # 動画をエクスポート
    print("動画をエクスポート中...")

    def progress_callback(current, total):
        pass  # tqdmで表示するので不要

    try:
        with tqdm(total=reader.frame_count, desc="エクスポート") as pbar:
            def update_pbar(current, total):
                pbar.n = current
                pbar.refresh()

            success = export_processed_video(
                args.input,
                args.output,
                store,
                anonymizer,
                ellipse=True,
                bbox_scale=1.0,  # 検出時に既に拡大済み
                codec="libx264",
                crf=args.crf,
                preset=args.preset,
                progress_callback=update_pbar,
            )
    except Exception as e:
        print(f"\nエラー: エクスポートに失敗: {e}", file=sys.stderr)
        return 1

    if success:
        print(f"\n完了: {args.output}")
        return 0
    else:
        print("\nエラー: エクスポートに失敗しました", file=sys.stderr)
        return 1


def main() -> int:
    """メインエントリーポイント"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "gui":
        return run_gui(args)
    elif args.command == "auto":
        return run_auto(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

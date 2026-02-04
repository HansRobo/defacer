# Defacer

動画内の顔を自動検知してモザイク処理を行うソフトウェア

## 特徴

- **自動顔検知**: YOLOv11による高精度な顔検出（WIDERFACE訓練済み、Easy=94.2%, Medium=92.1%）
- **手動アノテーション**: 検知漏れを手動で指定可能
- **フレーム間補間**: 開始/終了フレームを指定して中間を自動生成
- **複数のモザイク方式**: ピクセル化モザイク、ぼかし、塗りつぶし
- **GUI/CLI対応**: 使いやすいGUIと自動処理可能なCLI

## インストール

### pipx（推奨）

```bash
# グローバルインストール
pipx install defacer

# 使用
defacer gui
```

### pip（venv内）

```bash
pip install defacer
```

### 開発環境

```bash
git clone https://github.com/kotaroyoshimoto/defacer.git
cd defacer

# 仮想環境を作成してインストール
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 必要なシステムパッケージ

```bash
# Ubuntu/Debian
sudo apt install ffmpeg libgl1-mesa-glx libglib2.0-0 libqt5gui5

# Fedora/RHEL
sudo dnf install ffmpeg mesa-libGL glib2 qt5-qtbase-gui
```

## 使用方法

### GUIモード

```bash
defacer gui [動画ファイル]
```

#### 操作方法

- **Space**: 再生/一時停止
- **←/→** または **A/D**: フレーム移動
- **マウスドラッグ**: 描画モードで領域を描画
- **Delete**: 選択領域を削除
- **F**: 選択を次フレームにコピー
- **Ctrl+D**: 自動顔検出を実行
- **Ctrl+S**: アノテーションを保存
- **Ctrl+Shift+E**: 動画をエクスポート

### CLIモード（自動処理）

```bash
# 基本的な使用
defacer auto input.mp4 -o output.mp4

# オプション指定
defacer auto input.mp4 -o output.mp4 \
  --threshold 0.4 \
  --mosaic-type blur \
  --bbox-scale 1.2 \
  --crf 20
```

#### CLIオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--detector` | 検出器（yolo11-face） | yolo11-face |
| `--threshold` | 検出信頼度の閾値（0.0-1.0） | 0.5 |
| `--mosaic-type` | モザイク種類（mosaic/blur/solid） | mosaic |
| `--block-size` | モザイクブロックサイズ | 10 |
| `--bbox-scale` | 領域拡大率 | 1.1 |
| `--no-tracking` | トラッキングを無効化 | - |
| `--crf` | 出力品質（0-51、低いほど高品質） | 18 |
| `--preset` | エンコード速度 | medium |

## オプション機能

基本インストールにはYOLOv11顔検出（HuggingFaceから自動ダウンロード）とDeepSORTトラッキングが含まれています。以下のオプション機能を追加できます：

```bash
# 全オプション機能をインストール
pip install -e ".[all]"
```

## ワークフロー例

1. **動画を開く**: GUI起動後、`Ctrl+O`で動画を開く
2. **自動検出**: `Ctrl+D`で顔を自動検出
3. **手動修正**: 検知漏れがあれば描画モードで手動追加
4. **補間**: 開始/終了フレームで領域を指定し、補間ボタンで中間フレームを自動生成
5. **エクスポート**: `Ctrl+Shift+E`でモザイク処理した動画を出力

## プロジェクト構造

```
defacer/
├── detection/      # 顔検知（YOLOv11-Face）
├── tracking/       # トラッキング（DeepSORT, Simple）
├── anonymization/  # モザイク処理（Mosaic, Blur, Solid）
├── video/          # 動画入出力（OpenCV, FFmpeg）
├── pipeline/       # 処理パイプライン
└── gui/            # PyQt5 GUI
    ├── app.py              # メインウィンドウ
    ├── video_player.py     # 動画プレーヤー
    ├── annotation.py       # アノテーション管理
    ├── detection_dialog.py # 自動検出ダイアログ
    └── export_dialog.py    # エクスポートダイアログ
```

## トラブルシューティング

### Qt platform plugin エラー

環境変数でプラットフォームを指定してください：

```bash
# X11で起動
QT_QPA_PLATFORM=xcb defacer gui

# Waylandで起動
QT_QPA_PLATFORM=wayland defacer gui
```

### GPU関連の問題

デフォルトでは CPU で動作します。GPU を使用する場合は、適切な PyTorch と CUDA/ROCm をインストールしてください。

### メモリ不足エラー

大きな動画を処理する場合、トラッキングを無効化することでメモリ使用量を削減できます：

```bash
defacer auto input.mp4 -o output.mp4 --no-tracking
```

## ライセンス

MIT

## 依存関係

### 必須

- opencv-python: 動画処理とGUI表示
- PyQt5: GUIフレームワーク
- numpy: 数値計算
- tqdm: 進捗表示
- ultralytics: YOLOv11顔検出
- huggingface-hub: YOLOv11モデルの自動ダウンロード
- deep-sort-realtime: DeepSORTトラッキング

### オプション

- ffmpeg-python: 高度な動画エンコード

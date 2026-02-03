# Defacer

動画内の顔を自動検知してモザイク処理を行うソフトウェア

## 特徴

- **自動顔検知**: RetinaFaceによる高精度な顔検出
- **手動アノテーション**: 検知漏れを手動で指定可能
- **フレーム間補間**: 開始/終了フレームを指定して中間を自動生成
- **複数のモザイク方式**: ピクセル化モザイク、ぼかし、塗りつぶし
- **GUI/CLI対応**: 使いやすいGUIと自動処理可能なCLI

## インストール

### uvを使用（推奨）

```bash
# グローバルインストール
uv tool install defacer

# すぐに使用可能
defacer gui
```

### 開発環境

#### 自動インストールスクリプト（推奨）

環境（CUDA/ROCm/CPU）を自動検出して適切な依存関係をインストールします：

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/defacer.git
cd defacer

# 環境を自動検出してインストール
./scripts/install.sh

# 仮想環境を有効化
source .venv/bin/activate

# GUIを起動
defacer gui
```

#### インストールスクリプトのオプション

```bash
# CUDA環境を明示的に指定
./scripts/install.sh --cuda

# ROCm環境を明示的に指定
./scripts/install.sh --rocm

# CPU環境を明示的に指定
./scripts/install.sh --cpu

# 開発依存関係も含める
./scripts/install.sh --dev

# 全オプション依存関係を含める
./scripts/install.sh --all
```

#### 手動インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/defacer.git
cd defacer

# 仮想環境を作成してインストール
uv venv
source .venv/bin/activate
uv pip install -e .

# GUIを起動
defacer gui
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
| `--detector` | 検出器（retinaface/yolov8-face） | retinaface |
| `--threshold` | 検出信頼度の閾値（0.0-1.0） | 0.5 |
| `--mosaic-type` | モザイク種類（mosaic/blur/solid） | mosaic |
| `--block-size` | モザイクブロックサイズ | 10 |
| `--bbox-scale` | 領域拡大率 | 1.1 |
| `--no-tracking` | トラッキングを無効化 | - |
| `--crf` | 出力品質（0-51、低いほど高品質） | 18 |
| `--preset` | エンコード速度 | medium |

## オプション機能

基本インストールにはRetinaFace顔検出が含まれています。以下のオプション機能を追加できます：

```bash
# YOLOv8顔検出を追加（RetinaFaceの代替）
uv pip install -e ".[yolo]"

# トラッキング機能を追加（フレーム間の顔追跡）
uv pip install -e ".[tracking]"

# 全オプション機能をインストール
uv pip install -e ".[all]"
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
├── detection/      # 顔検知（RetinaFace, YOLOv8-Face）
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

### GPUが検出されない

TensorFlowはCPUで動作します。GPUを使用する場合：

1. **CUDA環境**: CUDAドライバーとcudnnをインストール後、`./scripts/install.sh --cuda`を実行
2. **ROCm環境**: ROCmドライバーをインストール後、`./scripts/install.sh --rocm`を実行

インストールスクリプトは環境に応じて適切なTensorFlow/PyTorchバージョンを自動選択します。

### メモリ不足エラー

大きな動画を処理する場合、フレームサンプリングを調整してください：

```bash
defacer auto input.mp4 -o output.mp4 --sample-rate 5
```

## ライセンス

MIT

## 依存関係

### 必須

- opencv-python: 動画処理とGUI表示
- PyQt5: GUIフレームワーク
- numpy: 数値計算
- tqdm: 進捗表示
- retina-face: RetinaFace顔検出
- tf-keras: TensorFlowバックエンド

### オプション

- ultralytics: YOLOv8顔検出（RetinaFaceの代替）
- deep-sort-realtime: DeepSORTトラッキング
- ffmpeg-python: 高度な動画エンコード

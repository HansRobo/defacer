#!/bin/bash
# ROCm環境でdefacerを実行するためのラッパースクリプト
# gfx1103（Radeon 780M）などの新しいGPUでGPUを強制使用する

# 使用方法:
#   ./scripts/run_with_rocm.sh gui
#   ./scripts/run_with_rocm.sh auto input.mp4 -o output.mp4
#
# 注意: このスクリプトはGPUを強制使用しますが、不安定な場合があります。
#       安定性を優先する場合は、通常のdefacerコマンドを使用してください（CPUモード）。

# ROCm GPU強制使用フラグ
export DEFACER_FORCE_ROCM=1

# ROCm環境変数の設定
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # gfx1103をgfx1100としてエミュレート
export HSA_ENABLE_SDMA=0  # SDMA（System DMA）を無効化して安定性向上
export MIOPEN_DEBUG_CONV_DIRECT=0  # MIOpen convolution問題を回避
export PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"

# デバッグ用（必要に応じてコメント解除）
# export AMD_LOG_LEVEL=3
# export AMD_SERIALIZE_KERNEL=3

echo "=== ROCm GPU強制使用モード ==="
echo "⚠️  警告: gfx1103など新しいGPUアーキテクチャは公式PyTorchで完全にサポートされていません。"
echo "    不安定な場合は、通常のdefacerコマンドを使用してください（CPUモード）。"
echo ""
echo "環境変数:"
echo "  DEFACER_FORCE_ROCM=$DEFACER_FORCE_ROCM"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
echo "  MIOPEN_DEBUG_CONV_DIRECT=$MIOPEN_DEBUG_CONV_DIRECT"
echo ""

# 引数をdefacerコマンドに渡す
exec defacer "$@"

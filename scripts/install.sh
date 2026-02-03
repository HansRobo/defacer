#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
INSTALL_DEV=false
INSTALL_ALL=false
FORCE_ENV=""

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

環境に応じて defacer の依存関係を自動インストールします。

OPTIONS:
  --cuda      CUDA環境を強制指定
  --rocm      ROCm環境を強制指定
  --cpu       CPU環境を強制指定
  --dev       開発依存関係も含める
  --all       全オプション依存関係を含める
  -h, --help  このヘルプを表示

EXAMPLES:
  # 自動検出（推奨）
  $0

  # CUDA環境を明示的に指定
  $0 --cuda

  # 開発依存関係も含めてインストール
  $0 --dev

  # すべての依存関係をインストール
  $0 --all
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            FORCE_ENV="cuda"
            shift
            ;;
        --rocm)
            FORCE_ENV="rocm"
            shift
            ;;
        --cpu)
            FORCE_ENV="cpu"
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# Detect environment
detect_environment() {
    if [[ -n "$FORCE_ENV" ]]; then
        echo "$FORCE_ENV"
        return
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "cuda"
            return
        fi
    fi

    # Check for AMD ROCm (multiple detection methods)
    # Method 1: rocm-smi command
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            echo "rocm"
            return
        fi
    fi
    # Method 2: rocminfo command
    if command -v rocminfo &> /dev/null; then
        if rocminfo &> /dev/null; then
            echo "rocm"
            return
        fi
    fi
    # Method 3: Check for ROCm installation directory
    if [[ -d "/opt/rocm" ]] || [[ -d "/opt/rocm-"* ]]; then
        echo "rocm"
        return
    fi

    # Default to CPU
    echo "cpu"
}

# Check for uv or pip
get_pip_command() {
    if command -v uv &> /dev/null; then
        echo "uv pip"
    elif command -v pip &> /dev/null; then
        echo "pip"
    else
        print_error "pip または uv が見つかりません"
        exit 1
    fi
}

# Main installation
main() {
    print_info "defacer インストールスクリプトを開始します..."

    cd "$PROJECT_DIR"

    # Detect environment
    ENV=$(detect_environment)
    print_info "検出された環境: $ENV"

    # Get pip command
    PIP_CMD=$(get_pip_command)
    print_info "使用するパッケージマネージャ: $PIP_CMD"

    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        print_info "仮想環境を作成しています..."
        python3 -m venv .venv
        print_success "仮想環境を作成しました"
    else
        print_info "既存の仮想環境を使用します"
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip (only if using pip, not uv)
    if [[ "$PIP_CMD" == "pip" ]]; then
        print_info "pip をアップグレードしています..."
        python -m pip install --upgrade pip
    fi

    # Install base dependencies
    print_info "基本パッケージをインストールしています..."
    $PIP_CMD install -e .

    # Install environment-specific dependencies
    case $ENV in
        cuda)
            print_info "CUDA対応パッケージをインストールしています..."
            $PIP_CMD install "tensorflow[and-cuda]"
            # PyTorch (standard PyPI version has CUDA support)
            $PIP_CMD install torch torchvision
            print_success "CUDA対応パッケージのインストールが完了しました"
            ;;
        rocm)
            print_info "ROCm対応パッケージをインストールしています..."
            print_warning "tensorflow-rocm は手動でインストールする必要があります:"
            print_warning "  pip install tensorflow-rocm"
            # PyTorch for ROCm
            print_info "PyTorch (ROCm版) をインストールしています..."
            $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
            print_success "ROCm対応PyTorchのインストールが完了しました"
            ;;
        cpu)
            print_info "CPU版パッケージをインストールしています..."
            $PIP_CMD install tensorflow-cpu
            $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            print_success "CPU版パッケージのインストールが完了しました"
            ;;
    esac

    # Install optional dependencies
    if [[ "$INSTALL_ALL" == true ]]; then
        print_info "全オプション依存関係をインストールしています..."
        $PIP_CMD install -e ".[all]"
    fi

    # Install development dependencies
    if [[ "$INSTALL_DEV" == true ]]; then
        print_info "開発依存関係をインストールしています..."
        $PIP_CMD install -e ".[dev]"
    fi

    # Disable cv2/qt/plugins (PyQt5 conflict workaround)
    print_info "OpenCV Qt プラグインを無効化しています..."
    export QT_QPA_PLATFORM_PLUGIN_PATH=""

    print_success "インストールが完了しました!"
    echo ""
    print_info "仮想環境を有効化するには:"
    echo "  source .venv/bin/activate"
    echo ""
    print_info "defacer を実行するには:"
    echo "  defacer gui"
    echo ""

    # Show GPU info if available
    case $ENV in
        cuda)
            print_info "GPU情報を確認しています..."
            python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'検出されたGPU数: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]" 2>/dev/null || print_warning "GPU情報の取得に失敗しました"
            ;;
        rocm)
            print_info "ROCm情報を確認しています..."
            python -c "import torch; print(f'PyTorch ROCm available: {torch.cuda.is_available()}'); print(f'ROCm devices: {torch.cuda.device_count()}')" 2>/dev/null || print_warning "ROCm情報の取得に失敗しました"
            ;;
        cpu)
            print_info "CPU環境で動作します"
            ;;
    esac
}

# Run main
main

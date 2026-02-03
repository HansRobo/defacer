"""Defacerのエントリーポイント"""

# OpenCVとPyQt5のQtプラグイン競合を回避（他のインポートより先に実行）
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

from defacer.cli import main

if __name__ == "__main__":
    main()

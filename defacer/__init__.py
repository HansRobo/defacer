"""Defacer - 動画内の顔を自動検知してモザイク処理を行うソフトウェア"""

import os

# OpenCVとPyQt5のQtプラグイン競合を回避
# cv2がインポートされる前にQT_QPA_PLATFORM_PLUGIN_PATHを削除
_cv2_qt_path = None
for key in list(os.environ.keys()):
    if "QT" in key and "cv2" in os.environ.get(key, ""):
        _cv2_qt_path = os.environ.pop(key, None)

# OpenCVのQtプラグインディレクトリを使用しないようにする
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

__version__ = "0.4.0"

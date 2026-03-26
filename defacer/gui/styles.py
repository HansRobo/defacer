"""GUIスタイルシート定数"""

ERROR_STYLE = "color: red;"
MONOSPACE_LABEL_STYLE = "font-family: monospace; font-size: 12px;"

DARK_THEME = """
    QMainWindow, QWidget {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    QGroupBox {
        border: 1px solid #555;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
    }
    QPushButton {
        background-color: #404040;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 5px 10px;
    }
    QPushButton:hover {
        background-color: #505050;
    }
    QPushButton:pressed {
        background-color: #606060;
    }
    QPushButton:disabled {
        background-color: #353535;
        color: #707070;
    }
    QPushButton:checked {
        background-color: #0078d4;
        border-color: #0078d4;
    }
    QSlider::groove:horizontal {
        border: 1px solid #444;
        height: 8px;
        background: #333;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #0078d4;
        border: 1px solid #0078d4;
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }
    QSlider::sub-page:horizontal {
        background: #0078d4;
        border-radius: 4px;
    }
    QMenuBar {
        background-color: #2d2d2d;
    }
    QMenuBar::item:selected {
        background-color: #404040;
    }
    QMenu {
        background-color: #2d2d2d;
        border: 1px solid #444;
    }
    QMenu::item:selected {
        background-color: #0078d4;
    }
    QStatusBar {
        background-color: #252525;
    }
    QToolBar {
        background-color: #2d2d2d;
        border: none;
        spacing: 5px;
    }
    QSpinBox {
        background-color: #404040;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px;
    }
"""

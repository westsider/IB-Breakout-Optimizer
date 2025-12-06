#!/usr/bin/env python
"""
IB Breakout Optimizer - Desktop Application

PySide6-based desktop UI for backtesting and optimizing the IB Breakout strategy.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QIcon

from desktop_ui.main_window import MainWindow


def setup_dark_theme(app: QApplication):
    """Apply a dark theme to the application."""
    app.setStyle("Fusion")

    palette = QPalette()

    # Base colors
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))

    app.setPalette(palette)

    # Additional styling via stylesheet
    app.setStyleSheet("""
        QToolTip {
            color: #dcdcdc;
            background-color: #2a2a2a;
            border: 1px solid #555555;
            padding: 4px;
        }
        QTabWidget::pane {
            border: 1px solid #444444;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #dcdcdc;
            padding: 8px 16px;
            border: 1px solid #444444;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            border-bottom: 2px solid #2a82da;
        }
        QTabBar::tab:hover:!selected {
            background-color: #3d3d3d;
        }
        QGroupBox {
            border: 1px solid #444444;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #2d2d2d;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 6px 16px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #3d3d3d;
            border-color: #2a82da;
        }
        QPushButton:pressed {
            background-color: #2a82da;
        }
        QPushButton:disabled {
            background-color: #252525;
            color: #666666;
        }
        QPushButton#primary {
            background-color: #2a82da;
            border-color: #2a82da;
        }
        QPushButton#primary:hover {
            background-color: #3a92ea;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #252525;
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 4px 8px;
            selection-background-color: #2a82da;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #2a82da;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid #888888;
            margin-right: 6px;
        }
        QProgressBar {
            border: 1px solid #444444;
            border-radius: 4px;
            text-align: center;
            background-color: #252525;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            border-radius: 3px;
        }
        QTableWidget {
            background-color: #1e1e1e;
            alternate-background-color: #252525;
            gridline-color: #333333;
            border: 1px solid #444444;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QTableWidget::item:selected {
            background-color: #2a82da;
        }
        QHeaderView::section {
            background-color: #2d2d2d;
            border: 1px solid #444444;
            padding: 6px;
            font-weight: bold;
        }
        QScrollBar:vertical {
            background-color: #1e1e1e;
            width: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            border-radius: 4px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar:horizontal {
            background-color: #1e1e1e;
            height: 12px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            border-radius: 4px;
            min-width: 20px;
            margin: 2px;
        }
        QLabel#metric-value {
            font-size: 18px;
            font-weight: bold;
        }
        QLabel#metric-label {
            color: #888888;
            font-size: 11px;
        }
    """)


def main():
    """Main entry point for the desktop application."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("IB Breakout Optimizer")
    app.setOrganizationName("Trading Tools")

    # Set application icon
    resources_dir = Path(__file__).parent / "resources"
    for ext in ['.png', '.jpg', '.ico']:
        icon_path = resources_dir / f"app_icon{ext}"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break

    # Apply dark theme
    setup_dark_theme(app)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# week4/labs/base.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QTextEdit
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class LabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(80)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 11px;")

        controls = self.build_controls()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(controls)
        splitter.setSizes([700, 300])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        layout.addWidget(self.log_box)

    def build_controls(self) -> QWidget:
        raise NotImplementedError

    def run_training(self):
        raise NotImplementedError

    def log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

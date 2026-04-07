import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from labs.lab1_1d import Lab1Widget
from labs.lab2_projectile import Lab2Widget
from labs.lab3_overfitting import Lab3Widget
from labs.lab4_pendulum import Lab4Widget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 4 — Physics Neural Networks")
        self.setMinimumSize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(Lab1Widget(), "Lab 1: 1D 근사")
        tabs.addTab(Lab2Widget(), "Lab 2: 포사체")
        tabs.addTab(Lab3Widget(), "Lab 3: 오버피팅")
        tabs.addTab(Lab4Widget(), "Lab 4: 진자")

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

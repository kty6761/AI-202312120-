import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton
)
from labs.base import LabWidget

MODEL_CONFIGS = [
    ("Underfit [4]",           [4],             0.0, "#e55"),
    ("Good Fit [32,16]",       [32, 16],         0.2, "#4caf50"),
    ("Overfit [256,128,64,32]", [256,128,64,32], 0.0, "#9c27b0"),
]


class Lab3Widget(LabWidget):
    def build_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.epochs_edit = QLineEdit("200")
        self.noise_edit = QLineEdit("0.3")

        form.addRow("Epochs:", self.epochs_edit)
        form.addRow("노이즈 크기:", self.noise_edit)

        self.run_btn = QPushButton("▶ 학습 시작")
        self.run_btn.clicked.connect(self.run_training)

        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addStretch()
        return widget

    def run_training(self):
        epochs = int(self.epochs_edit.text())
        noise = float(self.noise_edit.text())
        self.log(f"학습 시작: epochs={epochs}, noise={noise}")

        np.random.seed(42)
        x = np.linspace(-3, 3, 200)
        y = np.sin(2 * x) + 0.5 * x + np.random.normal(0, noise, len(x))
        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]

        x_test = np.linspace(-3, 3, 300)
        y_ideal = np.sin(2 * x_test) + 0.5 * x_test

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(x, y, alpha=0.3, s=10, color="#aaa", label="데이터")
        ax.plot(x_test, y_ideal, color="#4f8ef7", linewidth=2, label="실제 함수")

        for name, layer_sizes, dropout, color in MODEL_CONFIGS:
            model_layers = []
            for n in layer_sizes:
                model_layers.append(keras.layers.Dense(n, activation="relu"))
                if dropout > 0:
                    model_layers.append(keras.layers.Dropout(dropout))
            model_layers.append(keras.layers.Dense(1, activation="linear"))
            model = keras.Sequential(model_layers)
            model.compile(optimizer="adam", loss="mse")
            model.fit(x, y, epochs=epochs, validation_split=0.2, verbose=0)

            y_pred = model.predict(x_test, verbose=0).flatten()
            ax.plot(x_test, y_pred, label=name, color=color,
                    linewidth=1.5, linestyle="--")
            self.log(f"{name} 완료")

        ax.set_title("과적합 vs 과소적합 비교")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

        os.makedirs("outputs", exist_ok=True)
        self.figure.savefig(
            "outputs/03_overfitting_comparison.png", dpi=150, bbox_inches="tight"
        )
        self.log("저장됨: outputs/03_overfitting_comparison.png")

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, QPushButton
)
from labs.base import LabWidget

FUNCTIONS = {
    "sin(x)":             lambda x: np.sin(x),
    "cos(x)+0.5sin(2x)":  lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
    "x·sin(x)":           lambda x: x * np.sin(x),
}


class Lab1Widget(LabWidget):
    def build_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.func_combo = QComboBox()
        self.func_combo.addItems(list(FUNCTIONS.keys()))
        self.epochs_edit = QLineEdit("3000")
        self.layers_edit = QLineEdit("128, 128, 64")
        self.lr_edit = QLineEdit("0.001")

        form.addRow("함수:", self.func_combo)
        form.addRow("Epochs:", self.epochs_edit)
        form.addRow("Layers:", self.layers_edit)
        form.addRow("Learning Rate:", self.lr_edit)

        self.run_btn = QPushButton("▶ 학습 시작")
        self.run_btn.clicked.connect(self.run_training)

        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addStretch()
        return widget

    def run_training(self):
        func_name = self.func_combo.currentText()
        fn = FUNCTIONS[func_name]
        epochs = int(self.epochs_edit.text())
        layers = [int(v.strip()) for v in self.layers_edit.text().split(",")]
        lr = float(self.lr_edit.text())

        self.log(f"학습 시작: {func_name}, epochs={epochs}, layers={layers}")

        # Data — shuffle so validation_split covers full range
        x = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = fn(x)
        x_norm = x / (3 * np.pi)

        # Build model
        model_layers = [keras.layers.Dense(n, activation="tanh") for n in layers]
        model_layers.append(keras.layers.Dense(1, activation="linear"))
        model = keras.Sequential(model_layers)
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")

        hist = model.fit(
            x_norm, y,
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
        )
        self.log(f"완료 — MSE: {hist.history['loss'][-1]:.6f}")

        # Plot
        x_test = np.linspace(-3 * np.pi, 3 * np.pi, 500)
        y_true = fn(x_test)
        y_pred = model.predict(x_test / (3 * np.pi), verbose=0).flatten()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_test, y_true, label="실제값", color="#4f8ef7", linewidth=2)
        ax.plot(x_test, y_pred, label="예측값", color="#e55",
                linewidth=1.5, linestyle="--")
        ax.set_title(f"1D 함수 근사: {func_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

        os.makedirs("outputs", exist_ok=True)
        self.figure.savefig(
            "outputs/perfect_1d_approximation.png", dpi=150, bbox_inches="tight"
        )
        self.log("저장됨: outputs/perfect_1d_approximation.png")

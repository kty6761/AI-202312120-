import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton
)
from labs.base import LabWidget


class Lab2Widget(LabWidget):
    def build_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.v0_edit = QLineEdit("30")
        self.theta_edit = QLineEdit("45")
        self.epochs_edit = QLineEdit("2000")

        form.addRow("초기속력 v₀ (m/s):", self.v0_edit)
        form.addRow("발사각 θ (°):", self.theta_edit)
        form.addRow("Epochs:", self.epochs_edit)

        self.run_btn = QPushButton("▶ 학습 시작")
        self.run_btn.clicked.connect(self.run_training)

        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addStretch()
        return widget

    def run_training(self):
        v0_test = float(self.v0_edit.text())
        theta_test = float(self.theta_edit.text())
        epochs = int(self.epochs_edit.text())
        g = 9.81

        self.log(f"학습 시작: v₀={v0_test} m/s, θ={theta_test}°, epochs={epochs}")

        # Training data
        np.random.seed(42)
        n = 2000
        v0 = np.random.uniform(10, 50, n)
        theta_deg = np.random.uniform(10, 80, n)
        theta_rad = np.radians(theta_deg)
        t_max = 2 * v0 * np.sin(theta_rad) / g
        t = np.random.uniform(0, 1, n) * t_max
        x_data = v0 * np.cos(theta_rad) * t + np.random.normal(0, 0.1, n)
        y_data = (v0 * np.sin(theta_rad) * t
                  - 0.5 * g * t**2
                  + np.random.normal(0, 0.1, n))
        y_data = np.maximum(y_data, 0)

        X = np.stack([v0, theta_deg, t], axis=1)
        Y = np.stack([x_data, y_data], axis=1)
        X_mean, X_std = X.mean(0), X.std(0)
        X_norm = (X - X_mean) / X_std

        model = keras.Sequential([
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(2, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
        hist = model.fit(X_norm, Y, epochs=epochs, validation_split=0.2, verbose=0)
        self.log(f"완료 — loss: {hist.history['loss'][-1]:.4f}")

        # Test trajectory
        theta_rad_test = np.radians(theta_test)
        t_max_test = 2 * v0_test * np.sin(theta_rad_test) / g
        t_vals = np.linspace(0, t_max_test, 100)

        x_true = v0_test * np.cos(theta_rad_test) * t_vals
        y_true = np.maximum(
            v0_test * np.sin(theta_rad_test) * t_vals - 0.5 * g * t_vals**2, 0
        )

        X_test = np.stack([
            np.full(100, v0_test),
            np.full(100, theta_test),
            t_vals,
        ], axis=1)
        X_test_norm = (X_test - X_mean) / X_std
        pred = model.predict(X_test_norm, verbose=0)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x_true, y_true, label="실제 궤적", color="#4f8ef7", linewidth=2)
        ax.plot(pred[:, 0], np.maximum(pred[:, 1], 0),
                label="예측 궤적", color="#e55", linewidth=1.5, linestyle="--")
        ax.set_title(f"포사체 운동: v₀={v0_test} m/s, θ={theta_test}°")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

        os.makedirs("outputs", exist_ok=True)
        self.figure.savefig(
            "outputs/02_projectile_trajectories.png", dpi=150, bbox_inches="tight"
        )
        self.log("저장됨: outputs/02_projectile_trajectories.png")

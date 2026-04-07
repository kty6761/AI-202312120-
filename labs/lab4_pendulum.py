import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QPushButton
)
from labs.base import LabWidget


def rk4_pendulum(theta0, omega0, L, g, dt, n_steps):
    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    theta[0], omega[0] = theta0, omega0
    for i in range(n_steps - 1):
        def d(th, om): return om, -(g / L) * np.sin(th)
        k1 = d(theta[i], omega[i])
        k2 = d(theta[i] + 0.5*dt*k1[0], omega[i] + 0.5*dt*k1[1])
        k3 = d(theta[i] + 0.5*dt*k2[0], omega[i] + 0.5*dt*k2[1])
        k4 = d(theta[i] + dt*k3[0],     omega[i] + dt*k3[1])
        theta[i+1] = theta[i] + (dt/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        omega[i+1] = omega[i] + (dt/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    return theta, omega


def compute_period_rk4(L, theta0_deg, g=9.81):
    theta0 = np.radians(theta0_deg)
    T0 = 2 * np.pi * np.sqrt(L / g)
    dt = T0 / 1000
    n = int(10 * T0 / dt)
    theta, _ = rk4_pendulum(theta0, 0.0, L, g, dt, n)
    crossings = []
    for i in range(1, len(theta)):
        if (theta[i-1] >= 0 and theta[i] < 0) or (theta[i-1] < 0 and theta[i] >= 0):
            crossings.append(i * dt)
        if len(crossings) == 2:
            break
    if len(crossings) == 2:
        return 2 * (crossings[1] - crossings[0])
    return T0


class Lab4Widget(LabWidget):
    def build_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.L_edit = QLineEdit("1.0")
        self.theta_edit = QLineEdit("30")
        self.epochs_edit = QLineEdit("2000")

        form.addRow("길이 L (m):", self.L_edit)
        form.addRow("각도 θ₀ (°):", self.theta_edit)
        form.addRow("Epochs:", self.epochs_edit)

        self.run_btn = QPushButton("▶ 학습 시작")
        self.run_btn.clicked.connect(self.run_training)

        layout.addLayout(form)
        layout.addWidget(self.run_btn)
        layout.addStretch()
        return widget

    def run_training(self):
        L_test = float(self.L_edit.text())
        theta_test = float(self.theta_edit.text())
        epochs = int(self.epochs_edit.text())
        g = 9.81

        self.log(f"학습 시작: L={L_test} m, θ₀={theta_test}°, epochs={epochs}")

        # Training data
        np.random.seed(42)
        n = 2000
        L_vals = np.random.uniform(0.1, 3.0, n)
        theta_vals = np.random.uniform(5, 80, n)
        self.log("RK4 데이터 생성 중 (약 10초)...")
        T_vals = np.array([compute_period_rk4(l, t) for l, t in zip(L_vals, theta_vals)])

        X = np.stack([L_vals, theta_vals], axis=1)
        X_mean, X_std = X.mean(0), X.std(0)
        X_norm = (X - X_mean) / X_std

        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
        hist = model.fit(X_norm, T_vals, epochs=epochs, validation_split=0.2, verbose=0)
        self.log(f"완료 — loss: {hist.history['loss'][-1]:.6f}")

        # Angle sweep for test L
        angles = np.linspace(5, 80, 50)
        T_true = np.array([compute_period_rk4(L_test, a) for a in angles])
        X_plot = np.stack([np.full(50, L_test), angles], axis=1)
        T_pred = model.predict((X_plot - X_mean) / X_std, verbose=0).flatten()

        # RK4 time simulation
        T0 = 2 * np.pi * np.sqrt(L_test / g)
        dt = T0 / 500
        n_sim = int(4 * T0 / dt)
        theta_sim, _ = rk4_pendulum(np.radians(theta_test), 0.0, L_test, g, dt, n_sim)
        t_sim = np.arange(n_sim) * dt

        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax1.plot(angles, T_true, label="RK4 실제", color="#4f8ef7", linewidth=2)
        ax1.plot(angles, T_pred, label="NN 예측", color="#e55",
                 linewidth=1.5, linestyle="--")
        ax1.set_title(f"주기 예측 (L={L_test} m)")
        ax1.set_xlabel("초기 각도 (°)")
        ax1.set_ylabel("주기 T (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(122)
        ax2.plot(t_sim, np.degrees(theta_sim), color="#4caf50", linewidth=1.5)
        ax2.set_title(f"RK4 시뮬레이션 (θ₀={theta_test}°)")
        ax2.set_xlabel("시간 (s)")
        ax2.set_ylabel("각도 (°)")
        ax2.grid(True, alpha=0.3)

        self.canvas.draw()

        os.makedirs("outputs", exist_ok=True)
        self.figure.savefig(
            "outputs/04_pendulum_prediction.png", dpi=150, bbox_inches="tight"
        )
        self.log("저장됨: outputs/04_pendulum_prediction.png")

"""
Week 3: 신경망 기초 — PySide6 인터랙티브 교육 앱
Lab 1: Perceptron
Lab 2: Activation Functions
Lab 3: Forward Propagation
Lab 4: MLP (NumPy)
Lab 5: Universal Approximation
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QMessageBox, QProgressBar, QSizePolicy, QFrame,
    QScrollArea,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QFont, QPalette, QColor

# ──────────────────────────────────────────────
#  STYLE
# ──────────────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e0e0e0;
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    background: #13151f;
}
QTabBar::tab {
    background: #1c1f2e;
    color: #8888aa;
    padding: 10px 22px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: bold;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #2563eb;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background: #252840;
    color: #ccccff;
}
QGroupBox {
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
    color: #7c9aff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover { background-color: #3b82f6; }
QPushButton:pressed { background-color: #1d4ed8; }
QPushButton:disabled { background-color: #2a2d3a; color: #555; }
QPushButton#reset_btn {
    background-color: #374151;
}
QPushButton#reset_btn:hover { background-color: #4b5563; }
QPushButton#warn_btn {
    background-color: #7c3aed;
}
QPushButton#warn_btn:hover { background-color: #8b5cf6; }
QComboBox {
    background-color: #1c1f2e;
    border: 1px solid #2a2d3a;
    border-radius: 5px;
    padding: 5px 10px;
    color: #e0e0e0;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #1c1f2e;
    color: #e0e0e0;
    selection-background-color: #2563eb;
}
QSpinBox, QDoubleSpinBox {
    background-color: #1c1f2e;
    border: 1px solid #2a2d3a;
    border-radius: 5px;
    padding: 5px 8px;
    color: #e0e0e0;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #2a2d3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #2563eb;
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}
QTableWidget {
    background-color: #1c1f2e;
    border: 1px solid #2a2d3a;
    gridline-color: #2a2d3a;
    border-radius: 5px;
}
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #2563eb; }
QHeaderView::section {
    background-color: #252840;
    color: #7c9aff;
    padding: 5px;
    border: 1px solid #2a2d3a;
    font-weight: bold;
}
QProgressBar {
    background-color: #1c1f2e;
    border: 1px solid #2a2d3a;
    border-radius: 5px;
    text-align: center;
    color: white;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}
QLabel#info_label {
    color: #7c9aff;
    font-size: 12px;
    padding: 4px;
}
QLabel#result_label {
    color: #34d399;
    font-size: 13px;
    font-weight: bold;
    padding: 4px;
}
QLabel#warn_label {
    color: #f87171;
    font-size: 12px;
    font-weight: bold;
    padding: 4px;
}
QCheckBox {
    spacing: 6px;
    color: #e0e0e0;
}
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 2px solid #2a2d3a;
    border-radius: 4px;
    background: #1c1f2e;
}
QCheckBox::indicator:checked {
    background: #2563eb;
    border-color: #2563eb;
}
QScrollArea { border: none; background: transparent; }
"""

# ──────────────────────────────────────────────
#  Matplotlib 공통 설정
# ──────────────────────────────────────────────
MPLSTYLE = {
    "figure.facecolor": "#13151f",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#aab4cc",
    "axes.titlecolor":  "#c8d0ff",
    "xtick.color":      "#7080a0",
    "ytick.color":      "#7080a0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "text.color":       "#c8d0ff",
    "lines.linewidth":  2.0,
    "legend.facecolor": "#1c1f2e",
    "legend.edgecolor": "#2a2d3a",
    "legend.labelcolor":"#c8d0ff",
}
for k, v in MPLSTYLE.items():
    plt.rcParams[k] = v

COLORS = ["#3b82f6", "#f59e0b", "#34d399", "#f87171", "#a78bfa", "#fb923c"]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, nrows=1, ncols=1, figsize=(8, 4)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.fig.patch.set_facecolor("#13151f")
        if nrows == 1 and ncols == 1:
            self.axes = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.subplots(nrows, ncols)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear_axes(self):
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flat:
                ax.cla()
                ax.set_facecolor("#1a1d2e")
                ax.grid(True)
        else:
            self.axes.cla()
            self.axes.set_facecolor("#1a1d2e")
            self.axes.grid(True)


# ══════════════════════════════════════════════
#  LAB 1: PERCEPTRON
# ══════════════════════════════════════════════
class Perceptron:
    def __init__(self, lr=0.1):
        self.weights = np.zeros(2)
        self.bias = 0.0
        self.lr = lr
        self.loss_history = []

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train(self, X, y, epochs):
        self.loss_history = []
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                err = yi - pred
                self.weights += self.lr * err * xi
                self.bias    += self.lr * err
                errors += abs(err)
            self.loss_history.append(errors / len(y))
        return self.loss_history


GATE_DATA = {
    "AND": (np.array([[0,0],[0,1],[1,0],[1,1]], float), np.array([0,0,0,1])),
    "OR":  (np.array([[0,0],[0,1],[1,0],[1,1]], float), np.array([0,1,1,1])),
    "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]], float), np.array([0,1,1,0])),
}


class PerceptronTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # ── 왼쪽 패널
        left = QWidget(); left.setFixedWidth(280)
        lv = QVBoxLayout(left); lv.setSpacing(14)

        ctrl = QGroupBox("파라미터")
        form = QFormLayout(ctrl); form.setSpacing(10)

        self.gate_cb = QComboBox()
        self.gate_cb.addItems(["AND", "OR", "XOR"])
        self.gate_cb.currentTextChanged.connect(self._on_gate_change)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0); self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(0.1); self.lr_spin.setDecimals(3)

        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000); self.epoch_spin.setValue(50)

        form.addRow("게이트:", self.gate_cb)
        form.addRow("학습률 (η):", self.lr_spin)
        form.addRow("에포크:", self.epoch_spin)

        self.train_btn = QPushButton("▶  학습 시작")
        self.train_btn.clicked.connect(self._train)
        reset_btn = QPushButton("↺  초기화"); reset_btn.setObjectName("reset_btn")
        reset_btn.clicked.connect(self._reset)

        lv.addWidget(ctrl)
        lv.addWidget(self.train_btn)
        lv.addWidget(reset_btn)

        # 결과
        res = QGroupBox("결과")
        rv = QVBoxLayout(res); rv.setSpacing(6)
        self.w_label   = QLabel("w₁: —   w₂: —"); self.w_label.setObjectName("result_label")
        self.b_label   = QLabel("bias: —");          self.b_label.setObjectName("result_label")
        self.acc_label = QLabel("정확도: —");         self.acc_label.setObjectName("result_label")
        self.xor_warn  = QLabel("⚠ XOR은 단일 퍼셉트론으로\n선형 분리 불가능합니다.")
        self.xor_warn.setObjectName("warn_label"); self.xor_warn.hide()
        for w in [self.w_label, self.b_label, self.acc_label, self.xor_warn]:
            rv.addWidget(w)

        # 진리표
        tbl_grp = QGroupBox("진리표")
        tv = QVBoxLayout(tbl_grp)
        self.truth_tbl = QTableWidget(4, 3)
        self.truth_tbl.setHorizontalHeaderLabels(["x₁","x₂","y"])
        self.truth_tbl.verticalHeader().setVisible(False)
        self.truth_tbl.setFixedHeight(130)
        tv.addWidget(self.truth_tbl)
        self._fill_truth_table("AND")

        lv.addWidget(res)
        lv.addWidget(tbl_grp)
        lv.addStretch()

        # ── 오른쪽 캔버스
        right = QWidget()
        rv2 = QVBoxLayout(right); rv2.setContentsMargins(0,0,0,0)
        self.canvas = MplCanvas(1, 2, figsize=(10, 4))
        toolbar = NavigationToolbar2QT(self.canvas, self)
        rv2.addWidget(toolbar); rv2.addWidget(self.canvas)

        layout.addWidget(left)
        layout.addWidget(right, 1)
        self._reset()

    def _fill_truth_table(self, gate):
        X, y = GATE_DATA[gate]
        for i, (xi, yi) in enumerate(zip(X, y)):
            for j, v in enumerate([xi[0], xi[1], yi]):
                item = QTableWidgetItem(str(int(v)))
                item.setTextAlignment(Qt.AlignCenter)
                self.truth_tbl.setItem(i, j, item)

    def _on_gate_change(self, gate):
        self.xor_warn.setVisible(gate == "XOR")
        self._fill_truth_table(gate)

    def _train(self):
        gate = self.gate_cb.currentText()
        X, y = GATE_DATA[gate]
        lr = self.lr_spin.value()
        epochs = self.epoch_spin.value()

        self.model = Perceptron(lr=lr)
        self.model.train(X, y, epochs)

        # 정확도
        preds = np.array([self.model.predict(x) for x in X])
        acc = np.mean(preds == y) * 100

        self.w_label.setText(f"w₁: {self.model.weights[0]:.4f}   w₂: {self.model.weights[1]:.4f}")
        self.b_label.setText(f"bias: {self.model.bias:.4f}")
        self.acc_label.setText(f"정확도: {acc:.0f}%")

        self._plot(X, y, gate)

    def _plot(self, X, y, gate):
        self.canvas.clear_axes()
        ax1, ax2 = self.canvas.axes[0], self.canvas.axes[1]

        # Loss
        ax1.plot(self.model.loss_history, color=COLORS[0])
        ax1.set_title("학습 Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Error Rate")

        # 결정 경계
        xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
        Z = np.array([self.model.predict(np.array([a, b])) for a, b in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, alpha=0.25, cmap="RdYlBu", levels=[-0.5, 0.5, 1.5])
        for xi, yi in zip(X, y):
            color = COLORS[0] if yi == 1 else COLORS[3]
            ax2.scatter(xi[0], xi[1], color=color, s=120, zorder=5,
                        edgecolors="white", linewidths=1.5)
        ax2.set_title(f"{gate} 결정 경계"); ax2.set_xlim(-0.3, 1.3); ax2.set_ylim(-0.3, 1.3)
        ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _reset(self):
        self.model = None
        self.w_label.setText("w₁: —   w₂: —")
        self.b_label.setText("bias: —")
        self.acc_label.setText("정확도: —")
        self.canvas.clear_axes()
        ax1, ax2 = self.canvas.axes[0], self.canvas.axes[1]
        ax1.set_title("학습 Loss"); ax2.set_title("결정 경계")
        ax1.text(0.5, 0.5, "학습을 시작하세요", ha='center', va='center',
                 transform=ax1.transAxes, color="#555577", fontsize=14)
        ax2.text(0.5, 0.5, "학습을 시작하세요", ha='center', va='center',
                 transform=ax2.transAxes, color="#555577", fontsize=14)
        self.canvas.draw()
        self._on_gate_change(self.gate_cb.currentText())


# ══════════════════════════════════════════════
#  LAB 2: ACTIVATION FUNCTIONS
# ══════════════════════════════════════════════
def sigmoid(x):        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(x):      s = sigmoid(x); return s * (1 - s)
def tanh_f(x):         return np.tanh(x)
def tanh_d(x):         return 1 - np.tanh(x)**2
def relu(x):           return np.maximum(0, x)
def relu_d(x):         return (x > 0).astype(float)
def lrelu(x, a=0.01):  return np.where(x > 0, x, a * x)
def lrelu_d(x, a=0.01):return np.where(x > 0, 1.0, a)


ACT_FUNCS = {
    "Sigmoid":    (sigmoid,  sigmoid_d),
    "Tanh":       (tanh_f,   tanh_d),
    "ReLU":       (relu,     relu_d),
    "Leaky ReLU": (lrelu,    lrelu_d),
}


class ActivationTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ── 상단 컨트롤
        ctrl_row = QHBoxLayout(); ctrl_row.setSpacing(20)

        # 체크박스
        cb_grp = QGroupBox("함수 선택")
        cb_lay = QHBoxLayout(cb_grp)
        self.checks = {}
        defaults = {"Sigmoid": True, "Tanh": True, "ReLU": True, "Leaky ReLU": False}
        for name, checked in defaults.items():
            cb = QCheckBox(name); cb.setChecked(checked)
            cb.stateChanged.connect(self._plot)
            self.checks[name] = cb
            cb_lay.addWidget(cb)

        # 슬라이더
        rng_grp = QGroupBox("x 범위")
        rng_lay = QHBoxLayout(rng_grp)
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setRange(3, 20); self.range_slider.setValue(10)
        self.range_label = QLabel("±10")
        self.range_slider.valueChanged.connect(lambda v: (
            self.range_label.setText(f"±{v}"), self._plot()))
        rng_lay.addWidget(self.range_slider); rng_lay.addWidget(self.range_label)

        # α
        alpha_grp = QGroupBox("Leaky ReLU α")
        alpha_lay = QHBoxLayout(alpha_grp)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.5); self.alpha_spin.setValue(0.01)
        self.alpha_spin.setSingleStep(0.01); self.alpha_spin.setDecimals(3)
        self.alpha_spin.valueChanged.connect(self._plot)
        alpha_lay.addWidget(self.alpha_spin)

        # 포인트 하이라이트
        pt_grp = QGroupBox("포인트 x")
        pt_lay = QHBoxLayout(pt_grp)
        self.point_spin = QDoubleSpinBox()
        self.point_spin.setRange(-20, 20); self.point_spin.setValue(2.0)
        self.point_spin.valueChanged.connect(self._plot)
        pt_lay.addWidget(self.point_spin)

        ctrl_row.addWidget(cb_grp, 3)
        ctrl_row.addWidget(rng_grp, 2)
        ctrl_row.addWidget(alpha_grp, 1)
        ctrl_row.addWidget(pt_grp, 1)

        # ── 캔버스
        self.canvas = MplCanvas(1, 2, figsize=(12, 5))
        toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addLayout(ctrl_row)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)

        self._plot()

    def _plot(self):
        xmax = self.range_slider.value()
        x = np.linspace(-xmax, xmax, 500)
        xp = self.point_spin.value()
        alpha = self.alpha_spin.value()

        self.canvas.clear_axes()
        ax_f, ax_d = self.canvas.axes[0], self.canvas.axes[1]
        ax_f.set_title("활성화 함수"); ax_d.set_title("Gradient (미분)")

        for i, (name, (f, df)) in enumerate(ACT_FUNCS.items()):
            if not self.checks[name].isChecked():
                continue
            color = COLORS[i]
            if name == "Leaky ReLU":
                y  = f(x, alpha)
                dy = df(x, alpha)
                yp = f(np.array([xp]), alpha)[0]
                dp = df(np.array([xp]), alpha)[0]
            else:
                y  = f(x)
                dy = df(x)
                yp = float(f(np.array([xp]))[0])
                dp = float(df(np.array([xp]))[0])

            ax_f.plot(x, y,  color=color, label=name)
            ax_d.plot(x, dy, color=color, label=name)

            # 포인트 하이라이트
            ax_f.scatter([xp], [yp], color=color, s=80, zorder=6, edgecolors="white")
            ax_d.scatter([xp], [dp], color=color, s=80, zorder=6, edgecolors="white")

        # Vanishing 구간
        for name in ["Sigmoid", "Tanh"]:
            if self.checks[name].isChecked():
                ax_d.axvspan(-xmax, -5, alpha=0.07, color="#f87171")
                ax_d.axvspan(5,  xmax, alpha=0.07, color="#f87171")
                ax_d.text((-xmax - 5) / 2, ax_d.get_ylim()[1] * 0.9 if ax_d.get_ylim()[1] > 0 else 0.05,
                          "Vanishing\n구간", color="#f87171", fontsize=9, ha='center')
                break

        ax_f.axhline(0, color="#333355", lw=1); ax_f.axvline(0, color="#333355", lw=1)
        ax_d.axhline(0, color="#333355", lw=1); ax_d.axvline(0, color="#333355", lw=1)
        ax_f.legend(); ax_d.legend()
        ax_f.set_xlabel("x"); ax_d.set_xlabel("x")

        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════
#  LAB 3: FORWARD PROPAGATION
# ══════════════════════════════════════════════
STEP_NAMES = ["초기화", "z₁ 계산", "a₁ = ReLU(z₁)", "z₂ 계산", "a₂ = σ(z₂)"]


class ForwardPropTab(QWidget):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.X = None
        self._build_ui()
        self._init_network()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # ── 왼쪽
        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        # 구조 설정
        struct_grp = QGroupBox("네트워크 구조")
        sf = QFormLayout(struct_grp); sf.setSpacing(8)
        self.in_spin  = QSpinBox(); self.in_spin.setRange(1, 4); self.in_spin.setValue(2)
        self.hid_spin = QSpinBox(); self.hid_spin.setRange(1, 6); self.hid_spin.setValue(3)
        self.out_spin = QSpinBox(); self.out_spin.setRange(1, 3); self.out_spin.setValue(1)
        self.act_cb   = QComboBox(); self.act_cb.addItems(["ReLU", "Sigmoid", "Tanh"])
        sf.addRow("입력 크기:", self.in_spin)
        sf.addRow("은닉 크기:", self.hid_spin)
        sf.addRow("출력 크기:", self.out_spin)
        sf.addRow("은닉 활성화:", self.act_cb)

        # 입력 벡터
        inp_grp = QGroupBox("입력 벡터 X")
        inp_lay = QVBoxLayout(inp_grp)
        self.x_inputs = []
        self.x_container = QWidget()
        self.x_form = QFormLayout(self.x_container)
        self._rebuild_x_inputs()
        self.in_spin.valueChanged.connect(self._rebuild_x_inputs)
        inp_lay.addWidget(self.x_container)

        # 버튼
        self.step_btn  = QPushButton("▶  다음 단계")
        self.step_btn.clicked.connect(self._next_step)
        reinit_btn = QPushButton("↺  가중치 재생성"); reinit_btn.setObjectName("reset_btn")
        reinit_btn.clicked.connect(self._init_network)

        # 진행 표시
        self.step_label = QLabel("단계: 초기화"); self.step_label.setObjectName("info_label")
        self.prog = QProgressBar(); self.prog.setRange(0, 4); self.prog.setValue(0)

        # 결과 테이블
        tbl_grp = QGroupBox("행렬 값")
        tv = QVBoxLayout(tbl_grp)
        self.val_tbl = QTableWidget(); self.val_tbl.setFixedHeight(180)
        tv.addWidget(self.val_tbl)

        lv.addWidget(struct_grp)
        lv.addWidget(inp_grp)
        lv.addWidget(self.step_label)
        lv.addWidget(self.prog)
        lv.addWidget(self.step_btn)
        lv.addWidget(reinit_btn)
        lv.addWidget(tbl_grp)
        lv.addStretch()

        # ── 오른쪽
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0)
        self.canvas = MplCanvas(1, 2, figsize=(10, 5))
        toolbar = NavigationToolbar2QT(self.canvas, self)
        rv.addWidget(toolbar); rv.addWidget(self.canvas)

        layout.addWidget(left)
        layout.addWidget(right, 1)

    def _rebuild_x_inputs(self):
        for i in reversed(range(self.x_form.count())):
            self.x_form.removeRow(0)
        self.x_inputs = []
        n = self.in_spin.value()
        for i in range(n):
            sp = QDoubleSpinBox(); sp.setRange(-5, 5); sp.setValue(round(np.random.randn(), 2))
            sp.setSingleStep(0.1); sp.setDecimals(2)
            self.x_form.addRow(f"x{i+1}:", sp)
            self.x_inputs.append(sp)

    def _init_network(self):
        n_in  = self.in_spin.value()
        n_hid = self.hid_spin.value()
        n_out = self.out_spin.value()
        self.W1 = np.random.randn(n_in, n_hid) * 0.5
        self.b1 = np.random.randn(n_hid) * 0.1
        self.W2 = np.random.randn(n_hid, n_out) * 0.5
        self.b2 = np.random.randn(n_out) * 0.1
        self.step = 0
        self.step_label.setText("단계: 초기화 — [다음 단계] 버튼을 누르세요")
        self.prog.setValue(0)
        self.step_btn.setEnabled(True)
        self.val_tbl.clear()
        self._draw_network(highlight=-1)

    def _get_activation(self, z):
        act = self.act_cb.currentText()
        if act == "ReLU":     return np.maximum(0, z)
        elif act == "Sigmoid":return sigmoid(z)
        else:                  return np.tanh(z)

    def _next_step(self):
        self.step += 1
        self.step_label.setText(f"단계 {self.step}/4: {STEP_NAMES[self.step]}")
        self.prog.setValue(self.step)

        X = np.array([[sp.value() for sp in self.x_inputs]])

        if self.step == 1:
            z1 = X @ self.W1 + self.b1
            self._show_table(["z₁"], z1)
            self.z1 = z1
        elif self.step == 2:
            self.a1 = self._get_activation(self.z1)
            self._show_table(["a₁"], self.a1)
        elif self.step == 3:
            self.z2 = self.a1 @ self.W2 + self.b2
            self._show_table(["z₂"], self.z2)
        elif self.step == 4:
            self.a2 = sigmoid(self.z2)
            self._show_table(["a₂ (출력)"], self.a2)
            self.step_btn.setEnabled(False)

        self._draw_network(highlight=self.step - 1)

    def _show_table(self, headers, matrix):
        data = matrix.flatten()
        self.val_tbl.setRowCount(1)
        self.val_tbl.setColumnCount(len(data))
        self.val_tbl.setHorizontalHeaderLabels([f"{headers[0]}[{i}]" for i in range(len(data))])
        self.val_tbl.verticalHeader().setVisible(False)
        for j, v in enumerate(data):
            item = QTableWidgetItem(f"{v:.4f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.val_tbl.setItem(0, j, item)

    def _draw_network(self, highlight=-1):
        self.canvas.clear_axes()
        ax_net = self.canvas.axes[0]
        ax_val = self.canvas.axes[1]

        n_in  = self.in_spin.value()
        n_hid = self.hid_spin.value()
        n_out = self.out_spin.value()
        layers = [n_in, n_hid, n_out]
        layer_names = ["Input", "Hidden", "Output"]
        x_pos = [0, 2, 4]

        # 엣지
        edge_colors = ["#2a2d3a", "#2563eb", "#34d399"]
        ec_idx = [0, 0 if highlight < 0 else (1 if highlight < 2 else 0),
                     0 if highlight < 0 else (1 if highlight >= 2 else 0)]
        for li in range(2):
            color = COLORS[0] if (li == 0 and 0 <= highlight <= 1) or (li == 1 and 2 <= highlight <= 3) else "#2a2d3a"
            alpha = 0.8 if color != "#2a2d3a" else 0.3
            for ni in range(layers[li]):
                y_src = ni - (layers[li] - 1) / 2
                for nj in range(layers[li+1]):
                    y_dst = nj - (layers[li+1] - 1) / 2
                    ax_net.plot([x_pos[li], x_pos[li+1]], [y_src, y_dst],
                                color=color, alpha=alpha, lw=1.2, zorder=1)

        # 노드
        for li, (n, name) in enumerate(zip(layers, layer_names)):
            for ni in range(n):
                y = ni - (n - 1) / 2
                active = (li == 1 and highlight in [1, 2, 3]) or (li == 2 and highlight == 3)
                c = COLORS[0] if (li == 0) else (COLORS[2] if (li == 1 and highlight in [1,2]) else
                     (COLORS[1] if (li == 2 and highlight == 3) else "#2a2d3a"))
                circle = plt.Circle((x_pos[li], y), 0.28, color=c, zorder=3)
                ax_net.add_patch(circle)
                ax_net.text(x_pos[li], y, str(ni+1), ha='center', va='center',
                            fontsize=9, color='white', fontweight='bold', zorder=4)
            ax_net.text(x_pos[li], -(n)/2 - 0.7, name, ha='center', color="#8899cc", fontsize=10)

        ax_net.set_xlim(-0.8, 4.8); ax_net.set_ylim(-max(layers)/2-1.2, max(layers)/2+0.8)
        ax_net.set_aspect('equal'); ax_net.axis('off')
        ax_net.set_title("네트워크 다이어그램")

        # 값 플롯
        if self.step >= 1 and hasattr(self, 'z1'):
            labels, vals, colors_bar = [], [], []
            if self.step >= 1:
                for i, v in enumerate(self.z1.flatten()):
                    labels.append(f"z₁[{i}]"); vals.append(v); colors_bar.append(COLORS[0])
            if self.step >= 2:
                for i, v in enumerate(self.a1.flatten()):
                    labels.append(f"a₁[{i}]"); vals.append(v); colors_bar.append(COLORS[2])
            if self.step >= 3:
                for i, v in enumerate(self.z2.flatten()):
                    labels.append(f"z₂[{i}]"); vals.append(v); colors_bar.append(COLORS[1])
            if self.step >= 4:
                for i, v in enumerate(self.a2.flatten()):
                    labels.append(f"a₂[{i}]"); vals.append(v); colors_bar.append(COLORS[3])
            bars = ax_val.bar(labels, vals, color=colors_bar, edgecolor="#0f1117")
            ax_val.axhline(0, color="#555577", lw=1)
            ax_val.set_title("각 단계 값"); ax_val.set_ylabel("값")
            for bar, v in zip(bars, vals):
                ax_val.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f"{v:.3f}", ha='center', va='bottom', fontsize=8, color='#c8d0ff')
        else:
            ax_val.text(0.5, 0.5, "단계를 진행하세요", ha='center', va='center',
                        transform=ax_val.transAxes, color="#555577", fontsize=13)
            ax_val.set_title("각 단계 값")

        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════
#  LAB 4: MLP
# ══════════════════════════════════════════════
class MLP:
    def __init__(self, hidden=4, lr=0.1):
        self.W1 = np.random.randn(2, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.sum(0) / m
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float)
        dW1 = X.T @ dz1 / m
        db1 = dz1.sum(0) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X, y):
        out = self.forward(X)
        self.backward(X, y)
        return float(np.mean((out - y.reshape(-1,1))**2))


XOR_X = np.array([[0,0],[0,1],[1,0],[1,1]], float)
XOR_Y = np.array([0,1,1,0], float)


class MLPTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loss_history = []
        self.current_epoch = 0
        self.total_epochs = 5000
        self.timer = QTimer()
        self.timer.timeout.connect(self._batch_train)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # ── 왼쪽
        left = QWidget(); left.setFixedWidth(290)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        param_grp = QGroupBox("파라미터")
        pf = QFormLayout(param_grp); pf.setSpacing(8)
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(1, 16); self.hidden_spin.setValue(4)
        self.lr_spin2    = QDoubleSpinBox(); self.lr_spin2.setRange(0.001, 1.0)
        self.lr_spin2.setValue(0.1); self.lr_spin2.setSingleStep(0.01); self.lr_spin2.setDecimals(3)
        self.epoch_spin2 = QSpinBox(); self.epoch_spin2.setRange(100, 50000); self.epoch_spin2.setValue(5000)
        self.epoch_spin2.setSingleStep(500)
        pf.addRow("은닉층 뉴런 수:", self.hidden_spin)
        pf.addRow("학습률:", self.lr_spin2)
        pf.addRow("에포크:", self.epoch_spin2)

        self.start_btn = QPushButton("▶  학습 시작")
        self.start_btn.clicked.connect(self._start_train)
        self.stop_btn = QPushButton("■  중지"); self.stop_btn.setObjectName("warn_btn")
        self.stop_btn.clicked.connect(self._stop_train); self.stop_btn.setEnabled(False)
        reset_btn = QPushButton("↺  초기화"); reset_btn.setObjectName("reset_btn")
        reset_btn.clicked.connect(self._reset)

        self.prog2 = QProgressBar(); self.prog2.setValue(0)
        self.epoch_label = QLabel("에포크: 0 / 0"); self.epoch_label.setObjectName("info_label")

        # 결과
        res_grp = QGroupBox("결과")
        rv = QVBoxLayout(res_grp); rv.setSpacing(5)
        self.loss_label2 = QLabel("Loss: —"); self.loss_label2.setObjectName("result_label")
        self.acc_label2  = QLabel("정확도: —"); self.acc_label2.setObjectName("result_label")
        rv.addWidget(self.loss_label2); rv.addWidget(self.acc_label2)

        # 예측 테이블
        pred_grp = QGroupBox("XOR 예측 결과")
        pv = QVBoxLayout(pred_grp)
        self.pred_tbl = QTableWidget(4, 4)
        self.pred_tbl.setHorizontalHeaderLabels(["x₁","x₂","정답","예측 (확률)"])
        self.pred_tbl.verticalHeader().setVisible(False)
        self.pred_tbl.setFixedHeight(140)
        pv.addWidget(self.pred_tbl)

        lv.addWidget(param_grp)
        lv.addWidget(self.start_btn)
        lv.addWidget(self.stop_btn)
        lv.addWidget(reset_btn)
        lv.addWidget(self.epoch_label)
        lv.addWidget(self.prog2)
        lv.addWidget(res_grp)
        lv.addWidget(pred_grp)
        lv.addStretch()

        # ── 오른쪽
        right = QWidget()
        rv2 = QVBoxLayout(right); rv2.setContentsMargins(0,0,0,0)
        self.canvas = MplCanvas(1, 2, figsize=(10, 5))
        toolbar = NavigationToolbar2QT(self.canvas, self)
        rv2.addWidget(toolbar); rv2.addWidget(self.canvas)

        layout.addWidget(left)
        layout.addWidget(right, 1)
        self._reset()

    def _start_train(self):
        self.model = MLP(hidden=self.hidden_spin.value(), lr=self.lr_spin2.value())
        self.loss_history = []
        self.current_epoch = 0
        self.total_epochs = self.epoch_spin2.value()
        self.prog2.setRange(0, self.total_epochs)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)

    def _batch_train(self):
        batch = min(200, self.total_epochs - self.current_epoch)
        for _ in range(batch):
            loss = self.model.train_step(XOR_X, XOR_Y)
            self.loss_history.append(loss)
            self.current_epoch += 1

        self.prog2.setValue(self.current_epoch)
        self.epoch_label.setText(f"에포크: {self.current_epoch} / {self.total_epochs}")
        self._update_plot()

        if self.current_epoch >= self.total_epochs:
            self._stop_train()

    def _stop_train(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _update_plot(self):
        if self.model is None: return
        self.canvas.clear_axes()
        ax_loss, ax_bound = self.canvas.axes[0], self.canvas.axes[1]

        # Loss
        ax_loss.plot(self.loss_history, color=COLORS[0])
        ax_loss.set_title("학습 Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("MSE")
        ax_loss.set_yscale("log")

        # 결정 경계
        xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 200), np.linspace(-0.3, 1.3, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.forward(grid).reshape(xx.shape)
        ax_bound.contourf(xx, yy, Z, levels=50, cmap="RdYlBu_r", alpha=0.6)
        ax_bound.contour(xx, yy, Z, levels=[0.5], colors="white", linewidths=2)
        for xi, yi in zip(XOR_X, XOR_Y):
            c = COLORS[0] if yi == 1 else COLORS[3]
            ax_bound.scatter(xi[0], xi[1], color=c, s=130, zorder=5, edgecolors="white", lw=1.5)
        ax_bound.set_title("XOR 결정 경계"); ax_bound.set_xlim(-0.3, 1.3); ax_bound.set_ylim(-0.3, 1.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # 결과
        preds = self.model.forward(XOR_X)
        acc = np.mean((preds.flatten() > 0.5) == XOR_Y) * 100
        self.loss_label2.setText(f"Loss: {self.loss_history[-1]:.6f}")
        self.acc_label2.setText(f"정확도: {acc:.0f}%")

        # 예측 테이블
        for i, (xi, yi, pi) in enumerate(zip(XOR_X, XOR_Y, preds.flatten())):
            pred_cls = int(pi > 0.5)
            color = "#1e3a2f" if pred_cls == int(yi) else "#3a1e1e"
            for j, v in enumerate([xi[0], xi[1], yi, pi]):
                txt = f"{v:.3f}" if j == 3 else str(int(v))
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QColor(color))
                self.pred_tbl.setItem(i, j, item)

    def _reset(self):
        self.timer.stop()
        self.model = None
        self.loss_history = []
        self.current_epoch = 0
        self.prog2.setValue(0)
        self.epoch_label.setText("에포크: 0 / 0")
        self.loss_label2.setText("Loss: —")
        self.acc_label2.setText("정확도: —")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.canvas.clear_axes()
        ax1, ax2 = self.canvas.axes[0], self.canvas.axes[1]
        ax1.set_title("학습 Loss"); ax2.set_title("XOR 결정 경계")
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, "학습을 시작하세요", ha='center', va='center',
                    transform=ax.transAxes, color="#555577", fontsize=13)
        self.canvas.draw()
        self.pred_tbl.clearContents()


# ══════════════════════════════════════════════
#  LAB 5: UNIVERSAL APPROXIMATION
# ══════════════════════════════════════════════
TARGET_FUNCS = {
    "Sine Wave":       lambda x: np.sin(2 * np.pi * x),
    "Step Function":   lambda x: (x > 0.5).astype(float),
    "Complex Function":lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x) + 0.3*np.cos(6*np.pi*x),
}


class UniversalModel:
    """Adam optimizer를 사용한 1-hidden-layer 근사 모델"""
    def __init__(self, n, lr=0.01, epochs=3000):
        self.n = n
        self.lr = lr
        self.epochs = epochs
        self._init_weights()

    def _init_weights(self):
        np.random.seed(None)
        self.W1 = np.random.randn(1, self.n) * np.sqrt(2.0)
        self.b1 = np.zeros(self.n)
        self.W2 = np.random.randn(self.n, 1) * np.sqrt(2.0 / self.n)
        self.b2 = np.zeros(1)
        # Adam moments
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def _adam_update(self, p, g, m, v, t):
        m = self.beta1 * m + (1 - self.beta1) * g
        v = self.beta2 * v + (1 - self.beta2) * g**2
        mc = m / (1 - self.beta1**t)
        vc = v / (1 - self.beta2**t)
        return p - self.lr * mc / (np.sqrt(vc) + self.eps), m, v

    def fit(self, X, y):
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        losses = []
        for t in range(1, self.epochs + 1):
            z1  = X @ self.W1 + self.b1
            a1  = np.tanh(z1)
            out = a1 @ self.W2 + self.b2
            loss = float(np.mean((out - y)**2))
            losses.append(loss)
            dout = 2 * (out - y) / len(y)
            dW2  = a1.T @ dout;  db2 = dout.sum(0)
            da1  = dout @ self.W2.T
            dz1  = da1 * (1 - a1**2)
            dW1  = X.T @ dz1;    db1 = dz1.sum(0)
            self.W2, self.mW2, self.vW2 = self._adam_update(self.W2, dW2, self.mW2, self.vW2, t)
            self.b2, self.mb2, self.vb2 = self._adam_update(self.b2, db2, self.mb2, self.vb2, t)
            self.W1, self.mW1, self.vW1 = self._adam_update(self.W1, dW1, self.mW1, self.vW1, t)
            self.b1, self.mb1, self.vb1 = self._adam_update(self.b1, db1, self.mb1, self.vb1, t)
        return losses

    def predict(self, X):
        X = X.reshape(-1, 1)
        return (np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2).flatten()


class UniversalTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 상단 컨트롤
        ctrl = QHBoxLayout(); ctrl.setSpacing(16)

        func_grp = QGroupBox("근사 함수")
        fl = QHBoxLayout(func_grp)
        self.func_cb = QComboBox()
        self.func_cb.addItems(list(TARGET_FUNCS.keys()))
        fl.addWidget(self.func_cb)

        neuron_grp = QGroupBox("뉴런 수")
        nl = QHBoxLayout(neuron_grp)
        self.neuron_slider = QSlider(Qt.Horizontal)
        self.neuron_slider.setRange(2, 100); self.neuron_slider.setValue(10)
        self.neuron_label = QLabel("10")
        self.neuron_slider.valueChanged.connect(lambda v: self.neuron_label.setText(str(v)))
        nl.addWidget(self.neuron_slider); nl.addWidget(self.neuron_label)

        epoch_grp = QGroupBox("에포크")
        el = QHBoxLayout(epoch_grp)
        self.epoch_spin3 = QSpinBox(); self.epoch_spin3.setRange(100, 20000)
        self.epoch_spin3.setValue(3000); self.epoch_spin3.setSingleStep(500)
        el.addWidget(self.epoch_spin3)

        self.run_btn    = QPushButton("▶  근사 실행")
        self.run_btn.clicked.connect(self._run_single)
        self.cmp_btn    = QPushButton("🔀  3/10/50 비교"); self.cmp_btn.setObjectName("warn_btn")
        self.cmp_btn.clicked.connect(self._run_compare)

        self.mse_label = QLabel("MSE: —"); self.mse_label.setObjectName("result_label")

        ctrl.addWidget(func_grp, 2)
        ctrl.addWidget(neuron_grp, 3)
        ctrl.addWidget(epoch_grp, 1)
        ctrl.addWidget(self.run_btn)
        ctrl.addWidget(self.cmp_btn)
        ctrl.addWidget(self.mse_label)

        # 캔버스 (단일 모드)
        self.canvas_single = MplCanvas(1, 2, figsize=(12, 4))
        toolbar1 = NavigationToolbar2QT(self.canvas_single, self)

        # 캔버스 (비교 모드)
        self.canvas_cmp = MplCanvas(1, 3, figsize=(14, 4))
        toolbar2 = NavigationToolbar2QT(self.canvas_cmp, self)

        layout.addLayout(ctrl)
        layout.addWidget(toolbar1)
        layout.addWidget(self.canvas_single, 1)
        layout.addWidget(toolbar2)
        layout.addWidget(self.canvas_cmp, 1)

        self._init_plots()

    def _init_plots(self):
        for canvas in [self.canvas_single, self.canvas_cmp]:
            canvas.clear_axes()
            if isinstance(canvas.axes, np.ndarray):
                for ax in canvas.axes.flat:
                    ax.text(0.5, 0.5, "실행 버튼을 누르세요", ha='center', va='center',
                            transform=ax.transAxes, color="#555577", fontsize=12)
            canvas.draw()

    def _get_data(self):
        fname = self.func_cb.currentText()
        fn = TARGET_FUNCS[fname]
        X = np.linspace(0, 1, 300)
        y = fn(X)
        return X, y, fname

    def _run_single(self):
        X, y, fname = self._get_data()
        n = self.neuron_slider.value()
        epochs = self.epoch_spin3.value()

        self.run_btn.setEnabled(False)
        self.run_btn.setText("학습 중...")
        QApplication.processEvents()

        model = UniversalModel(n=n, lr=0.01, epochs=epochs)
        losses = model.fit(X, y)
        y_pred = model.predict(X)
        mse = float(np.mean((y_pred - y)**2))

        self.mse_label.setText(f"MSE: {mse:.5f}")
        self.canvas_single.clear_axes()
        ax_f, ax_l = self.canvas_single.axes[0], self.canvas_single.axes[1]

        ax_f.plot(X, y,      color=COLORS[0], label="원본 함수", lw=2.5)
        ax_f.plot(X, y_pred, color=COLORS[3], label=f"근사 (n={n})", lw=2, linestyle="--")
        ax_f.set_title(f"{fname} — 신경망 근사 (neurons={n})")
        ax_f.set_xlabel("x"); ax_f.legend()
        ax_f.fill_between(X, y, y_pred, alpha=0.15, color=COLORS[1])

        ax_l.plot(losses, color=COLORS[2])
        ax_l.set_title("학습 Loss (MSE)"); ax_l.set_xlabel("Epoch"); ax_l.set_yscale("log")

        self.canvas_single.fig.tight_layout()
        self.canvas_single.draw()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  근사 실행")

    def _run_compare(self):
        X, y, fname = self._get_data()
        epochs = self.epoch_spin3.value()

        self.cmp_btn.setEnabled(False)
        self.cmp_btn.setText("학습 중...")
        QApplication.processEvents()

        self.canvas_cmp.clear_axes()
        for idx, n in enumerate([3, 10, 50]):
            model = UniversalModel(n=n, lr=0.01, epochs=epochs)
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = float(np.mean((y_pred - y)**2))
            ax = self.canvas_cmp.axes[idx]
            ax.plot(X, y,      color=COLORS[0], label="원본", lw=2.5)
            ax.plot(X, y_pred, color=COLORS[idx+3 if idx+3 < len(COLORS) else idx], label=f"n={n}", lw=2, ls="--")
            ax.set_title(f"neurons={n}\nMSE={mse:.4f}")
            ax.legend(fontsize=9)
            ax.set_xlabel("x")
            QApplication.processEvents()

        self.canvas_cmp.fig.tight_layout()
        self.canvas_cmp.draw()
        self.cmp_btn.setEnabled(True)
        self.cmp_btn.setText("🔀  3/10/50 비교")


# ══════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3: 신경망 기초 — 인터랙티브 학습")
        self.setMinimumSize(1280, 820)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(PerceptronTab(),   "Lab 1: Perceptron")
        tabs.addTab(ActivationTab(),   "Lab 2: Activation Functions")
        tabs.addTab(ForwardPropTab(),  "Lab 3: Forward Propagation")
        tabs.addTab(MLPTab(),          "Lab 4: MLP (NumPy)")
        tabs.addTab(UniversalTab(),    "Lab 5: Universal Approximation")
        self.setCentralWidget(tabs)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

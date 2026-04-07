# TRD (Technical Requirements Document)
# Week 3: 신경망 기초 — PySide6 교육용 인터랙티브 앱

---

## 1. 기술 스택

| 항목 | 버전 | 용도 |
|------|------|------|
| Python | 3.10+ | 런타임 |
| PySide6 | 6.6+ | UI 프레임워크 |
| NumPy | 1.24+ | 수치 연산, 신경망 구현 |
| Matplotlib | 3.7+ | 그래프 시각화 |
| matplotlib-backend-qt | PySide6 연동 | FigureCanvasQTAgg |

### 의존성 설치

```bash
pip install PySide6 numpy matplotlib
```

---

## 2. 프로젝트 구조

```
week3_app/
├── main.py                  # 진입점, MainWindow 실행
├── main_window.py           # QMainWindow + QTabWidget
├── tabs/
│   ├── tab_perceptron.py        # Lab 1
│   ├── tab_activation.py        # Lab 2
│   ├── tab_forward_prop.py      # Lab 3
│   ├── tab_mlp.py               # Lab 4
│   └── tab_universal.py         # Lab 5
├── models/
│   ├── perceptron.py            # Perceptron 클래스
│   ├── activation_functions.py  # 활성화 함수 모음
│   ├── mlp_numpy.py             # MLP 클래스 (NumPy)
│   └── universal_approx.py      # 1-hidden-layer 근사 모델
└── utils/
    ├── canvas.py                # MplCanvas 래퍼
    └── workers.py               # QThread 작업자 클래스
```

---

## 3. 공통 컴포넌트

### 3.1 MplCanvas (utils/canvas.py)

```python
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(6, 4)):
        fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = fig.subplots(nrows, ncols)
        super().__init__(fig)
```

### 3.2 TrainWorker (utils/workers.py)

학습 중 UI 블로킹 방지를 위해 QThread 기반 작업자를 사용한다.

```python
from PySide6.QtCore import QThread, Signal

class TrainWorker(QThread):
    progress = Signal(int, float)   # epoch, loss
    finished = Signal(object)       # 학습 완료된 모델

    def __init__(self, model, X, y, epochs, lr):
        super().__init__()
        self.model = model
        self.X, self.y = X, y
        self.epochs, self.lr = epochs, lr

    def run(self):
        for epoch in range(self.epochs):
            loss = self.model.train_step(self.X, self.y, self.lr)
            if epoch % max(1, self.epochs // 100) == 0:
                self.progress.emit(epoch, loss)
        self.finished.emit(self.model)
```

---

## 4. Lab별 기술 명세

### Lab 1: Perceptron (tab_perceptron.py)

#### 4.1.1 모델 (models/perceptron.py)

```python
class Perceptron:
    def __init__(self, input_size: int, lr: float = 0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.lr = lr

    def activation(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def predict(self, x: np.ndarray) -> int:
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        total_error = 0
        for xi, yi in zip(X, y):
            pred = self.predict(xi)
            err = yi - pred
            self.weights += self.lr * err * xi
            self.bias    += self.lr * err
            total_error  += abs(err)
        return total_error / len(y)
```

#### 4.1.2 레이아웃

```
QHBoxLayout
├── 좌측 QFormLayout (width: 260px)
│   ├── QComboBox: 게이트 선택 (AND / OR / XOR)
│   ├── QDoubleSpinBox: 학습률 (0.001 ~ 1.0, 기본 0.1)
│   ├── QSpinBox: 에포크 수 (1 ~ 1000, 기본 100)
│   ├── QPushButton: "학습 시작"
│   └── QLabel: 가중치/편향 출력
└── 우측 MplCanvas (1행 2열)
    ├── axes[0]: Loss 곡선
    └── axes[1]: 결정 경계
```

#### 4.1.3 결정 경계 시각화

```python
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                      np.linspace(-0.5, 1.5, 200))
Z = np.array([model.predict(np.array([x, y]))
              for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
```

---

### Lab 2: Activation Functions (tab_activation.py)

#### 4.2.1 함수 구현 (models/activation_functions.py)

```python
import numpy as np

def sigmoid(x):       return 1 / (1 + np.exp(-x))
def sigmoid_d(x):     s = sigmoid(x); return s * (1 - s)

def tanh_f(x):        return np.tanh(x)
def tanh_d(x):        return 1 - np.tanh(x)**2

def relu(x):          return np.maximum(0, x)
def relu_d(x):        return (x > 0).astype(float)

def leaky_relu(x, a=0.01):   return np.where(x > 0, x, a * x)
def leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)
```

#### 4.2.2 레이아웃

```
QVBoxLayout
├── 상단 컨트롤 QHBoxLayout
│   ├── QCheckBox × 4: Sigmoid / Tanh / ReLU / Leaky ReLU
│   ├── QDoubleSpinBox: α (Leaky ReLU, 0.001~0.5)
│   ├── QSlider: x 범위 (-20 ~ 20)
│   └── QDoubleSpinBox: 포인트 x 값 (하이라이트용)
└── MplCanvas (1행 2열)
    ├── axes[0]: 함수 그래프
    └── axes[1]: 미분(Gradient) 그래프
```

#### 4.2.3 Vanishing Gradient 강조

Sigmoid/Tanh의 경우 |x| > 5 구간을 `axvspan`으로 연한 빨간색 영역으로 표시한다.

```python
ax.axvspan(-x_max, -5, alpha=0.1, color='red', label='Vanishing 구간')
ax.axvspan(5, x_max, alpha=0.1, color='red')
```

---

### Lab 3: Forward Propagation (tab_forward_prop.py)

#### 4.3.1 상태 머신

단계별 실행을 위해 `current_step` 상태 변수를 관리한다.

```
STEPS = ['init', 'z1', 'a1', 'z2', 'a2']
current_step: int  # 0 ~ 4
```

버튼 클릭마다 `current_step += 1`, 마지막 단계 이후 버튼 비활성화.

#### 4.3.2 행렬 값 테이블 표시

각 단계별 행렬을 `QTableWidget`에 표시한다.

```python
def fill_table(widget: QTableWidget, matrix: np.ndarray):
    r, c = (1, matrix.size) if matrix.ndim == 1 else matrix.shape
    widget.setRowCount(r)
    widget.setColumnCount(c)
    for i in range(r):
        for j in range(c):
            val = matrix.flat[i * c + j]
            widget.setItem(i, j, QTableWidgetItem(f"{val:.4f}"))
```

#### 4.3.3 네트워크 다이어그램

Matplotlib으로 노드-엣지를 직접 그린다. 활성화된 단계의 엣지를 강조색으로 표시한다.

```python
# 노드 위치 계산
layer_sizes = [input_size, hidden_size, output_size]
for layer_idx, size in enumerate(layer_sizes):
    for neuron_idx in range(size):
        x = layer_idx * 2
        y = neuron_idx - size / 2
        circle = plt.Circle((x, y), 0.3, color=node_color)
        ax.add_patch(circle)
```

---

### Lab 4: MLP (tab_mlp.py)

#### 4.4.1 모델 (models/mlp_numpy.py)

```python
class MLP:
    def __init__(self, hidden_size: int = 4, lr: float = 0.1):
        self.W1 = np.random.randn(2, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.5
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = (self.a1.T @ dz2) / m
        db2 = dz2.sum(axis=0) / m
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_d(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = dz1.sum(axis=0) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X, y, lr=None) -> float:
        if lr: self.lr = lr
        out = self.forward(X)
        self.backward(X, y)
        return float(np.mean((out - y.reshape(-1,1))**2))
```

#### 4.4.2 실시간 학습 업데이트

`QTimer`를 사용해 100 에포크씩 배치 학습하고 그래프를 갱신한다.

```python
self.timer = QTimer()
self.timer.setInterval(50)   # 50ms마다 갱신
self.timer.timeout.connect(self._train_batch)

def _train_batch(self):
    for _ in range(100):      # 한 번에 100 에포크
        loss = self.model.train_step(self.X, self.y)
        self.loss_history.append(loss)
        self.current_epoch += 1
    self._update_plots()
    if self.current_epoch >= self.total_epochs:
        self.timer.stop()
```

#### 4.4.3 예측 결과 테이블

```
QTableWidget (4행 3열)
열: 입력 (x1, x2) | 정답 | 예측 | 확률
```

---

### Lab 5: Universal Approximation (tab_universal.py)

#### 4.5.1 모델 (models/universal_approx.py)

1-hidden-layer 네트워크를 NumPy로 구현한다.

```python
class UniversalApprox:
    def __init__(self, n_neurons: int, lr: float = 0.01, epochs: int = 5000):
        self.n = n_neurons
        self.W1 = np.random.randn(1, n_neurons)
        self.b1 = np.random.randn(n_neurons)
        self.W2 = np.random.randn(n_neurons, 1)
        self.b2 = np.zeros(1)
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        losses = []
        for _ in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            out = a1 @ self.W2 + self.b2
            loss = np.mean((out - y)**2)
            losses.append(loss)
            # Backprop
            dout = 2 * (out - y) / len(y)
            dW2 = a1.T @ dout
            da1 = dout @ self.W2.T
            dz1 = da1 * (1 - a1**2)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * dout.sum(axis=0)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2
```

#### 4.5.2 비교 모드

[3개 비교] 버튼 클릭 시 뉴런 수 3 / 10 / 50에 대해 각각 학습 후
`MplCanvas(1, 3)`에 서브플롯으로 동시 표시한다.

```python
for idx, n in enumerate([3, 10, 50]):
    model = UniversalApprox(n_neurons=n)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = float(np.mean((y_pred - y_test)**2))
    ax = canvas.axes[idx]
    ax.plot(X_test, y_test, label='원본')
    ax.plot(X_test, y_pred, label=f'근사 (n={n})')
    ax.set_title(f'neurons={n}, MSE={mse:.4f}')
    ax.legend()
```

---

## 5. 메인 윈도우 (main_window.py)

```python
from PySide6.QtWidgets import QMainWindow, QTabWidget
from tabs.tab_perceptron   import PerceptronTab
from tabs.tab_activation   import ActivationTab
from tabs.tab_forward_prop import ForwardPropTab
from tabs.tab_mlp          import MLPTab
from tabs.tab_universal    import UniversalTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3: 신경망 기초 — 인터랙티브 학습")
        self.setMinimumSize(1280, 800)

        tabs = QTabWidget()
        tabs.addTab(PerceptronTab(),   "Lab 1: Perceptron")
        tabs.addTab(ActivationTab(),   "Lab 2: Activation Functions")
        tabs.addTab(ForwardPropTab(),  "Lab 3: Forward Propagation")
        tabs.addTab(MLPTab(),          "Lab 4: MLP")
        tabs.addTab(UniversalTab(),    "Lab 5: Universal Approximation")
        self.setCentralWidget(tabs)
```

---

## 6. 진입점 (main.py)

```python
import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

---

## 7. 오류 처리 명세

| 상황 | 처리 방법 |
|------|---------|
| 학습률 = 0 입력 | QMessageBox.warning, 기본값 0.1 복원 |
| 에포크 수 < 1 | QMessageBox.warning, 기본값 복원 |
| 은닉층 뉴런 수 < 1 | QMessageBox.warning |
| 학습 중 NaN Loss 발생 | 학습 중단, "학습률을 낮추세요" 메시지 |
| X 입력값 파싱 실패 | QMessageBox.critical, 입력 필드 초기화 |

---

## 8. 테스트 계획

| 테스트 | 내용 | 기대 결과 |
|--------|------|---------|
| Unit: Perceptron | AND 게이트, η=0.1, epoch=100 | Loss → 0, 정확도 100% |
| Unit: MLP | XOR 게이트, hidden=4, epoch=10000 | 정확도 100% |
| Unit: UniversalApprox | sin(2πx), n=50 | MSE < 0.005 |
| UI: 슬라이더 반응 | 뉴런 수 슬라이더 드래그 | 플롯 즉시 갱신 |
| UI: QThread | 학습 중 슬라이더 조작 | UI 블로킹 없음 |
| 예외: 잘못된 입력 | 학습률 = -1 입력 | 경고창 표시 |

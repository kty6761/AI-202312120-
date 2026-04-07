# TRD — Week 4 Physics Neural Networks GUI

**버전:** 1.0
**날짜:** 2026-04-06

---

## 1. 기술 스택

| 구분 | 기술 | 버전 |
|------|------|------|
| GUI | PySide6 | 6.x |
| ML | TensorFlow / Keras | 2.x |
| 수치계산 | NumPy | 1.x / 2.x |
| 시각화 | Matplotlib | 3.x |
| 런타임 | Python | 3.9+ |
| OS | Windows 10/11 | — |

---

## 2. 아키텍처

```
main.py
└── MainWindow(QMainWindow)
    └── QTabWidget
        ├── Lab1Widget(LabWidget)   ← labs/lab1_1d.py
        ├── Lab2Widget(LabWidget)   ← labs/lab2_projectile.py
        ├── Lab3Widget(LabWidget)   ← labs/lab3_overfitting.py
        └── Lab4Widget(LabWidget)   ← labs/lab4_pendulum.py

labs/base.py
└── LabWidget(QWidget)             ← 공통 레이아웃 베이스
```

---

## 3. 파일 구조

```
week4/
├── main.py                  # 앱 진입점
├── labs/
│   ├── __init__.py
│   ├── base.py              # LabWidget 베이스 클래스
│   ├── lab1_1d.py
│   ├── lab2_projectile.py
│   ├── lab3_overfitting.py
│   └── lab4_pendulum.py
├── tests/                   # pytest 단위 테스트 (12개)
└── outputs/                 # 학습 완료 시 PNG 자동 저장
```

---

## 4. LabWidget 베이스 클래스 (`labs/base.py`)

각 Lab 탭의 공통 레이아웃을 정의하는 추상 클래스.

```
QVBoxLayout
├── QSplitter(Horizontal)
│   ├── FigureCanvasQTAgg  [좌, 700px]  ← matplotlib 임베드
│   └── controls widget    [우, 300px]  ← 서브클래스가 구현
└── QTextEdit (read-only, 80px)          ← 학습 로그
```

**추상 메서드:**
- `build_controls() -> QWidget` — 파라미터 UI 구성
- `run_training()` — 학습 실행 + 그래프 업데이트

---

## 5. 각 Lab 모델 구조

### Lab 1 — 1D 함수 근사
- 입력: x (정규화: x / 3π)
- 모델: Dense(128, tanh) → Dense(128, tanh) → Dense(64, tanh) → Dense(1, linear)
- 손실함수: MSE / 옵티마이저: Adam
- 출력: `outputs/perfect_1d_approximation.png`

### Lab 2 — 포사체 운동
- 입력: [v₀, θ, t] (3차원, 표준화 정규화)
- 모델: Dense(128, relu) → Dropout(0.1) → Dense(64) → Dropout(0.1) → Dense(32) → Dropout(0.1) → Dense(2, linear)
- 출력: x, y 좌표 / `outputs/02_projectile_trajectories.png`

### Lab 3 — 오버피팅 데모
- 함수: y = sin(2x) + 0.5x + noise
- 3개 모델 동시 학습:
  - Underfit: Dense(4, relu) → Dense(1)
  - Good Fit: Dense(32) → Dropout(0.2) → Dense(16) → Dense(1)
  - Overfit: Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(1)
- 출력: `outputs/03_overfitting_comparison.png`

### Lab 4 — 진자 주기 예측
- 입력: [L, θ₀] (표준화 정규화)
- 모델: Dense(64, relu) → Dropout(0.1) → Dense(32) → Dropout(0.1) → Dense(16) → Dropout(0.1) → Dense(1, linear)
- 훈련 데이터: RK4 수치적분으로 2000개 주기 계산
- 출력: 2-subplot 그래프 / `outputs/04_pendulum_prediction.png`

---

## 6. RK4 수치적분 (`lab4_pendulum.py`)

진자 운동방정식 `d²θ/dt² = -(g/L)sin(θ)` 를 4차 Runge-Kutta로 적분.

**주기 계산:** 연속 두 zero-crossing(방향 무관)의 간격 × 2

```python
if (theta[i-1] >= 0 and theta[i] < 0) or (theta[i-1] < 0 and theta[i] >= 0):
    crossings.append(i * dt)
```

---

## 7. 학습 실행 흐름

```
버튼 클릭
  → run_training() 호출 (UI 블로킹)
  → 훈련 데이터 생성
  → keras model.fit() (verbose=0)
  → self.log() 로 결과 출력
  → figure.clear() + subplot 그리기
  → canvas.draw()
  → os.makedirs("outputs") + figure.savefig()
```

---

## 8. 테스트

```bash
cd week4
python -m pytest tests/ -v
```

| 파일 | 테스트 항목 |
|------|------------|
| test_lab1.py | 데이터 shape, 정규화 범위, layer 파싱 |
| test_lab2.py | 데이터 shape, y≥0, 정규화 |
| test_lab3.py | 데이터 shape, 노이즈 효과, 모델 설정 |
| test_lab4.py | RK4 에너지 보존, 소각도 주기, 길이-주기 관계 |

총 **12개** 테스트, 전체 통과 확인됨.

---

## 9. 실행 방법

```bash
pip install pyside6 tensorflow numpy matplotlib pytest
cd week4
python main.py
```

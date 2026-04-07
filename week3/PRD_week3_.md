# PRD (Product Requirements Document)
# Week 3: 신경망 기초 — PySide6 교육용 인터랙티브 앱

---

## 1. 개요

### 1.1 제품 목적

Week 3 강의 자료(신경망 기초)의 5개 Lab을 PySide6 기반 데스크탑 애플리케이션으로 구현한다.
각 Lab은 독립적인 탭(또는 서브윈도우)으로 구성되며, 학습자가 파라미터를 직접 조작하고 결과를 실시간으로 시각화하면서 신경망의 핵심 개념을 체험할 수 있는 인터랙티브 교육 도구를 제공한다.

### 1.2 대상 사용자

- 신경망을 처음 배우는 대학교 수강생
- Python/NumPy 기초 지식 보유
- PySide6 환경에서 실행 가능한 로컬 PC 사용자

### 1.3 범위

| 항목 | 내용 |
|------|------|
| 플랫폼 | Windows / macOS / Linux (PySide6 지원 환경) |
| 언어 | Python 3.10+ |
| UI 프레임워크 | PySide6 |
| 시각화 | Matplotlib (FigureCanvasQTAgg 임베딩) |
| 수치 연산 | NumPy |

---

## 2. 기능 요구사항

### Lab 1: Perceptron

**목표:** 단일 퍼셉트론으로 논리 게이트를 학습시키고 결정 경계를 시각화한다.

| ID | 요구사항 |
|----|---------|
| F1-1 | AND / OR / XOR 게이트 선택 콤보박스 제공 |
| F1-2 | 학습률(η), 에포크 수 입력 필드 제공 |
| F1-3 | [학습 시작] 버튼 클릭 시 퍼셉트론 학습 실행 |
| F1-4 | 에포크별 Loss 그래프 실시간 출력 |
| F1-5 | 학습 완료 후 결정 경계(Decision Boundary) 2D 플롯 표시 |
| F1-6 | XOR 선택 시 "선형 분리 불가" 경고 메시지 표시 |
| F1-7 | 최종 가중치(w₁, w₂, b) 수치 출력 |

### Lab 2: Activation Functions

**목표:** Sigmoid, Tanh, ReLU, Leaky ReLU의 함수 그래프와 미분 그래프를 비교한다.

| ID | 요구사항 |
|----|---------|
| F2-1 | 함수 선택 체크박스: Sigmoid / Tanh / ReLU / Leaky ReLU (다중 선택) |
| F2-2 | x 범위 슬라이더 (기본 -10 ~ 10) |
| F2-3 | Leaky ReLU의 α 값 입력 필드 (기본 0.01) |
| F2-4 | 함수 그래프 / Gradient 그래프를 나란히 표시 (2열 레이아웃) |
| F2-5 | 특정 x 값 입력 시 해당 지점의 함수값·미분값 하이라이트 |
| F2-6 | Vanishing Gradient 구간 시각적 강조 표시 |

### Lab 3: Forward Propagation

**목표:** 2-Layer 네트워크의 순전파를 단계별로 시각화한다.

| ID | 요구사항 |
|----|---------|
| F3-1 | 네트워크 구조 입력: 입력층 크기, 은닉층 크기, 출력층 크기 |
| F3-2 | 입력 벡터 X 직접 입력 기능 |
| F3-3 | [단계별 실행] 버튼: 클릭할 때마다 z₁ → a₁ → z₂ → a₂ 순서로 한 단계씩 시각화 |
| F3-4 | 각 단계의 행렬 값(숫자)을 테이블 위젯으로 표시 |
| F3-5 | 네트워크 구조 다이어그램(노드-엣지 그래프) 표시 |
| F3-6 | [랜덤 가중치 재생성] 버튼 제공 |
| F3-7 | 활성화 함수 선택 (ReLU / Sigmoid / Tanh) |

### Lab 4: MLP (Multi-Layer Perceptron)

**목표:** Pure NumPy MLP로 XOR 문제를 학습시키고 학습 과정을 시각화한다.

| ID | 요구사항 |
|----|---------|
| F4-1 | 은닉층 뉴런 수 입력 (기본 4) |
| F4-2 | 학습률, 에포크 수 입력 |
| F4-3 | [학습 시작] / [학습 초기화] 버튼 |
| F4-4 | 학습 중 Loss 곡선 실시간 업데이트 (QTimer 활용) |
| F4-5 | 학습 완료 후 결정 경계 2D 플롯 표시 |
| F4-6 | 은닉층 뉴런별 활성화 값 바 차트 표시 |
| F4-7 | XOR 4개 입력에 대한 예측 결과 테이블 표시 |
| F4-8 | 최종 정확도(Accuracy) 수치 표시 |

### Lab 5: Universal Approximation

**목표:** 뉴런 수에 따른 함수 근사 품질을 비교한다.

| ID | 요구사항 |
|----|---------|
| F5-1 | 근사 대상 함수 선택: Sine Wave / Step Function / Complex Function |
| F5-2 | 뉴런 수 슬라이더 (3 ~ 100) |
| F5-3 | [근사 실행] 버튼 클릭 시 1-hidden-layer 네트워크 학습 및 근사 곡선 표시 |
| F5-4 | 원본 함수 vs 근사 함수 오버레이 플롯 |
| F5-5 | MSE(Mean Squared Error) 수치 실시간 표시 |
| F5-6 | 뉴런 수 3 / 10 / 50 고정 비교 버튼 제공 (3개 서브플롯 동시 표시) |

---

## 3. 비기능 요구사항

| ID | 항목 | 요구사항 |
|----|------|---------|
| NF-1 | 응답성 | 학습 실행 중 UI가 멈추지 않도록 QThread 또는 QTimer 활용 |
| NF-2 | 레이아웃 | 최소 창 크기 1280×800, 해상도 독립적 레이아웃 |
| NF-3 | 탭 구조 | 메인 윈도우에 5개 탭으로 각 Lab 분리 |
| NF-4 | 초기화 | 각 탭에 [초기화] 버튼 제공, 파라미터를 기본값으로 복원 |
| NF-5 | 오류 처리 | 잘못된 입력값에 대해 QMessageBox 경고 표시 |
| NF-6 | 그래프 저장 | Matplotlib 툴바 통해 PNG 저장 기능 제공 |

---

## 4. UI 구조

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Tab 1: Perceptron
    │   ├── 좌측 패널: 파라미터 입력 (QFormLayout)
    │   └── 우측 패널: Matplotlib Canvas (Loss + Decision Boundary)
    ├── Tab 2: Activation Functions
    │   ├── 상단: 체크박스 + 슬라이더
    │   └── 하단: Matplotlib Canvas (함수 + Gradient)
    ├── Tab 3: Forward Propagation
    │   ├── 좌측: 구조 입력 + 단계 버튼 + 테이블
    │   └── 우측: 네트워크 다이어그램 + 값 표시
    ├── Tab 4: MLP
    │   ├── 좌측: 파라미터 입력 + 제어 버튼
    │   └── 우측: Loss 곡선 + 결정 경계 + 예측 테이블
    └── Tab 5: Universal Approximation
        ├── 상단: 함수 선택 + 뉴런 슬라이더
        └── 하단: 근사 비교 플롯 + MSE 표시
```

---

## 5. 제약 조건

- 외부 딥러닝 프레임워크(TensorFlow, PyTorch) 미사용 — NumPy만으로 구현
- 인터넷 연결 불필요 (완전 로컬 실행)
- 단일 `.py` 파일 또는 모듈 분리 구조 모두 허용

---

## 6. 인수 기준

| Lab | 인수 조건 |
|-----|---------|
| Lab 1 | AND/OR 100% 학습 수렴, XOR 경고 메시지 표시 |
| Lab 2 | 4가지 함수 동시 표시, 미분 그래프 정확 |
| Lab 3 | 단계별 버튼으로 순전파 4단계 진행 |
| Lab 4 | XOR 정확도 100% 달성 후 결정 경계 표시 |
| Lab 5 | 뉴런 수 변경 시 MSE 변화 확인 가능 |

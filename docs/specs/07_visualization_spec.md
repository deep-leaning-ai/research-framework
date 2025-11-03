# Visualization 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [ExperimentVisualizer 클래스](#2-experimentvisualizer-클래스)
3. [8-패널 차트 레이아웃](#3-8-패널-차트-레이아웃)
4. [각 패널 상세](#4-각-패널-상세)
5. [스타일링 시스템](#5-스타일링-시스템)
6. [API 명세](#6-api-명세)
7. [테스트 요구사항](#7-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Visualization 시스템은 머신러닝 실험 결과를 종합적으로 시각화하는 8-패널 차트를 생성합니다. 학습 진행, 성능 비교, 효율성 분석을 한눈에 파악할 수 있도록 설계되었습니다.

### 1.2 설계 원칙

- **정보 밀도**: 한 화면에 최대한 많은 정보 표시
- **시각적 일관성**: 통일된 색상과 스타일
- **접근성**: 색맹 사용자 고려
- **하드코딩 지양**: 모든 설정은 파라미터화

### 1.3 파일 구조

```
research/visualization/
├── __init__.py          # Visualization exports
├── visualizer.py        # ExperimentVisualizer 클래스 (구현 완료)
└── utils.py             # 시각화 유틸리티 (선택)
```

---

## 2. ExperimentVisualizer 클래스

### 2.1 개요

정적 메서드로 구성된 시각화 클래스입니다. 실험 결과를 8개 패널로 나누어 종합적으로 표시합니다.

### 2.2 클래스 구조

```python
class ExperimentVisualizer:
    """실험 결과 시각화"""

    # 클래스 상수
    DEFAULT_FIGSIZE = (24, 12)
    DEFAULT_DPI = 300
    GRID_ROWS = 2
    GRID_COLS = 4
    PANEL_COUNT = 8

    # 스타일 상수
    COLOR_MAP = 'tab10'  # matplotlib colormap
    MARKER_STYLES = ['o', 's', '^', 'D', 'v']
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    GRID_ALPHA = 0.3
    FONT_SIZE_TITLE = 14
    FONT_SIZE_LABEL = 12
    FONT_SIZE_TICK = 10

    @staticmethod
    def plot_comparison(results: List[ExperimentResult],
                       save_path: str = None,
                       figsize: tuple = DEFAULT_FIGSIZE,
                       dpi: int = DEFAULT_DPI):
        """8-패널 종합 비교 차트 생성"""
```

---

## 3. 8-패널 차트 레이아웃

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Comparison                      │
├────────────────┬────────────────┬────────────────┬──────────┤
│  [0,0]         │  [0,1]         │  [0,2]         │  [0,3]   │
│  Training      │  Test Loss     │  Metric        │  Best    │
│  Progress      │                │  Comparison    │  Perform │
├────────────────┼────────────────┼────────────────┼──────────┤
│  [1,0]         │  [1,1]         │  [1,2]         │  [1,3]   │
│  Efficiency    │  Epoch Time    │  Inference     │  Overfit │
│  Scatter       │                │  Time          │  Gap     │
└────────────────┴────────────────┴────────────────┴──────────┘
```

### 3.2 좌표 시스템

```python
# matplotlib subplot 인덱싱
axes[0, 0]  # 첫 번째 행, 첫 번째 열
axes[0, 1]  # 첫 번째 행, 두 번째 열
axes[1, 3]  # 두 번째 행, 네 번째 열
```

### 3.3 레이아웃 설정

```python
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(24, 12),
    tight_layout=True
)
fig.suptitle('Experiment Comparison', fontsize=16)
```

---

## 4. 각 패널 상세

### 4.1 Panel [0,0]: Training Progress (Overfitting Check)

**목적**: 과적합 모니터링

**구성 요소**:
- X축: Epochs
- Y축: Loss
- 라인: Train loss (dashed, alpha=0.6), Val loss (solid + markers, alpha=0.9)

**코드 구조**:
```python
ax = axes[0, 0]
for i, result in enumerate(results):
    color = plt.cm.tab10(i)
    # Train loss
    ax.plot(epochs, result.train_loss,
            linestyle='--', alpha=0.6, color=color,
            label=f'{result.model_name} (train)')
    # Val loss
    ax.plot(epochs, result.val_loss,
            linestyle='-', marker=markers[i % 5], alpha=0.9,
            color=color, label=f'{result.model_name} (val)')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Progress (Overfitting Check)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
```

### 4.2 Panel [0,1]: Test Loss

**목적**: 최종 테스트 성능 추이

**구성 요소**:
- X축: Epochs
- Y축: Test Loss
- 라인: 각 모델별 test loss (solid + markers)

**특징**: 테스트 손실의 에폭별 변화 관찰

### 4.3 Panel [0,2]: Primary Metric Comparison

**목적**: 주요 메트릭의 train/val/test 비교

**구성 요소**:
- X축: Epochs
- Y축: Primary metric value
- 라인 스타일:
  - Train: dashed, alpha=0.5
  - Val: dash-dot, alpha=0.7, markers every 5 epochs
  - Test: solid, alpha=0.9

**코드 구조**:
```python
# Train metric
ax.plot(epochs, train_metrics,
        linestyle='--', alpha=0.5, color=color)
# Val metric
ax.plot(epochs[::5], val_metrics[::5],  # 5 에폭마다 마커
        linestyle='-.', marker=markers[i % 5],
        alpha=0.7, color=color)
# Test metric
ax.plot(epochs, test_metrics,
        linestyle='-', alpha=0.9, color=color)
```

### 4.4 Panel [0,3]: Best Performance Bar Chart

**목적**: 모델별 최고 성능 직접 비교

**구성 요소**:
- X축: Model names
- Y축: Best primary metric value
- 요소: Bar chart with value labels

**코드 구조**:
```python
models = [r.model_name for r in results]
scores = [r.best_test_metric for r in results]
bars = ax.bar(models, scores, color=colors)

# 값 레이블 추가
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom')
```

### 4.5 Panel [1,0]: Parameter Efficiency Scatter

**목적**: 파라미터 대비 성능 시각화

**구성 요소**:
- X축: Parameters (log scale)
- Y축: Best metric value
- 요소: Scatter plot with large markers

**코드 구조**:
```python
params = [r.parameters for r in results]
metrics = [r.best_test_metric for r in results]

ax.scatter(params, metrics, s=200, c=colors, alpha=0.7)
ax.set_xscale('log')  # 로그 스케일

# 모델명 레이블
for i, (x, y, name) in enumerate(zip(params, metrics, models)):
    ax.annotate(name, (x, y), xytext=(5, 5),
                textcoords='offset points')
```

### 4.6 Panel [1,1]: Average Epoch Time

**목적**: 학습 속도 비교

**구성 요소**:
- X축: Model names
- Y축: Average time per epoch (seconds)
- 요소: Bar chart

**계산**:
```python
avg_times = [np.mean(r.epoch_times) for r in results]
```

### 4.7 Panel [1,2]: Average Inference Time

**목적**: 추론 속도 비교

**구성 요소**:
- X축: Model names
- Y축: Inference time (milliseconds)
- 요소: Bar chart with time labels

**특징**: 실시간 처리 가능성 평가

### 4.8 Panel [1,3]: Overfitting Gap

**목적**: 과적합 정도 시각화

**구성 요소**:
- X축: Model names
- Y축: Gap percentage (train - val)
- 요소: Bar chart with horizontal line at y=0

**계산**:
```python
overfitting_gaps = [r.final_overfitting_gap for r in results]
colors = ['red' if gap > 0.05 else 'green' for gap in gaps]
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
```

---

## 5. 스타일링 시스템

### 5.1 색상 체계

```python
# 기본 컬러맵
plt.cm.tab10  # 10개 구별 가능한 색상

# 색상 할당
colors = [plt.cm.tab10(i) for i in range(len(results))]
```

**문제점**:
- 10개 이상 모델시 색상 중복
- 색맹 친화성 미고려

### 5.2 마커 시스템

```python
MARKER_STYLES = ['o', 's', '^', 'D', 'v']  # 5개 스타일
marker = MARKER_STYLES[i % len(MARKER_STYLES)]
```

### 5.3 투명도 계층

| 요소 | Alpha 값 | 용도 |
|------|----------|------|
| Train lines | 0.5-0.6 | 배경 |
| Val lines | 0.7 | 중간 |
| Test lines | 0.9 | 전경 |
| Grid | 0.3 | 보조선 |
| Scatter | 0.7 | 점 |

### 5.4 폰트 크기

```python
# 계층별 폰트 크기
FONT_SIZES = {
    'suptitle': 16,
    'title': 14,
    'label': 12,
    'tick': 10,
    'legend': 10
}
```

---

## 6. API 명세

### 6.1 기본 사용법

```python
from research.visualization import ExperimentVisualizer
from research.experiment import ExperimentResult

# 실험 결과 리스트
results = [result1, result2, result3]

# 8-패널 차트 생성
ExperimentVisualizer.plot_comparison(
    results=results,
    save_path='comparison.png',
    figsize=(24, 12),
    dpi=300
)
```

### 6.2 파라미터 설명

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| results | List[ExperimentResult] | 필수 | 비교할 실험 결과들 |
| save_path | str | None | 저장 경로 (None시 화면 출력) |
| figsize | tuple | (24, 12) | 전체 figure 크기 |
| dpi | int | 300 | 해상도 (dots per inch) |

### 6.3 보조 메서드

```python
@staticmethod
def plot_metric_comparison(results: List[ExperimentResult],
                          metric_name: str = 'accuracy',
                          save_path: str = None):
    """특정 메트릭 비교 차트 (1x2 레이아웃)

    왼쪽: 메트릭 추이 (train/val/test)
    오른쪽: 최종 비교 막대 차트
    """
```

### 6.4 커스터마이징

```python
# 커스텀 색상 사용
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for i, (result, color) in enumerate(zip(results, custom_colors)):
    # 커스텀 색상으로 플롯
    axes[0, 0].plot(result.train_loss, color=color)
```

---

## 7. 테스트 요구사항

### 7.1 단위 테스트

```python
def test_plot_comparison():
    """plot_comparison 메서드 테스트"""

    # 1. 더미 결과 생성
    results = create_dummy_results(3)

    # 2. 차트 생성
    temp_path = 'test_comparison.png'
    ExperimentVisualizer.plot_comparison(
        results=results,
        save_path=temp_path
    )

    # 3. 파일 생성 확인
    assert os.path.exists(temp_path)
    assert os.path.getsize(temp_path) > 0

    # 4. 이미지 검증
    img = plt.imread(temp_path)
    assert img.shape[0] > 0  # height
    assert img.shape[1] > 0  # width
```

### 7.2 레이아웃 테스트

```python
def test_panel_layout():
    """8-패널 레이아웃 테스트"""

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # 패널 수 확인
    assert axes.shape == (2, 4)

    # 각 패널 접근 가능
    for i in range(2):
        for j in range(4):
            assert axes[i, j] is not None
```

### 7.3 Edge Case 테스트

```python
def test_edge_cases():
    """Edge case 처리"""

    # 1. 단일 모델
    single_result = [create_dummy_result()]
    ExperimentVisualizer.plot_comparison(single_result)

    # 2. 많은 모델 (>10)
    many_results = create_dummy_results(15)
    ExperimentVisualizer.plot_comparison(many_results)
    # 색상 순환 확인

    # 3. 긴 모델명
    long_name_results = create_results_with_long_names()
    # 레이블 겹침 처리 확인
```

### 7.4 스타일링 테스트

```python
def test_styling():
    """스타일 적용 테스트"""

    results = create_dummy_results(5)

    # 기본 스타일
    ExperimentVisualizer.plot_comparison(results)

    # 색상 확인
    colors = [plt.cm.tab10(i) for i in range(5)]
    assert len(colors) == 5

    # 마커 확인
    markers = ['o', 's', '^', 'D', 'v']
    for i, marker in enumerate(markers):
        assert marker in ExperimentVisualizer.MARKER_STYLES
```

---

## 부록: 현재 구현 문제점

### 1. 레이아웃 문제

1. **고정 그리드**
   - 2x4 고정, 적응형 아님
   - 패널 수 변경 불가

2. **레이블 겹침**
   - 긴 모델명시 X축 레이블 겹침
   - rotation 미적용

### 2. 색상 문제

1. **제한된 색상**
   - tab10은 10개만 지원
   - 이상시 순환으로 구별 어려움

2. **접근성**
   - 색맹 친화적 팔레트 미지원
   - 패턴 구별 없음

### 3. 스케일링 문제

1. **Y축 동기화 부재**
   - 패널 간 Y축 범위 불일치
   - 비교 어려움

2. **정규화 부재**
   - 다른 스케일 메트릭 비교 곤란

### 4. 인터랙션 부재

1. **정적 차트**
   - 줌/팬 불가
   - 툴팁 없음

2. **선택적 표시 불가**
   - 특정 모델 강조 불가
   - 범례 클릭 비활성화

### 개선 제안

```python
class ImprovedVisualizer:
    @staticmethod
    def plot_interactive_comparison(results, **kwargs):
        """plotly를 사용한 인터랙티브 차트"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 인터랙티브 8-패널 생성
        fig = make_subplots(rows=2, cols=4, ...)
        # 줌, 팬, 툴팁 지원
```

---

## SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ✅ | 시각화만 담당 |
| **O**pen/Closed | ⚠️ | 새 패널 추가시 수정 필요 |
| **L**iskov Substitution | N/A | 상속 미사용 |
| **I**nterface Segregation | ✅ | 최소 인터페이스 |
| **D**ependency Inversion | ✅ | ExperimentResult 추상화에 의존 |

시각화 시스템은 OCP 개선이 필요합니다 (패널 플러그인 시스템 등).
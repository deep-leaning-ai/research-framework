# ML Framework Visualization 기능 목록

##  폴더 구조
- **위치**: `ml_framework/visualization/`
- **파일**:
  - `__init__.py` - 패키지 초기화
  - `visualizer.py` - 메인 시각화 클래스 (430줄)

---

## 🎨 ExperimentVisualizer 클래스

실험 결과를 종합적으로 시각화하는 정적 클래스입니다. `ExperimentRecorder`와 `ExperimentResult` 객체를 사용합니다.

---

##  주요 메서드

### 1. `plot_comparison()` - 종합 비교 대시보드

**메서드 시그니처:**
```python
ExperimentVisualizer.plot_comparison(
    recorder: ExperimentRecorder,
    save_path: str = "experiment_comparison.png"
)
```

**기능**: 모든 모델을 여러 차원에서 비교하는 2x4 그리드 시각화 생성

**입력 데이터:**
- `recorder`: 모든 실험 결과를 포함하는 ExperimentRecorder 객체
- `save_path`: PNG 저장 경로 (기본값: "experiment_comparison.png")

**출력:**
- 24x12인치 고해상도 이미지 (300 DPI)
- PNG 형식으로 저장
- 대화형 창에 표시

**8개 서브플롯 구성:**

| # | 차트 이름 | 타입 | 목적 | 사용 데이터 |
|---|---|---|---|---|
| 1 | Training Progress (과적합 체크) | 선 그래프 | 훈련/검증 손실 분기 모니터링 | train_loss, val_loss |
| 2 | Final Test Performance | 선 그래프 | 테스트 손실 변화 추이 | test_loss |
| 3 | Primary Metric Comparison | 다중 선 그래프 | 훈련/검증/테스트 주요 메트릭 비교 | train_metrics, val_metrics, test_metrics |
| 4 | Best Metric Comparison | 막대 그래프 | 모델별 최고 성능 비교 | best_test_metric |
| 5 | Parameter Efficiency | 산점도 | 성능 대비 모델 크기 분석 | parameters, best_test_metric |
| 6 | Average Training Time | 막대 그래프 | 에폭당 평균 훈련 시간 비교 | epoch_times |
| 7 | Average Inference Time | 막대 그래프 | 추론 속도 비교 | inference_time |
| 8 | Overfitting Gap (Train-Val) | 막대 그래프 | 최종 훈련-검증 차이 표시 | final_overfitting_gap |

**스타일 특징:**
- **색상**: matplotlib tab10 컬러맵 (최대 10개 색상)
- **마커**: 5가지 스타일 (o, s, ^, D, v)
- **선 스타일**: 훈련(점선), 검증(실선), 테스트(일점쇄선)
- **그리드**: 0.3 투명도
- **값 레이블**: 막대 위 숫자 표시 (소수점 2자리)

---

### 2. `plot_metric_comparison()` - 특정 메트릭 분석

**메서드 시그니처:**
```python
ExperimentVisualizer.plot_metric_comparison(
    recorder: ExperimentRecorder,
    metric_name: str,
    save_path: str = "metric_comparison.png"
)
```

**기능**: 특정 메트릭에 대한 1x2 집중 분석 시각화 생성

**입력 데이터:**
- `recorder`: ExperimentRecorder 객체
- `metric_name`: 시각화할 메트릭 이름 (예: "Accuracy", "F1-Score", "Precision")
- `save_path`: PNG 저장 경로 (기본값: "metric_comparison.png")

**출력:**
- 14x5인치 이미지 (300 DPI)
- PNG 형식으로 저장
- 대화형 표시

**2개 서브플롯 구성:**

| # | 차트 이름 | 타입 | 목적 | 사용 데이터 |
|---|---|---|---|---|
| 1 | Metric over Epochs | 다중 선 그래프 | 훈련/검증/테스트 메트릭 진행 상황 | train_metrics[metric_name], val_metrics[metric_name], test_metrics[metric_name] |
| 2 | Best Metric Comparison | 막대 그래프 | 모델별 최고 메트릭 값 비교 | max(test_metrics[metric_name]) |

**스타일 특징:**
- **색상**: tab10 컬러맵
- **마커**: 4가지 스타일 (o, s, ^, D)
- **선 스타일**: 훈련(점선), 검증(일점쇄선), 테스트(실선)
- **그리드**: 0.3 투명도
- **값 레이블**: 소수점 2자리

---

## 🎯 주요 특징

### 설계 원칙
1. **Task-Agnostic**: 모든 메트릭 이름 지원 (Accuracy, F1-Score, Loss 등)
2. **다중 모델 지원**: 2-10개 모델 동시 비교
3. **종합 모니터링**: 훈련 진행, 검증, 테스트, 효율성 메트릭 표시
4. **시각적 계층**: 검증/테스트 결과 강조

### 과적합 감지
- **Train vs Val Loss**: subplot 1에서 분기 시각화
- **Overfitting Gap**: subplot 8에서 최종 훈련-검증 차이 명시
- **시각 표시**: 점선 vs 실선으로 구분

### 성능 인사이트
- **절대 메트릭**: 최고 성능 막대 그래프
- **상대 효율성**: 파라미터 대비 성능 산점도
- **속도 분석**: 훈련 및 추론 시간 비교
- **추세 분석**: 시계열 플롯으로 수렴 패턴 표시

---

##  출력 형식

### 이미지 사양
- **형식**: PNG (래스터)
- **해상도**: 300 DPI (출판 품질)
- **레이아웃**: Tight layout
- **파일 크기**: 일반적으로 100-300 KB

### 대화형 디스플레이
- 저장 후 `plt.show()` 호출
- 창이 닫힐 때까지 스크립트 대기
- 줌, 팬, 저장 기능 지원

### 콘솔 출력
- 저장 확인 메시지: `"시각화 결과가 '{save_path}'로 저장되었습니다."`

---

##  의존성

### 외부 라이브러리
- `matplotlib.pyplot` - 핵심 플로팅 라이브러리
- `numpy` - 수치 연산 (평균 계산)
- `typing` - 타입 힌트

### 내부 의존성
- `ml_framework.experiment.recorder.ExperimentRecorder`
- `ExperimentResult` 데이터클래스

---

##  사용 예시

```python
from ml_framework.experiment import ExperimentRunner
from ml_framework.visualization import ExperimentVisualizer

# 실험 실행
runner = ExperimentRunner(...)
runner.run_multiple_experiments(models, train_loader, val_loader, test_loader)

# 1. 종합 비교 시각화
ExperimentVisualizer.plot_comparison(
    runner.get_recorder(),
    save_path="results/comparison.png"
)

# 2. 특정 메트릭 분석
ExperimentVisualizer.plot_metric_comparison(
    runner.get_recorder(),
    metric_name="Accuracy",
    save_path="results/accuracy.png"
)

ExperimentVisualizer.plot_metric_comparison(
    runner.get_recorder(),
    metric_name="F1-Score (macro)",
    save_path="results/f1_score.png"
)
```

---

##  기능 비교표

| 특징 | plot_comparison | plot_metric_comparison |
|------|----------------|----------------------|
| **차트 수** | 8개 | 2개 |
| **이미지 크기** | 24x12 인치 | 14x5 인치 |
| **DPI** | 300 | 300 |
| **표시 모델** | 모든 모델 | 모든 모델 |
| **표시 메트릭** | 주요 메트릭 + 손실 | 지정된 1개 메트릭 |
| **분석 차원** | 8개 (손실, 메트릭, 파라미터, 시간, 과적합) | 1개 (단일 메트릭) |
| **막대 그래프** | 4개 | 1개 |
| **선 그래프** | 3개 | 1개 |
| **산점도** | 1개 | 0개 |
| **기본 출력** | experiment_comparison.png | metric_comparison.png |

---

## 🎨 차트 스타일 상세

### 색상 시스템
- **팔레트**: matplotlib `tab10` (10가지 구분 색상)
- **할당**: 결과 딕셔너리 순서대로 순차 할당
- **투명도**: 레이어 가시성을 위한 0.5-0.9 알파값

### 마커 전략
- **마커**: [o, s, ^, D, v] 순환 (원, 사각형, 삼각형, 다이아몬드, 역삼각형)
- **마커 간격**: 에폭 수에 따라 적응형 (5-10 에폭마다)
- **크기**: 선 그래프 3-4pt, 산점도 200pt

### 선 스타일
- **훈련**: 점선 (`--`) - 보조 정보
- **검증**: 일점쇄선 (`-.`) 또는 실선 (`-`) - 주요 초점
- **테스트**: 실선 (`-`) - 최종 성능
- **선 두께**: 1.5-2pt

### 그리드 및 축
- **그리드**: `alpha=0.3`로 활성화 (비침습적 참조선)
- **그리드 타입**: 선 그래프는 직교 그리드, 막대 그래프는 y축만
- **축 레이블**: 11pt 폰트
- **제목**: 12pt 굵은 폰트
- **틱**: 모델 이름은 15도 회전 (오른쪽 정렬)

### 특수 기능
- **로그 스케일**: Parameter Efficiency 산점도는 로그 x축 사용
- **수평 기준선**: Overfitting Gap 플롯에 y=0 기준선 포함
- **값 주석**: 막대 그래프 위 숫자 표시
- **조건부 정렬**: 값 부호에 따라 레이블 수직 정렬 조정 (과적합 갭)

---

##  데이터 구조

### ExperimentResult (입력 데이터)
```python
@dataclass
class ExperimentResult:
    model_name: str                          # 모델 식별자
    task_type: str                           # 태스크 타입 (분류, 회귀 등)
    parameters: int                          # 전체 모델 파라미터 수
    train_metrics: Dict[str, List[float]]    # 훈련 중 메트릭 히스토리
    val_metrics: Dict[str, List[float]]      # 검증 중 메트릭 히스토리
    test_metrics: Dict[str, List[float]]     # 테스트 중 메트릭 히스토리
    train_loss: List[float]                  # 에폭당 훈련 손실
    val_loss: List[float]                    # 에폭당 검증 손실
    test_loss: List[float]                   # 에폭당 테스트 손실
    epoch_times: List[float]                 # 에폭당 훈련 시간 (초)
    inference_time: float                    # 평균 추론 시간 (초)
    primary_metric_name: str                 # 주요 메트릭 이름 (예: "Accuracy")
    best_test_metric: float                  # 테스트 세트 최고 메트릭 값
    final_overfitting_gap: Optional[float]   # 최종 훈련-검증 차이
    additional_info: Optional[Dict[str, Any]]# 추가 메타데이터
```

### ExperimentRecorder (데이터 관리)
```python
class ExperimentRecorder:
    results: Dict[str, ExperimentResult]  # model_name -> result 매핑

    # 주요 메서드:
    get_all_results() -> Dict[str, ExperimentResult]  # 모든 결과 반환
    add_result(result: ExperimentResult) -> None
    get_result(model_name: str) -> Optional[ExperimentResult]
```

---

##  완전한 사용 플로우

```python
import torch
from torchvision import datasets, transforms

from ml_framework.models import CNN, FullyConnectedNN
from ml_framework.strategies import MultiClassStrategy
from ml_framework.metrics import AccuracyMetric, F1ScoreMetric
from ml_framework.loaders import DataLoaderFactory
from ml_framework.experiment import ExperimentRunner
from ml_framework.visualization import ExperimentVisualizer

# 1. 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_ratio=0.8,
    batch_size=64
)

# 2. 전략 및 메트릭 정의
strategy = MultiClassStrategy(num_classes=10)
metrics = [AccuracyMetric(), F1ScoreMetric(average='macro')]

# 3. 실험 러너 초기화
runner = ExperimentRunner(
    device='cuda',
    task_strategy=strategy,
    metrics=metrics,
    primary_metric=AccuracyMetric(),
    num_epochs=20
)

# 4. 모델 실험 실행
models = [
    CNN(output_dim=10, task_strategy=strategy),
    FullyConnectedNN(hidden_size=128, output_dim=10, task_strategy=strategy),
    FullyConnectedNN(hidden_size=256, output_dim=10, task_strategy=strategy)
]

runner.run_multiple_experiments(models, train_loader, val_loader, test_loader)

# 5. 시각화 생성
# 종합 비교 (8개 차트)
ExperimentVisualizer.plot_comparison(
    runner.get_recorder(),
    save_path='comprehensive_comparison.png'
)

# 특정 메트릭 분석 (2개 차트)
ExperimentVisualizer.plot_metric_comparison(
    runner.get_recorder(),
    metric_name='Accuracy',
    save_path='accuracy_analysis.png'
)

ExperimentVisualizer.plot_metric_comparison(
    runner.get_recorder(),
    metric_name='F1-Score (macro)',
    save_path='f1_analysis.png'
)
```

---

## WARNING: 주의사항

1. **메트릭 이름 일치**: `plot_metric_comparison()`에 전달하는 `metric_name`은 실험 결과에 존재하는 메트릭 이름과 정확히 일치해야 합니다.

2. **최소 모델 수**: 의미 있는 비교를 위해 최소 2개 이상의 모델 결과가 필요합니다.

3. **메모리 사용**: 큰 이미지 크기(24x12, 14x5인치)와 높은 DPI(300)로 인해 메모리 사용량이 높을 수 있습니다.

4. **대화형 표시**: `plt.show()`는 스크립트를 차단하므로, 서버 환경에서는 창을 표시하지 않고 저장만 하도록 수정 필요합니다.

5. **파일 덮어쓰기**: 동일한 `save_path`를 사용하면 기존 파일을 덮어씁니다.

---

## [START] 확장 가능성

현재 시각화 모듈은 정적 메서드로 구현되어 있어 다음과 같은 확장이 가능합니다:

- **추가 차트 타입**: 혼동 행렬(Confusion Matrix), ROC 곡선, PR 곡선 등
- **대화형 시각화**: Plotly, Bokeh 등을 사용한 동적 차트
- **커스텀 스타일**: 색상 테마, 폰트, 레이아웃 커스터마이징
- **애니메이션**: 훈련 과정 애니메이션 생성
- **HTML 리포트**: 웹 기반 대화형 리포트 생성

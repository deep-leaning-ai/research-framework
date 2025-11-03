# KTB ML Framework 기능명세서

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [핵심 기능 명세](#3-핵심-기능-명세)
4. [모듈별 상세 기능](#4-모듈별-상세-기능)
5. [API 명세](#5-api-명세)
6. [설정 및 확장](#6-설정-및-확장)
7. [사용 예제](#7-사용-예제)

---

## 1. 프로젝트 개요

### 1.1 기본 정보

| 항목 | 내용 |
|------|------|
| **프로젝트명** | KTB ML Framework |
| **패키지명** | research |
| **버전** | 0.1.0 |
| **Python 버전** | 3.8+ |
| **주요 프레임워크** | PyTorch 2.0+ |
| **라이선스** | MIT |
| **코드 규모** | 50개 파일, 약 5,876 라인 |

### 1.2 프로젝트 목적

KTB ML Framework는 전이학습(Transfer Learning)과 일반 머신러닝 태스크를 통합한 실험 프레임워크입니다. 연구자와 개발자가 효율적으로 딥러닝 모델을 실험하고 비교 평가할 수 있도록 설계되었습니다.

### 1.3 핵심 가치

- **통합성**: 전이학습과 일반 ML 태스크를 하나의 프레임워크로 통합
- **확장성**: SOLID 원칙과 디자인 패턴을 적용한 확장 가능한 구조
- **사용성**: 직관적인 API와 자동화된 실험 관리
- **호환성**: 레거시 API와 100% 하위 호환성 유지

---

## 2. 시스템 아키텍처

### 2.1 설계 원칙

#### SOLID 원칙 적용
- **단일 책임 원칙(SRP)**: 각 클래스는 하나의 명확한 책임만 가짐
- **개방-폐쇄 원칙(OCP)**: 기존 코드 수정 없이 확장 가능
- **리스코프 치환 원칙(LSP)**: 모든 전략 클래스는 상호 교체 가능
- **인터페이스 분리 원칙(ISP)**: 최소한의 필수 인터페이스만 정의
- **의존성 역전 원칙(DIP)**: 구체적 구현이 아닌 추상화에 의존

### 2.2 디자인 패턴

```
┌─────────────────────────────────────────────────────────┐
│                    Design Patterns                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Strategy Pattern                                    │
│     ├─ TaskStrategy (MultiClass, Binary, Regression)   │
│     ├─ TrainingStrategy (Vanilla)                       │
│     └─ LoggingStrategy (Simple, WandB)                  │
│                                                          │
│  2. Factory + Registry Pattern                          │
│     └─ ModelRegistry (@register decorator)              │
│                                                          │
│  3. Template Method Pattern                             │
│     └─ BaseModel (abstract methods)                     │
│                                                          │
│  4. Facade Pattern                                      │
│     └─ Experiment (simplified interface)                │
│                                                          │
│  5. Observer Pattern                                    │
│     └─ ExperimentRecorder (result collection)           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.3 모듈 구조

```
research/
├── core/              # 핵심 추상 클래스
│   ├── base_model.py  # 전이학습 베이스 모델
│   ├── experiment.py  # 실험 관리 파사드
│   └── strategies.py  # 전략 패턴 베이스
│
├── models/
│   ├── pretrained/    # 사전학습 모델 (13종)
│   │   ├── registry.py   # 모델 팩토리
│   │   ├── resnet.py     # ResNet (5종)
│   │   └── vgg.py        # VGG (8종)
│   └── simple/        # 간단한 모델
│       ├── cnn.py        # CNN (MNIST용)
│       └── fc.py         # Fully Connected
│
├── strategies/        # 전략 패턴 구현체
│   ├── training/      # 학습 전략
│   ├── logging/       # 로깅 전략
│   ├── task/          # 태스크 전략
│   └── optimization/  # 최적화 전략 (예정)
│
├── metrics/           # 메트릭 시스템
│   ├── base.py        # 메트릭 인터페이스
│   ├── classification.py  # 분류 메트릭 (6종)
│   ├── regression.py      # 회귀 메트릭 (3종)
│   └── tracker.py         # 메트릭 추적기
│
├── experiment/        # 실험 관리
│   ├── runner.py      # 실험 실행기
│   ├── result.py      # 결과 데이터 클래스
│   └── recorder.py    # 결과 수집기
│
├── comparison/        # 모델 비교
│   ├── comparators.py # 비교 전략 (3종)
│   └── manager.py     # 비교 관리자
│
├── visualization/     # 시각화
│   └── visualizer.py  # 8-패널 차트 생성
│
├── data/              # 데이터 처리
│   ├── cifar10.py     # CIFAR-10 모듈
│   └── factory.py     # 데이터로더 팩토리
│
└── utils/             # 유틸리티
    └── common.py      # 공통 함수
```

---

## 3. 핵심 기능 명세

### 3.1 전이학습 시스템

#### 3.1.1 지원 모델 (13종)

| 모델 계열 | 변형 | 파라미터 수 | ImageNet 정확도 |
|----------|------|------------|----------------|
| **ResNet** | resnet18 | 11.7M | 69.76% |
| | resnet34 | 21.8M | 73.31% |
| | resnet50 | 25.6M | 76.13% |
| | resnet101 | 44.5M | 77.37% |
| | resnet152 | 60.2M | 78.31% |
| **VGG** | vgg11 | 132.9M | 69.02% |
| | vgg11_bn | 132.9M | 70.37% |
| | vgg13 | 133.0M | 69.93% |
| | vgg13_bn | 133.0M | 71.59% |
| | vgg16 | 138.4M | 71.59% |
| | vgg16_bn | 138.4M | 73.36% |
| | vgg19 | 143.7M | 72.38% |
| | vgg19_bn | 143.7M | 74.22% |

#### 3.1.2 학습 모드

```python
# 1. Feature Extraction (특징 추출)
# - 백본 동결, 분류기만 학습
# - 빠른 학습, 적은 데이터에 효과적
exp.run(strategy='feature_extraction')

# 2. Fine-tuning (미세 조정)
# - 전체 레이어 학습
# - 최고 성능, 충분한 데이터 필요
exp.run(strategy='fine_tuning')

# 3. Inference (추론)
# - 평가만 수행, 학습 없음
exp.run(strategy='inference')
```

#### 3.1.3 레이어 제어 기능

| 메서드 | 설명 | 사용 예 |
|--------|------|---------|
| `freeze_backbone()` | 백본 동결 | 특징 추출 모드 |
| `unfreeze_all()` | 전체 활성화 | 미세 조정 모드 |
| `freeze_all()` | 전체 동결 | 추론 모드 |
| `partial_unfreeze(n)` | 부분 해동 | 점진적 학습 |
| `freeze_until_layer(name)` | 특정 레이어까지 동결 | ResNet 전용 |
| `partial_unfreeze_features(n)` | 블록 단위 해동 | VGG 전용 |

### 3.2 태스크 전략 시스템

#### 3.2.1 지원 태스크

| 태스크 | 클래스 | 손실 함수 | 활성화 함수 | 주요 메트릭 |
|--------|--------|-----------|-------------|------------|
| **다중 분류** | `MultiClassStrategy` | CrossEntropyLoss | Softmax | Accuracy |
| **이진 분류** | `BinaryClassificationStrategy` | BCEWithLogitsLoss | Sigmoid | Accuracy, AUC |
| **회귀** | `RegressionStrategy` | MSELoss | None | MSE, MAE, R² |

#### 3.2.2 전략 패턴 구조

```python
# 태스크 전략 인터페이스
class TaskStrategy(ABC):
    @abstractmethod
    def get_criterion(self):
        """손실 함수 반환"""

    @abstractmethod
    def calculate_metric(self, predictions, targets):
        """메트릭 계산"""

    @abstractmethod
    def prepare_labels(self, labels, num_classes):
        """레이블 전처리"""
```

### 3.3 메트릭 시스템

#### 3.3.1 분류 메트릭 (6종)

| 메트릭 | 클래스 | 설명 | 평균 방식 |
|--------|--------|------|-----------|
| **정확도** | `AccuracyMetric` | 전체 정확도 | - |
| **정밀도** | `PrecisionMetric` | 양성 예측의 정확성 | macro/micro/weighted |
| **재현율** | `RecallMetric` | 실제 양성의 검출률 | macro/micro/weighted |
| **F1 점수** | `F1ScoreMetric` | 정밀도와 재현율의 조화평균 | macro/micro/weighted |
| **Top-5 정확도** | `Top5AccuracyMetric` | 상위 5개 예측 중 정답 포함 | - |

#### 3.3.2 회귀 메트릭 (3종)

| 메트릭 | 클래스 | 설명 | 범위 |
|--------|--------|------|------|
| **MSE** | `MSEMetric` | 평균 제곱 오차 | [0, ∞) |
| **MAE** | `MAEMetric` | 평균 절대 오차 | [0, ∞) |
| **R²** | `R2Metric` | 결정 계수 | (-∞, 1] |

#### 3.3.3 메트릭 추적기

```python
# MetricTracker 주요 기능
tracker = MetricTracker(['accuracy', 'precision', 'recall'])
tracker.update(predictions, targets)  # 메트릭 업데이트
tracker.get_latest()                  # 최신 값 조회
tracker.get_best()                    # 최고 값 조회
tracker.get_history()                 # 전체 이력
tracker.summary()                     # 요약 통계
```

### 3.4 실험 관리 시스템

#### 3.4.1 실험 라이프사이클

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐
│  Setup   │ -> │  Train  │ -> │ Evaluate │ -> │  Compare  │
└──────────┘    └─────────┘    └──────────┘    └───────────┘
      ↓               ↓              ↓                ↓
   Config         Metrics         Results          Report
```

#### 3.4.2 ExperimentResult 데이터 구조

```python
@dataclass
class ExperimentResult:
    # 기본 정보
    model_name: str
    task_type: str
    parameters: int

    # 메트릭
    train_metrics: List[float]
    val_metrics: List[float]
    test_metrics: float

    # 손실
    train_losses: List[float]
    val_losses: List[float]
    test_loss: float

    # 성능 지표
    best_val_metric: float
    best_epoch: int
    total_epochs: int

    # 시간 측정
    avg_epoch_time: float
    inference_time: float
    total_training_time: float

    # 설정
    config: dict
    strategy: str
```

### 3.5 모델 비교 시스템

#### 3.5.1 비교 전략 (3종)

| 비교기 | 클래스 | 평가 기준 | 사용 목적 |
|--------|--------|-----------|-----------|
| **성능** | `PerformanceComparator` | 메트릭 값 | 정확도 순위 |
| **효율성** | `EfficiencyComparator` | 성능/log₁₀(파라미터) | 경량 모델 선택 |
| **속도** | `SpeedComparator` | 학습/추론 시간 | 실시간 처리 |

#### 3.5.2 효율성 계산 공식

```
효율성 = 성능 메트릭 / log₁₀(파라미터 수 + 1)
```

### 3.6 시각화 시스템

#### 3.6.1 8-패널 종합 차트

```
┌─────────────────────────────────────────────────────────┐
│                 Experiment Comparison                    │
├──────────────┬──────────────┬──────────────┬───────────┤
│ Train & Val  │  Test Loss   │ Metric Comp  │ Best Perf │
│    Loss      │              │              │           │
├──────────────┼──────────────┼──────────────┼───────────┤
│  Efficiency  │  Epoch Time  │ Inference    │ Overfit   │
│   Scatter    │              │    Time      │    Gap    │
└──────────────┴──────────────┴──────────────┴───────────┘
```

#### 3.6.2 패널별 상세 기능

| 패널 | 내용 | 분석 목적 |
|------|------|-----------|
| **1. 학습/검증 손실** | 에폭별 손실 추이 | 과적합 검출 |
| **2. 테스트 손실** | 최종 성능 | 일반화 능력 |
| **3. 메트릭 비교** | Train/Val/Test 메트릭 | 종합 성능 |
| **4. 최고 성능** | 모델별 최고 점수 | 순위 비교 |
| **5. 효율성 산점도** | 파라미터 vs 성능 | 효율성 분석 |
| **6. 평균 에폭 시간** | 학습 속도 | 학습 효율 |
| **7. 추론 시간** | 예측 속도 | 실시간 가능성 |
| **8. 과적합 갭** | Train-Val 차이 | 과적합 정도 |

---

## 4. 모듈별 상세 기능

### 4.1 Core 모듈

#### 4.1.1 BaseModel 클래스

```python
class BaseModel(nn.Module):
    """전이학습 모델의 추상 베이스 클래스"""

    # 추상 메서드 (구현 필수)
    @abstractmethod
    def _load_pretrained(self) -> nn.Module:
        """사전학습 모델 로드"""

    @abstractmethod
    def _modify_classifier(self, model: nn.Module) -> nn.Module:
        """분류기 수정"""

    @abstractmethod
    def get_backbone_params(self) -> List[nn.Parameter]:
        """백본 파라미터 반환"""

    # 공통 메서드 (상속 가능)
    def freeze_backbone(self):
        """백본 레이어 동결"""

    def unfreeze_all(self):
        """모든 레이어 활성화"""

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
```

#### 4.1.2 Experiment 클래스

```python
class Experiment:
    """실험 관리 파사드 패턴"""

    def __init__(self, config: dict):
        """실험 설정 초기화"""

    def setup(self, model_name: str, data_module,
              task_strategy, logging_strategy=None):
        """실험 환경 설정"""

    def run(self, strategy: str = 'fine_tuning') -> ExperimentResult:
        """단일 실험 실행"""

    def compare_strategies(self, strategies: List[str]) -> List[ExperimentResult]:
        """전략 비교 실험"""

    def evaluate_pretrained(self) -> float:
        """사전학습 모델 평가"""
```

### 4.2 Models 모듈

#### 4.2.1 ModelRegistry

```python
class ModelRegistry:
    """모델 팩토리 + 레지스트리 패턴"""

    @classmethod
    def register(cls, model_type: str, variant: str = None):
        """데코레이터 방식 모델 등록"""

    @classmethod
    def create(cls, model_name: str, **kwargs) -> BaseModel:
        """모델 생성"""

    @classmethod
    def list_models(cls) -> List[str]:
        """등록된 모델 목록"""

    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """모델 정보 조회"""
```

#### 4.2.2 ResNetModel

```python
@ModelRegistry.register('resnet', variant='resnet50')
class ResNetModel(BaseModel):
    """ResNet 모델 구현"""

    def freeze_until_layer(self, layer_name: str):
        """특정 레이어까지 동결"""

    def get_layer_groups(self) -> List[str]:
        """레이어 그룹 목록"""
```

### 4.3 Strategies 모듈

#### 4.3.1 VanillaTrainingStrategy

```python
class VanillaTrainingStrategy(TrainingStrategy):
    """순수 PyTorch 학습 전략"""

    def __init__(self, optimizer='adam', lr=1e-4, device='auto'):
        """학습 설정 초기화"""

    def train(self, model, train_loader, val_loader,
              task_strategy, max_epochs, metric_tracker):
        """모델 학습"""

    def evaluate(self, model, test_loader, task_strategy):
        """모델 평가"""
```

### 4.4 Comparison 모듈

#### 4.4.1 ComparisonManager

```python
class ComparisonManager:
    """모델 비교 관리자"""

    def add_comparator(self, comparator):
        """비교기 추가"""

    def run_all_comparisons(self, results: List[ExperimentResult]):
        """모든 비교 실행"""

    def export_comparison_report(self, filepath: str):
        """비교 리포트 생성"""
```

### 4.5 Visualization 모듈

#### 4.5.1 ExperimentVisualizer

```python
class ExperimentVisualizer:
    """실험 결과 시각화"""

    @staticmethod
    def plot_comparison(results: List[ExperimentResult],
                       save_path: str = None,
                       figsize: tuple = (20, 16),
                       dpi: int = 300):
        """8-패널 종합 비교 차트 생성"""

    @staticmethod
    def plot_metric_comparison(results: List[ExperimentResult],
                              metric_name: str = 'accuracy'):
        """특정 메트릭 비교 차트"""
```

---

## 5. API 명세

### 5.1 주요 임포트

```python
from research import (
    # 핵심 클래스
    Experiment, BaseModel, ModelRegistry,

    # 모델
    ResNetModel, VGGModel, CNN, FullyConnectedNN,

    # 전략
    VanillaTrainingStrategy,
    MultiClassStrategy, BinaryClassificationStrategy, RegressionStrategy,
    SimpleLoggingStrategy, WandBLoggingStrategy,

    # 메트릭
    MetricTracker, AccuracyMetric, PrecisionMetric, RecallMetric,
    F1ScoreMetric, MSEMetric, MAEMetric, R2Metric,

    # 실험 관리
    ExperimentRunner, ExperimentRecorder, ExperimentResult,

    # 비교 및 시각화
    ComparisonManager, ExperimentVisualizer,

    # 데이터
    CIFAR10DataModule, DataLoaderFactory,

    # 유틸리티
    list_models, get_version, print_info
)
```

### 5.2 빠른 시작 API

```python
# 1. 설정 정의
config = {
    'num_classes': 10,
    'in_channels': 3,  # RGB 이미지
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32,
    'optimizer': 'adam'
}

# 2. 실험 생성 및 설정
exp = Experiment(config)
exp.setup(
    model_name='resnet50',
    data_module=CIFAR10DataModule(batch_size=32),
    task_strategy=MultiClassStrategy(num_classes=10)
)

# 3. 실험 실행
result = exp.run(strategy='fine_tuning')

# 4. 전략 비교
results = exp.compare_strategies(['feature_extraction', 'fine_tuning'])

# 5. 시각화
ExperimentVisualizer.plot_comparison(results, save_path='comparison.png')
```

### 5.3 고급 사용법

#### 5.3.1 커스텀 메트릭 추가

```python
from research.metrics import MetricCalculator

class CustomMetric(MetricCalculator):
    def calculate(self, predictions, targets):
        # 커스텀 계산 로직
        return metric_value

    def get_name(self):
        return "custom_metric"

    def is_higher_better(self):
        return True  # 높을수록 좋은 메트릭
```

#### 5.3.2 1채널 입력 처리

```python
# 흑백 이미지나 멜-스펙트로그램용 설정
config = {
    'num_classes': 10,
    'in_channels': 1,  # 1채널 입력
    'learning_rate': 1e-4,
    'max_epochs': 20
}

# 자동으로 첫 번째 Conv 레이어 조정
exp = Experiment(config)
```

#### 5.3.3 레이어별 학습률 조정

```python
model = ResNetModel(num_classes=10)

# 레이어 그룹별 파라미터 분리
backbone_params = model.get_backbone_params()
classifier_params = model.classifier.parameters()

# 차등 학습률 적용
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},  # 백본: 작은 학습률
    {'params': classifier_params, 'lr': 1e-3}  # 분류기: 큰 학습률
])
```

---

## 6. 설정 및 확장

### 6.1 설치 방법

```bash
# 기본 설치 (개발 모드)
pip install -e .

# 전체 종속성 설치
pip install -e ".[all]"

# 개발 도구만 설치
pip install -e ".[dev]"

# 특정 기능 설치
pip install -e ".[wandb]"     # WandB 로깅
pip install -e ".[notebook]"  # Jupyter 지원
```

### 6.2 환경 설정

#### 6.2.1 기본 설정 구조

```python
DEFAULT_CONFIG = {
    # 모델 설정
    'num_classes': int,        # 클래스 수
    'in_channels': 3,          # 입력 채널 (기본: RGB)

    # 학습 설정
    'learning_rate': 1e-4,     # 학습률
    'max_epochs': 100,         # 최대 에폭
    'batch_size': 32,          # 배치 크기
    'optimizer': 'adam',       # 옵티마이저 ('adam', 'adamw', 'sgd')
    'momentum': 0.9,           # SGD용 모멘텀

    # 데이터 설정
    'num_workers': 4,          # 데이터 로더 워커
    'pin_memory': True,        # GPU 메모리 고정

    # 로깅 설정
    'project_name': str,       # 프로젝트명 (WandB용)
    'run_name': str,           # 실행명
    'log_interval': 10,        # 로그 출력 간격

    # 장치 설정
    'device': 'auto',          # 'auto', 'cuda', 'cpu'
}
```

### 6.3 확장 가이드

#### 6.3.1 새로운 사전학습 모델 추가

```python
from research import BaseModel, ModelRegistry

@ModelRegistry.register('efficientnet', variant='b0')
class EfficientNetModel(BaseModel):
    def _load_pretrained(self):
        # 사전학습 모델 로드
        return efficientnet_b0(pretrained=True)

    def _modify_classifier(self, model):
        # 분류기 수정
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, self.num_classes)
        return model

    def get_backbone_params(self):
        # 백본 파라미터 반환
        return list(self.features.parameters())
```

#### 6.3.2 새로운 태스크 전략 추가

```python
from research.strategies.task import TaskStrategy

class MultiLabelStrategy(TaskStrategy):
    """다중 레이블 분류 전략"""

    def get_criterion(self):
        return nn.BCEWithLogitsLoss()

    def calculate_metric(self, predictions, targets):
        preds = torch.sigmoid(predictions) > 0.5
        correct = (preds == targets).float().mean()
        return correct.item()

    def prepare_labels(self, labels, num_classes):
        # 다중 레이블 형식으로 변환
        return labels.float()
```

#### 6.3.3 새로운 비교기 추가

```python
from research.comparison import ModelComparator

class MemoryEfficiencyComparator(ModelComparator):
    """메모리 효율성 비교"""

    def get_comparison_name(self):
        return "Memory Efficiency"

    def compare(self, results):
        # 메모리 사용량 대비 성능 계산
        rankings = []
        for result in results:
            memory_mb = result.parameters * 4 / 1024 / 1024
            efficiency = result.best_val_metric / memory_mb
            rankings.append({
                'model': result.model_name,
                'memory_mb': memory_mb,
                'efficiency': efficiency
            })
        return sorted(rankings, key=lambda x: x['efficiency'], reverse=True)
```

---

## 7. 사용 예제

### 7.1 기본 실험 워크플로우

```python
import torch
from research import (
    Experiment, CIFAR10DataModule,
    MultiClassStrategy, ExperimentRecorder,
    ComparisonManager, ExperimentVisualizer
)

# 1. 데이터 준비
data_module = CIFAR10DataModule(
    batch_size=32,
    num_workers=4,
    image_size=224  # 사전학습 모델용 크기
)
data_module.prepare_data()
data_module.setup()

# 2. 실험 설정
config = {
    'num_classes': 10,
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32
}

# 3. 실험 실행
exp = Experiment(config)
exp.setup(
    model_name='resnet50',
    data_module=data_module,
    task_strategy=MultiClassStrategy(num_classes=10)
)

# 4. 전략 비교
results = exp.compare_strategies(['feature_extraction', 'fine_tuning'])

# 5. 결과 시각화
ExperimentVisualizer.plot_comparison(
    results,
    save_path='cifar10_comparison.png',
    dpi=300
)

# 6. 성능 비교
manager = ComparisonManager()
manager.add_comparator(PerformanceComparator('accuracy'))
manager.add_comparator(EfficiencyComparator('accuracy'))
manager.add_comparator(SpeedComparator())
manager.run_all_comparisons(results)
```

### 7.2 다중 모델 비교

```python
# 여러 모델 비교 실험
models_to_compare = ['resnet18', 'resnet50', 'vgg16', 'vgg19']
recorder = ExperimentRecorder()

for model_name in models_to_compare:
    exp = Experiment(config)
    exp.setup(
        model_name=model_name,
        data_module=data_module,
        task_strategy=MultiClassStrategy(num_classes=10)
    )

    # 미세 조정 실행
    result = exp.run(strategy='fine_tuning')
    recorder.add_result(result)

# 모든 결과 시각화
all_results = recorder.get_all_results()
ExperimentVisualizer.plot_comparison(
    all_results,
    save_path='model_comparison.png'
)

# 요약 출력
recorder.print_summary()
```

### 7.3 커스텀 데이터셋 사용

```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # 데이터 로드 로직

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 데이터 반환 로직
        return image, label

# 커스텀 데이터로더 생성
train_dataset = CustomDataset(transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 실험에 사용
exp = Experiment(config)
exp.setup(
    model_name='resnet50',
    data_module={'train': train_loader, 'val': val_loader, 'test': test_loader},
    task_strategy=MultiClassStrategy(num_classes=num_classes)
)
```

### 7.4 WandB 로깅 통합

```python
from research import WandBLoggingStrategy

# WandB 로깅 설정
logging_strategy = WandBLoggingStrategy(
    project_name='ktb-ml-experiments',
    run_name='resnet50_cifar10',
    config=config
)

# 실험에 로깅 추가
exp = Experiment(config)
exp.setup(
    model_name='resnet50',
    data_module=data_module,
    task_strategy=MultiClassStrategy(num_classes=10),
    logging_strategy=logging_strategy
)

result = exp.run(strategy='fine_tuning')
```

### 7.5 점진적 언프리징 (Progressive Unfreezing)

```python
model = ResNetModel(num_classes=10)
trainer = VanillaTrainingStrategy(lr=1e-4)

# 단계별 언프리징 전략
unfreezing_schedule = [
    (0, 0),    # 0-5 에폭: 백본 동결
    (5, 2),    # 5-10 에폭: 마지막 2개 레이어 언프리징
    (10, 4),   # 10-15 에폭: 마지막 4개 레이어 언프리징
    (15, -1)   # 15+ 에폭: 전체 언프리징
]

for epoch in range(max_epochs):
    # 언프리징 스케줄 적용
    for schedule_epoch, num_layers in unfreezing_schedule:
        if epoch == schedule_epoch:
            if num_layers == -1:
                model.unfreeze_all()
            elif num_layers == 0:
                model.freeze_backbone()
            else:
                model.partial_unfreeze(num_layers)

    # 학습 수행
    train_loss = trainer.train_epoch(model, train_loader, task_strategy)
```

---

## 부록 A. 성능 벤치마크

### CIFAR-10 데이터셋 기준

| 모델 | 전략 | 정확도 | 학습 시간 | 추론 시간 (ms) |
|------|------|--------|-----------|---------------|
| ResNet18 | Feature Extraction | 91.2% | 5분 | 8.3 |
| ResNet18 | Fine-tuning | 93.5% | 20분 | 8.3 |
| ResNet50 | Feature Extraction | 92.1% | 8분 | 15.2 |
| ResNet50 | Fine-tuning | 94.8% | 35분 | 15.2 |
| VGG16 | Feature Extraction | 90.8% | 12분 | 22.1 |
| VGG16 | Fine-tuning | 93.2% | 45분 | 22.1 |

## 부록 B. 트러블슈팅

### 일반적인 문제 해결

| 문제 | 원인 | 해결 방법 |
|------|------|-----------|
| CUDA out of memory | 배치 크기가 너무 큼 | 배치 크기 감소 |
| 낮은 정확도 | 학습률이 부적절 | 학습률 조정 (1e-3 ~ 1e-5) |
| 과적합 | 데이터 부족 | 데이터 증강, Dropout 추가 |
| 느린 수렴 | 백본 동결 | Fine-tuning 모드 사용 |

## 부록 C. 버전 히스토리

| 버전 | 날짜 | 주요 변경사항 |
|------|------|--------------|
| 0.1.0 | 2024.01 | 초기 릴리즈, 13개 모델 지원 |
| 0.0.9 | 2023.12 | 1채널 입력 지원 추가 |
| 0.0.8 | 2023.11 | 8-패널 시각화 추가 |

---

## 문서 정보

- **작성일**: 2025년 11월
- **버전**: 1.0.0
- **작성자**: KTB AI Lab
- **라이선스**: MIT

본 기능명세서는 KTB ML Framework의 모든 기능을 상세히 문서화한 것입니다.
프레임워크 사용에 대한 추가 질문이나 지원이 필요한 경우,
GitHub Issues 또는 공식 문서를 참조해 주시기 바랍니다.
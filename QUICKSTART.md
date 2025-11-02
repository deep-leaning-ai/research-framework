# KTB ML Framework - 빠른 시작 가이드

이 가이드는 KTB ML Framework를 빠르게 시작할 수 있도록 도와줍니다.

## 설치

### 기본 설치

```bash
pip install -e .
```

### 모든 기능 포함 설치

```bash
pip install -e ".[all]"
```

### 개발 모드 설치

```bash
pip install -e ".[dev]"
```

---

## 5분 빠른 시작

### 1. 메트릭 트래커 사용하기

```python
from research import MetricTracker, AccuracyMetric, F1ScoreMetric
import torch

# 메트릭 트래커 생성
tracker = MetricTracker([
    AccuracyMetric(),
    F1ScoreMetric(average='macro')
])

# 모델 출력과 레이블
outputs = torch.randn(32, 10)  # (batch_size, num_classes)
labels = torch.randint(0, 10, (32,))

# 메트릭 계산
metrics = tracker.update(outputs, labels)
print(metrics)  # {'Accuracy': 12.5, 'F1-Score (macro)': 10.3}
```

### 2. Task Strategy 사용하기

```python
from research import MultiClassStrategy, BinaryClassificationStrategy
import torch.nn as nn

# 다중 분류 전략
multiclass_strategy = MultiClassStrategy(num_classes=10)
criterion = multiclass_strategy.get_criterion()  # CrossEntropyLoss
metric_name = multiclass_strategy.get_metric_name()  # "Accuracy"

# 이진 분류 전략
binary_strategy = BinaryClassificationStrategy()
criterion = binary_strategy.get_criterion()  # BCEWithLogitsLoss
activation = binary_strategy.get_output_activation()  # Sigmoid
```

### 3. 실험 결과 기록 및 시각화

```python
from research import (
    ExperimentRecorder,
    ExperimentResult,
    ExperimentVisualizer
)

# 기록기 생성
recorder = ExperimentRecorder()

# 실험 결과 추가
result = ExperimentResult(
    model_name='ResNet18',
    task_type='MultiClassStrategy',
    train_loss=[1.5, 1.2, 0.9, 0.7, 0.5],
    val_loss=[1.6, 1.3, 1.0, 0.8, 0.6],
    test_loss=[1.65, 1.35, 1.05, 0.85, 0.65],
    train_metrics={'Accuracy': [70, 75, 80, 85, 90]},
    val_metrics={'Accuracy': [68, 73, 78, 83, 88]},
    test_metrics={'Accuracy': [67, 72, 77, 82, 87]},
    primary_metric_name='Accuracy',
    best_test_metric=87.0,
    parameters=11_000_000,
    epoch_times=[2.0, 2.1, 2.0, 2.2, 2.1],
    inference_time=0.05
)

recorder.add_result(result)

# 시각화 생성 (8-panel 차트)
ExperimentVisualizer.plot_comparison(
    recorder=recorder,
    save_path='results.png'
)
```

### 4. 모델 비교

```python
from research import (
    ComparisonManager,
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator
)

# 비교 관리자 생성
manager = ComparisonManager()

# 비교 방식 추가
manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
manager.add_comparator(EfficiencyComparator('Accuracy'))
manager.add_comparator(SpeedComparator())

# 모든 비교 실행
results = manager.run_all_comparisons(recorder.get_all_results())

# 리포트 저장
manager.export_comparison_report('comparison_report.txt')
```

---

## 전체 워크플로우 예제

전체 워크플로우를 한 번에 보려면:

```bash
python3 examples/quickstart.py
```

이 예제는 다음을 포함합니다:
- 메트릭 트래커 생성 및 사용
- 3개 모델의 학습 시뮬레이션
- 실험 결과 기록
- 다중 관점 모델 비교
- 종합 시각화 생성

---

## 전이학습 예제 (ResNet + CIFAR-10)

```python
from research import (
    Experiment,
    VanillaTrainingStrategy,
    SimpleLoggingStrategy,
    CIFAR10DataModule,
)

# 1. 데이터 모듈 생성
data_module = CIFAR10DataModule(
    data_dir='./data',
    batch_size=32,
    num_workers=4
)

# 2. 실험 설정
config = {
    'num_classes': 10,
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32,
    'optimizer': 'adam'
}

# 3. Experiment 생성
exp = Experiment(config)
exp.setup(
    model_name='resnet18',
    data_module=data_module,
    training_strategy=VanillaTrainingStrategy(),
    logging_strategy=SimpleLoggingStrategy()
)

# 4. 전략 비교 (Feature Extraction vs Fine-tuning)
comparison = exp.compare_strategies([
    'feature_extraction',  # 백본 동결, 분류기만 학습
    'fine_tuning'          # 전체 네트워크 학습
])

print(comparison)
```

---

## 사용 가능한 모델 확인

```python
import research as ktb

# Pretrained 모델 목록
ktb.list_models()

# Simple 모델 목록
ktb.list_simple_models()

# 버전 확인
print(ktb.get_version())

# 프레임워크 정보
ktb.print_info()
```

---

## 주요 기능별 예제

### 메트릭 시스템

```bash
python3 examples/test_metric_system.py
```

### 시각화 도구

```bash
python3 examples/test_visualization.py
```

### Task Strategy

```bash
python3 examples/test_task_strategies.py
```

### 비교 시스템

```bash
python3 examples/test_comparison_system.py
```

---

## pytest로 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/

# Unit 테스트만
pytest tests/unit/

# Integration 테스트만
pytest tests/integration/

# 특정 테스트만
pytest tests/unit/test_metrics.py

# 상세 출력
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=research
```

---

## 디렉토리 구조

```
research/
├── core/              # 핵심 추상 클래스
├── models/
│   ├── pretrained/    # ResNet, VGG 등
│   └── simple/        # CNN, FullyConnectedNN
├── strategies/
│   ├── training/      # 학습 전략
│   ├── logging/       # 로깅 전략
│   └── task/          # Task 전략
├── metrics/           # 메트릭 시스템
├── experiment/        # 실험 실행 및 기록
├── comparison/        # 모델 비교
├── visualization/     # 시각화 도구
└── analysis/          # 분석 도구

examples/              # 예제 코드
tests/                 # pytest 테스트
├── unit/             # 단위 테스트
└── integration/      # 통합 테스트
```

---

## 다음 단계

1. **예제 실행**: `examples/` 디렉토리의 예제들을 실행해보세요
2. **API 문서**: 각 모듈의 docstring을 참고하세요
3. **테스트 코드**: `tests/` 디렉토리의 테스트 코드를 참고하세요
4. **실제 프로젝트**: 자신의 데이터셋으로 전이학습을 시도해보세요

---

## 문의 및 지원

- **GitHub Issues**: https://github.com/ktb-ai/ktb-ml-framework/issues
- **Email**: ai-research@ktb.com
- **문서**: README.md, examples/README.md 참고

---

**KTB ML Framework** - Making ML experimentation easier and more structured.

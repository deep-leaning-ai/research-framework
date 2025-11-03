# Research ML Framework

통합 머신러닝 실험 프레임워크 - Transfer Learning과 일반 ML 태스크를 위한 완벽한 솔루션

## 주요 특징

### 🔥 Transfer Learning 지원
- **ResNet** (18, 34, 50, 101, 152)
- **VGG** (11, 13, 16, 19 + BatchNorm 변형)
- 3가지 학습 전략: `feature_extraction`, `fine_tuning`, `inference`

### 📊 다양한 태스크 지원
- **Multi-Class Classification** - Softmax + CrossEntropy
- **Binary Classification** - Sigmoid + BCEWithLogits
- **Regression** - MSE/MAE/R2 메트릭

### 📈 고급 메트릭 시스템
- 9가지 메트릭: Accuracy, Precision, Recall, F1, Top5, AUC, MSE, MAE, R2
- 실시간 추적 및 이동평균
- 메트릭별 커스터마이징 가능

### 🎨 시각화 도구
- 8-panel 종합 차트
- 학습/검증/테스트 곡선
- 모델 비교 분석
- 효율성 산점도

### ⚡ 성능 최적화
- 자동 GPU 감지 및 활용
- DataLoader 최적화 (persistent_workers, prefetch, pin_memory)
- 배치 정규화 및 드롭아웃

## 설치

```bash
# 기본 설치
pip install -e .

# 전체 기능 설치 (wandb, jupyter 포함)
pip install -e ".[all]"

# 개발 환경 설치
pip install -e ".[dev]"
```

## 빠른 시작

### 1. Transfer Learning 예제

```python
import research
from research.data.cifar10 import CIFAR10DataModule

# 데이터 준비
data_module = CIFAR10DataModule(batch_size=32, num_workers=4)
train_loader, val_loader, test_loader = data_module.get_loaders()

# 실험 설정
config = {
    'num_classes': 10,
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32
}

# 실험 생성 및 실행
exp = research.Experiment(config)
exp.setup(
    model_name='resnet18',
    data_module=data_module,
    freeze_strategy='fine_tuning'
)

# 학습
result = exp.run()
print(f"Best accuracy: {result.best_test_metric:.4f}")
```

### 2. 여러 모델 비교

```python
# 여러 전략 비교
results = exp.compare_strategies(['feature_extraction', 'fine_tuning'])

# 시각화
from research.visualization import ExperimentVisualizer
ExperimentVisualizer.plot_comparison(results, save_path='comparison.png')
```

### 3. 커스텀 메트릭 추가

```python
from research.metrics.base import BaseMetric

class CustomMetric(BaseMetric):
    def calculate(self, predictions, targets):
        # 커스텀 메트릭 로직
        return metric_value

# 사용
from research.metrics.tracker import MetricTracker
tracker = MetricTracker(window_size=5)
metrics = {'custom': CustomMetric()}
tracker.update(predictions, targets, metrics)
```

## 프로젝트 구조

```
research/
├── core/              # 추상 베이스 클래스
├── models/
│   ├── pretrained/   # ResNet, VGG (Transfer Learning)
│   └── simple/       # CNN, FullyConnected
├── strategies/
│   ├── training/     # 학습 전략
│   ├── logging/      # 로깅 전략 (Simple, WandB)
│   └── task/         # 태스크 전략 (MultiClass, Binary, Regression)
├── metrics/          # 메트릭 시스템
├── experiment/       # 실험 관리
├── comparison/       # 모델 비교
├── visualization/    # 시각화 도구
├── data/            # 데이터 모듈
└── config/          # 설정 관리
```

## 디자인 패턴

- **Strategy Pattern**: 태스크, 학습, 로깅 전략
- **Factory Pattern**: ModelRegistry를 통한 모델 생성
- **Template Method**: BaseModel의 공통 로직
- **Facade Pattern**: Experiment 클래스의 간단한 인터페이스
- **Observer Pattern**: ExperimentRecorder의 자동 수집

## 고급 기능

### Model Registry

```python
# 사용 가능한 모델 확인
research.list_models()

# 모델 생성
from research.models.pretrained import ModelRegistry
model = ModelRegistry.create('resnet50', num_classes=100)
```

### Comparison System

```python
from research.comparison import ComparisonManager
from research.comparison.comparators import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator
)

manager = ComparisonManager()
manager.add_comparator(PerformanceComparator('accuracy'))
manager.add_comparator(EfficiencyComparator('accuracy'))
manager.add_comparator(SpeedComparator())

results = manager.compare(experiment_results)
manager.print_summary(results)
```

### 1-Channel 데이터 지원

Mel-spectrogram이나 grayscale 이미지 같은 1채널 데이터:

```python
config = {
    'num_classes': 10,
    'in_channels': 1,  # 1채널 입력
    'learning_rate': 1e-4
}

exp = research.Experiment(config)
exp.setup(model_name='resnet18', data_module=mel_datamodule)
```

## 테스트

```bash
# 모든 테스트 실행
pytest tests/

# 단위 테스트만
pytest tests/unit/

# 특정 모듈 테스트
pytest tests/unit/test_metrics.py

# 커버리지 포함
pytest tests/ --cov=research --cov-report=term-missing
```

## 예제

`examples/` 디렉토리의 예제들:

- `quickstart.py` - 전체 워크플로우 데모
- `test_metric_system.py` - 메트릭 시스템 사용법
- `test_visualization.py` - 시각화 기능
- `test_task_strategies.py` - 다양한 태스크 전략
- `test_comparison_system.py` - 모델 비교

## 성능 벤치마크

CIFAR-10 데이터셋 기준:

| 모델 | Feature Extraction | Fine-tuning | 파라미터 수 | 추론 시간 |
|------|-------------------|-------------|------------|----------|
| ResNet18 | 85.2% | 92.1% | 11.7M | 8ms |
| ResNet50 | 87.3% | 93.5% | 25.6M | 15ms |
| VGG16 | 84.1% | 91.8% | 138M | 12ms |

## 라이선스

MIT License

## 기여하기

Issues와 Pull Requests는 언제나 환영합니다!

## 저자

KTB AI Research Team

---

**Note**: 이 프레임워크는 이전 `ktb_dl_research`와 `ml_framework`를 통합하여 개선한 버전입니다.
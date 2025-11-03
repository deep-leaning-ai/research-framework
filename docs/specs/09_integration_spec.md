# Integration & Experiment Facade 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [Experiment Facade 클래스](#2-experiment-facade-클래스)
3. [통합 워크플로우](#3-통합-워크플로우)
4. [Freeze 전략 시스템](#4-freeze-전략-시스템)
5. [End-to-End 프로세스](#5-end-to-end-프로세스)
6. [API 명세](#6-api-명세)
7. [통합 이슈 및 해결](#7-통합-이슈-및-해결)
8. [테스트 요구사항](#8-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Integration 시스템은 KTB ML Framework의 모든 컴포넌트를 통합하여 간단한 인터페이스로 제공합니다. Facade Pattern을 사용하여 복잡한 내부 구조를 은닉하고 사용자 친화적인 API를 제공합니다.

### 1.2 설계 원칙

- **Facade Pattern**: 복잡한 서브시스템을 단순한 인터페이스로 통합
- **일관성**: 모든 실험이 동일한 프로세스 준수
- **상태 관리**: 실험 간 상태 격리
- **하드코딩 지양**: 모든 설정은 config로 관리

### 1.3 파일 구조

```
research/core/
├── __init__.py           # Core exports
├── experiment.py         # Experiment Facade (구현 완료)
├── base_model.py         # BaseModel 추상 클래스
└── strategies.py         # Strategy 베이스 클래스들
```

---

## 2. Experiment Facade 클래스

### 2.1 개요

모든 실험 워크플로우를 관리하는 중앙 진입점입니다. 복잡한 설정과 실행 과정을 단순화합니다.

### 2.2 클래스 구조

```python
class Experiment:
    """실험 관리 Facade"""

    # 클래스 상수
    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_OPTIMIZER = 'adam'

    # Freeze 전략
    FREEZE_STRATEGIES = {
        'feature_extraction': 'freeze_backbone',
        'fine_tuning': 'unfreeze_all',
        'inference': 'freeze_all'
    }

    def __init__(self, config: Dict[str, Any]):
        """초기화

        Args:
            config: 실험 설정 딕셔너리
                - num_classes: int (필수)
                - in_channels: int (기본값: 3)
                - learning_rate: float
                - max_epochs: int
                - batch_size: int
                - optimizer: str
                - device: str
        """
        self.config = config
        self.model = None
        self.data_module = None
        self.training_strategy = None
        self.logging_strategy = None
        self.results = {}
        self.experiment_history = []
```

### 2.3 주요 메서드

#### setup() 메서드

```python
def setup(self,
          model_name: str,
          data_module: Union[DataModule, Dict[str, DataLoader]],
          training_strategy: TrainingStrategy = None,
          logging_strategy: LoggingStrategy = None):
    """실험 환경 설정

    프로세스:
    1. ModelRegistry로 모델 생성
    2. 데이터 모듈 저장
    3. 전략 설정 (기본값: VanillaTrainingStrategy)
    4. 로깅 전략 설정 (선택)
    5. 데이터 준비 (prepare_data, setup 호출)
    6. 설정 요약 출력

    Args:
        model_name: 모델 이름 ('resnet18', 'vgg16' 등)
        data_module: DataModule 또는 DataLoader 딕셔너리
        training_strategy: 학습 전략
        logging_strategy: 로깅 전략

    Example:
        exp.setup(
            model_name='resnet50',
            data_module=CIFAR10DataModule(),
            training_strategy=VanillaTrainingStrategy()
        )
    """
```

#### run() 메서드

```python
def run(self,
        strategy: str = 'fine_tuning',
        run_name: str = None) -> Dict[str, Any]:
    """단일 실험 실행

    프로세스:
    1. 로깅 초기화 (있을 경우)
    2. Freeze 전략 적용:
       - 'feature_extraction': model.freeze_backbone()
       - 'fine_tuning': model.unfreeze_all()
       - 'inference': model.freeze_all()
    3. 모델 정보 출력 (파라미터 수, 학습 가능 비율)
    4. 학습 실행 (inference 제외)
    5. 테스트 평가
    6. 결과 저장 및 로깅
    7. 히스토리 기록

    Args:
        strategy: 'feature_extraction', 'fine_tuning', 'inference'
        run_name: 실행 이름 (로깅용)

    Returns:
        {
            'model': 학습된 모델,
            'training_results': 학습 결과,
            'test_results': 테스트 결과,
            'model_info': 모델 정보
        }
    """
```

#### compare_strategies() 메서드

```python
def compare_strategies(self,
                      strategies: List[str],
                      reset_model: bool = True) -> Dict[str, Any]:
    """여러 전략 비교 실험

    프로세스:
    1. 각 전략에 대해:
       a. 모델 리셋 (옵션)
       b. run() 실행
       c. 결과 수집
    2. 비교 테이블 출력
    3. 모든 결과 반환

    Args:
        strategies: 비교할 전략 리스트
        reset_model: 각 실행 전 모델 리셋 여부

    Returns:
        {
            'feature_extraction': {...},
            'fine_tuning': {...},
            ...
        }

    Example:
        results = exp.compare_strategies(
            ['feature_extraction', 'fine_tuning']
        )
    """
```

#### evaluate_pretrained() 메서드

```python
def evaluate_pretrained(self) -> Dict[str, Any]:
    """사전학습 모델 평가 (추론 전용)

    프로세스:
    1. 모든 파라미터 동결
    2. 테스트 데이터로 평가
    3. 상세 메트릭 계산
    4. Confusion matrix 생성

    Returns:
        {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': ndarray,
            'classification_report': str
        }
    """
```

---

## 3. 통합 워크플로우

### 3.1 컴포넌트 상호작용

```
┌──────────────┐
│  Experiment  │  ← Facade (사용자 인터페이스)
└──────┬───────┘
        │
        ├─────────────┬──────────────┬─────────────┐
        ↓             ↓              ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ ModelRegistry│ │DataModule│ │ Training │ │ Logging  │
│              │ │          │ │ Strategy │ │ Strategy │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
        │             │              │             │
        ↓             ↓              ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   BaseModel  │ │DataLoader│ │  Trainer │ │  Logger  │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 3.2 데이터 플로우

```python
# 1. Config → Experiment
config = {'num_classes': 10, 'learning_rate': 1e-4}
exp = Experiment(config)

# 2. Setup → 컴포넌트 초기화
exp.setup(
    model_name='resnet18',
    data_module=data_module,
    training_strategy=strategy
)
# 내부:
# - ModelRegistry.create('resnet18', **config)
# - data_module.prepare_data()
# - data_module.setup()

# 3. Run → 실행
result = exp.run(strategy='fine_tuning')
# 내부:
# - model.unfreeze_all()
# - strategy.train(model, train_loader, val_loader, config)
# - strategy.evaluate(model, test_loader)

# 4. 결과 저장
exp.results['fine_tuning'] = result
exp.experiment_history.append(result)
```

---

## 4. Freeze 전략 시스템

### 4.1 전략 정의

| 전략 | 메서드 호출 | 학습 대상 | 사용 시나리오 |
|------|------------|-----------|--------------|
| **feature_extraction** | freeze_backbone() | 분류기만 | 작은 데이터셋 |
| **fine_tuning** | unfreeze_all() | 전체 모델 | 충분한 데이터 |
| **inference** | freeze_all() | 없음 | 평가만 |

### 4.2 구현 상세

```python
def _apply_freeze_strategy(self, strategy: str):
    """Freeze 전략 적용

    구현:
    if strategy == 'feature_extraction':
        self.model.freeze_backbone()
        print(f"Feature extraction mode: backbone frozen")

    elif strategy == 'fine_tuning':
        self.model.unfreeze_all()
        print(f"Fine-tuning mode: all layers trainable")

    elif strategy == 'inference':
        self.model.freeze_all()
        print(f"Inference mode: all layers frozen")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    """
```

### 4.3 파라미터 통계

```python
def _get_model_info(self) -> Dict[str, Any]:
    """모델 정보 수집

    Returns:
        {
            'total_params': 총 파라미터 수,
            'trainable_params': 학습 가능 파라미터 수,
            'trainable_ratio': 학습 가능 비율,
            'frozen_layers': 동결된 레이어 리스트
        }
    """
    total = sum(p.numel() for p in self.model.parameters())
    trainable = sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'trainable_ratio': trainable / total if total > 0 else 0
    }
```

### 4.4 현재 한계

1. **경직된 전략**
   - 3가지 고정 전략만 지원
   - 부분 언프리징 불가

2. **세밀한 제어 부족**
   - 레이어별 학습률 조정 불가
   - 점진적 언프리징 미지원

---

## 5. End-to-End 프로세스

### 5.1 전체 워크플로우

```python
# 완전한 E2E 예제

# 1. 데이터 준비
from research.data import CIFAR10DataModule

data_module = CIFAR10DataModule(
    batch_size=32,
    num_workers=4,
    image_size=224
)

# 2. 설정 정의
config = {
    'num_classes': 10,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32,
    'optimizer': 'adam'
}

# 3. Experiment 생성
from research import Experiment

exp = Experiment(config)

# 4. 환경 설정
from research.strategies.training import VanillaTrainingStrategy
from research.strategies.logging import SimpleLoggingStrategy

exp.setup(
    model_name='resnet18',
    data_module=data_module,
    training_strategy=VanillaTrainingStrategy(),
    logging_strategy=SimpleLoggingStrategy()
)

# 5. 단일 전략 실행
result = exp.run(strategy='fine_tuning', run_name='resnet18_ft')

# 6. 결과 확인
print(f"Test Accuracy: {result['test_results']['test_acc']:.4f}")
print(f"Training Time: {result['training_results']['training_time']:.2f}s")

# 7. 전략 비교
comparison = exp.compare_strategies(
    ['feature_extraction', 'fine_tuning'],
    reset_model=True
)

# 8. 시각화
from research.visualization import ExperimentVisualizer
from research.experiment import ExperimentRecorder

recorder = ExperimentRecorder()
for strategy, res in comparison.items():
    # ExperimentResult 생성 및 기록
    recorder.add_result(create_experiment_result(res))

ExperimentVisualizer.plot_comparison(
    recorder.get_all_results(),
    save_path='e2e_comparison.png'
)

# 9. 모델 비교
from research.comparison import ComparisonManager

manager = ComparisonManager()
manager.add_comparator(PerformanceComparator('accuracy'))
manager.add_comparator(EfficiencyComparator('accuracy'))
manager.run_all_comparisons(recorder.get_all_results())
```

### 5.2 프로세스 단계

| 단계 | 작업 | 컴포넌트 |
|------|------|----------|
| 1. 초기화 | Config 설정 | Experiment |
| 2. 설정 | 모델/데이터 준비 | ModelRegistry, DataModule |
| 3. 실행 | 학습/평가 | TrainingStrategy |
| 4. 기록 | 결과 수집 | ExperimentRecorder |
| 5. 비교 | 성능 분석 | ComparisonManager |
| 6. 시각화 | 차트 생성 | ExperimentVisualizer |

---

## 6. API 명세

### 6.1 Experiment 클래스 전체 API

```python
class Experiment:
    # 생성자
    def __init__(self, config: Dict[str, Any])

    # 설정
    def setup(self,
              model_name: str,
              data_module: Union[DataModule, Dict],
              training_strategy: TrainingStrategy = None,
              logging_strategy: LoggingStrategy = None)

    # 실행
    def run(self,
            strategy: str = 'fine_tuning',
            run_name: str = None) -> Dict[str, Any]

    # 비교
    def compare_strategies(self,
                          strategies: List[str],
                          reset_model: bool = True) -> Dict[str, Any]

    # 평가
    def evaluate_pretrained(self) -> Dict[str, Any]

    # 내부 메서드
    def _apply_freeze_strategy(self, strategy: str)
    def _get_model_info(self) -> Dict[str, Any]
    def _reset_model(self)
    def _print_comparison_table(self, results: Dict)
```

### 6.2 Config 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| num_classes | int | ✅ | - | 출력 클래스 수 |
| in_channels | int | ❌ | 3 | 입력 채널 수 |
| learning_rate | float | ❌ | 1e-4 | 학습률 |
| max_epochs | int | ❌ | 100 | 최대 에폭 |
| batch_size | int | ❌ | 32 | 배치 크기 |
| optimizer | str | ❌ | 'adam' | 옵티마이저 |
| momentum | float | ❌ | 0.9 | SGD momentum |
| device | str | ❌ | 'auto' | 실행 디바이스 |

### 6.3 반환값 구조

#### run() 반환값

```python
{
    'model': nn.Module,           # 학습된 모델
    'training_results': {
        'training_time': float,   # 학습 시간
        'best_val_acc': float,    # 최고 검증 정확도
        'history': {
            'train_loss': List[float],
            'train_acc': List[float],
            'val_loss': List[float],
            'val_acc': List[float]
        }
    },
    'test_results': {
        'test_loss': float,       # 테스트 손실
        'test_acc': float,        # 테스트 정확도
        'inference_time': float   # 추론 시간 (ms)
    },
    'model_info': {
        'total_params': int,      # 총 파라미터
        'trainable_params': int,  # 학습 가능 파라미터
        'trainable_ratio': float  # 학습 가능 비율
    }
}
```

---

## 7. 통합 이슈 및 해결

### 7.1 중대한 통합 문제

#### **TaskStrategy 미통합 (가장 심각)**

**문제**:
- VanillaTrainingStrategy가 TaskStrategy를 완전히 무시
- CrossEntropyLoss 하드코딩으로 추상화 파괴
- 전체 설계 의도 무력화

**영향**:
- 회귀 태스크 사용 불가
- 이진 분류 부정확
- 커스텀 손실 함수 불가

**해결 방안**:
```python
# VanillaTrainingStrategy 수정 필요
def __init__(self, device=None, task_strategy=None):
    self.task_strategy = task_strategy

def train(self, model, train_loader, val_loader, config):
    # TaskStrategy 사용
    criterion = self.task_strategy.get_criterion()
    # ...
    metric = self.task_strategy.calculate_metric(outputs, labels)
```

### 7.2 상태 관리 문제

**문제**:
- 모델 리셋시 새 인스턴스 생성 (학습 상태 손실)
- 실험 간 상태 격리 불완전
- experiment_history 메모리 누적

**해결 방안**:
```python
def _reset_model(self):
    # 상태 저장
    checkpoint = self.model.state_dict()

    # 새 모델 생성
    self.model = ModelRegistry.create(...)

    # 가중치만 리셋 (구조 유지)
    self.model.load_state_dict(checkpoint, strict=False)
```

### 7.3 캡슐화 위반

**문제**:
- 내부 모델 직접 노출 (Lines 155-161)
- 추상화 레벨 깨짐

**해결 방안**:
```python
# 모델 접근 제한
@property
def model(self):
    return self._model  # 읽기 전용

def get_model_for_inference(self):
    # 추론용 복사본 반환
    return copy.deepcopy(self._model).eval()
```

---

## 8. 테스트 요구사항

### 8.1 통합 테스트

```python
def test_full_integration():
    """전체 E2E 통합 테스트"""

    # 1. 최소 설정으로 시작
    config = {'num_classes': 10}
    exp = Experiment(config)

    # 2. Setup
    exp.setup(
        model_name='resnet18',
        data_module=create_dummy_datamodule()
    )

    # 3. 각 전략 테스트
    for strategy in ['feature_extraction', 'fine_tuning', 'inference']:
        result = exp.run(strategy=strategy)
        assert 'model' in result
        assert 'test_results' in result

    # 4. 비교
    comparison = exp.compare_strategies(
        ['feature_extraction', 'fine_tuning']
    )
    assert len(comparison) == 2
```

### 8.2 Freeze 전략 테스트

```python
def test_freeze_strategies():
    """Freeze 전략 동작 테스트"""

    exp = Experiment({'num_classes': 10})
    exp.setup('resnet18', dummy_datamodule)

    # Feature extraction
    exp._apply_freeze_strategy('feature_extraction')
    backbone_frozen = all(
        not p.requires_grad
        for p in exp.model.get_backbone_params()
    )
    assert backbone_frozen

    # Fine-tuning
    exp._apply_freeze_strategy('fine_tuning')
    all_trainable = all(
        p.requires_grad
        for p in exp.model.parameters()
    )
    assert all_trainable
```

### 8.3 상태 격리 테스트

```python
def test_state_isolation():
    """실험 간 상태 격리 테스트"""

    exp = Experiment({'num_classes': 10})
    exp.setup('resnet18', dummy_datamodule)

    # 첫 번째 실행
    result1 = exp.run('fine_tuning')
    model_state1 = exp.model.state_dict()

    # 리셋 후 두 번째 실행
    exp._reset_model()
    result2 = exp.run('fine_tuning')
    model_state2 = exp.model.state_dict()

    # 상태 다름 확인
    for key in model_state1:
        assert not torch.equal(model_state1[key], model_state2[key])
```

### 8.4 Edge Case 테스트

```python
def test_edge_cases():
    """Edge case 처리"""

    # 1. 빈 config
    exp = Experiment({})  # 기본값 사용
    assert exp.config['max_epochs'] == 100

    # 2. 잘못된 전략
    with pytest.raises(ValueError):
        exp.run(strategy='unknown_strategy')

    # 3. Setup 없이 run
    with pytest.raises(RuntimeError):
        exp.run()

    # 4. 1-채널 입력
    config = {'num_classes': 10, 'in_channels': 1}
    exp = Experiment(config)
    # 그레이스케일 지원 확인
```

---

## 부록: 개선 로드맵

### 우선순위 P0 (긴급)

1. **TaskStrategy 통합**
   - VanillaTrainingStrategy 수정
   - 전체 추상화 복원

2. **상태 관리 개선**
   - 체크포인트 시스템
   - 메모리 관리

### 우선순위 P1 (중요)

1. **고급 Freeze 전략**
   ```python
   strategies = {
       'gradual_unfreeze': partial_unfreeze,
       'discriminative_lr': layer_wise_lr,
       'freeze_bn': freeze_batch_norm
   }
   ```

2. **하이퍼파라미터 튜닝**
   ```python
   def tune_hyperparameters(self, param_grid: Dict):
       # Grid search 또는 Bayesian optimization
       pass
   ```

### 우선순위 P2 (개선)

1. **실험 추적 강화**
   - MLflow 통합
   - TensorBoard 지원

2. **분산 학습 지원**
   - Multi-GPU
   - Distributed training

---

## SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ⚠️ | Facade이지만 너무 많은 책임 |
| **O**pen/Closed | ⚠️ | 새 전략 추가시 수정 필요 |
| **L**iskov Substitution | ✅ | 인터페이스 일관성 유지 |
| **I**nterface Segregation | ✅ | 필요한 메서드만 노출 |
| **D**ependency Inversion | ⚠️ | 일부 구체적 구현 의존 |

Experiment 클래스는 Facade Pattern으로서 적절하지만, 책임 분리와 확장성 개선이 필요합니다.
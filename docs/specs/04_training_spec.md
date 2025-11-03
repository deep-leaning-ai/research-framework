# Training 전략 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [TrainingStrategy 인터페이스](#2-trainingstrategy-인터페이스)
3. [VanillaTrainingStrategy](#3-vanillatrainingstrategy)
4. [현재 구현 상태](#4-현재-구현-상태)
5. [API 명세](#5-api-명세)
6. [개선 필요사항](#6-개선-필요사항)
7. [테스트 요구사항](#7-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Training 전략 시스템은 모델 학습 로직을 캡슐화하여 다양한 학습 방법을 일관된 인터페이스로 제공합니다. Strategy Pattern을 사용하여 학습 알고리즘을 실행 시점에 선택할 수 있습니다.

### 1.2 설계 원칙

- **전략 패턴**: 학습 알고리즘을 독립적으로 캡슐화
- **독립성**: 모델과 데이터로더에 독립적
- **확장성**: 새로운 학습 전략 추가 시 기존 코드 수정 불필요
- **하드코딩 지양**: 모든 설정은 파라미터화

### 1.3 파일 구조

```
research/strategies/training/
├── __init__.py              # Strategy exports
├── base.py                  # TrainingStrategy 추상 클래스
└── vanilla_strategy.py      # VanillaTrainingStrategy 구현
```

---

## 2. TrainingStrategy 인터페이스

### 2.1 추상 클래스

```python
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

class TrainingStrategy(ABC):
    """학습 전략의 기본 인터페이스"""

    @abstractmethod
    def train(self,
              model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              config: Dict[str, Any]) -> Dict[str, Any]:
        """모델 학습 실행

        Args:
            model: 학습할 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            config: 학습 설정 (max_epochs, learning_rate, optimizer 등)

        Returns:
            학습 결과 딕셔너리 (history, best_metric, training_time 등)
        """
        pass

    @abstractmethod
    def evaluate(self,
                 model: nn.Module,
                 test_loader: DataLoader) -> Dict[str, Any]:
        """모델 평가

        Args:
            model: 평가할 모델
            test_loader: 테스트 데이터로더

        Returns:
            평가 결과 딕셔너리 (test_loss, test_metric, inference_time 등)
        """
        pass
```

---

## 3. VanillaTrainingStrategy

### 3.1 개요

순수 PyTorch를 사용한 기본 학습 전략입니다. 외부 라이브러리 없이 표준 학습 루프를 구현합니다.

### 3.2 클래스 구조

```python
class VanillaTrainingStrategy(TrainingStrategy):
    """순수 PyTorch 학습 전략"""

    # 클래스 상수
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_OPTIMIZER = 'adam'
    LOG_INTERVAL = 50  # 배치 단위 로그 출력 간격

    def __init__(self, device: str = None):
        """초기화

        Args:
            device: 학습 디바이스 ('cuda', 'cpu', None=자동)
        """
        self.device = self._get_device(device)
```

### 3.3 주요 메서드

#### train() 메서드

```python
def train(self, model, train_loader, val_loader, config):
    """모델 학습 실행

    프로세스:
    1. 설정 추출 (max_epochs, learning_rate, optimizer)
    2. Optimizer 생성 (Adam/AdamW/SGD)
    3. 학습 루프:
       - Forward pass
       - Loss 계산
       - Backward pass
       - Optimizer step
    4. 에폭별 검증
    5. 최고 성능 추적

    Returns:
        {
            'training_time': float,  # 전체 학습 시간 (초)
            'best_val_acc': float,    # 최고 검증 정확도
            'history': {
                'train_loss': List[float],
                'train_acc': List[float],
                'val_loss': List[float],
                'val_acc': List[float]
            },
            'model': nn.Module  # 학습된 모델
        }
    """
```

#### evaluate() 메서드

```python
def evaluate(self, model, test_loader):
    """모델 평가

    프로세스:
    1. model.eval() 모드 설정
    2. No gradient 컨텍스트
    3. 배치별 예측 수행
    4. 손실 및 정확도 계산
    5. 추론 시간 측정 (CUDA 동기화 포함)

    Returns:
        {
            'test_loss': float,       # 평균 테스트 손실
            'test_acc': float,        # 테스트 정확도
            'inference_time': float,  # 추론 시간 (ms)
            'predictions': Tensor,    # 예측 결과
            'labels': Tensor         # 실제 레이블
        }
    """
```

### 3.4 디바이스 처리

```python
def _get_device(self, device: Optional[str]) -> torch.device:
    """디바이스 자동 감지

    우선순위:
    1. 명시적 지정 디바이스
    2. CUDA 가용시 cuda:0
    3. 기본값 cpu
    """
    if device:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 4. 현재 구현 상태

### 4.1 구현 완료 기능

- ✅ 기본 학습 루프
- ✅ Adam/AdamW/SGD optimizer 지원
- ✅ 디바이스 자동 감지
- ✅ 학습/검증 손실 추적
- ✅ 추론 시간 측정

### 4.2 문제점 및 한계

#### **중대한 설계 결함**

1. **TaskStrategy 통합 누락**
   - Line 59: `criterion = nn.CrossEntropyLoss()` 하드코딩
   - TaskStrategy를 완전히 무시하여 전체 추상화 파괴
   - 회귀나 이진 분류 태스크 사용 불가

2. **메트릭 하드코딩**
   - 정확도만 계산 (Lines 97-99)
   - 다른 메트릭 사용 불가능

#### **성능 이슈**

1. **과도한 I/O**
   - Line 102: 50배치마다 print() 호출
   - 로깅 전략 부재

2. **GPU 동기화 문제**
   - 학습 시간 측정시 GPU 동기화 누락
   - 부정확한 타이밍 측정

3. **CPU-GPU 전송 비효율**
   - 메트릭 계산을 위한 불필요한 CPU 전송

#### **누락된 현대적 기능**

- ❌ Gradient clipping
- ❌ Learning rate scheduling
- ❌ Early stopping
- ❌ Mixed precision training
- ❌ Gradient accumulation
- ❌ Checkpoint 저장/로드

---

## 5. API 명세

### 5.1 사용 예제

```python
from research.strategies.training import VanillaTrainingStrategy

# 전략 생성
strategy = VanillaTrainingStrategy(device='cuda')

# 설정
config = {
    'max_epochs': 20,
    'learning_rate': 1e-4,
    'optimizer': 'adam'  # 'adam', 'adamw', 'sgd'
}

# 학습 실행
results = strategy.train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# 평가
test_results = strategy.evaluate(model, test_loader)
```

### 5.2 설정 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| max_epochs | int | 100 | 최대 학습 에폭 |
| learning_rate | float | 1e-4 | 학습률 |
| optimizer | str | 'adam' | optimizer 종류 |
| momentum | float | 0.9 | SGD momentum (SGD용) |

### 5.3 반환값 구조

#### train() 반환값

```python
{
    'training_time': 120.5,      # 초 단위
    'best_val_acc': 0.945,       # 최고 검증 정확도
    'history': {
        'train_loss': [2.3, 1.8, ...],  # 에폭별 학습 손실
        'train_acc': [0.65, 0.78, ...], # 에폭별 학습 정확도
        'val_loss': [2.1, 1.6, ...],    # 에폭별 검증 손실
        'val_acc': [0.70, 0.82, ...]    # 에폭별 검증 정확도
    },
    'model': model  # 학습된 모델 인스턴스
}
```

#### evaluate() 반환값

```python
{
    'test_loss': 0.453,         # 평균 테스트 손실
    'test_acc': 0.923,          # 테스트 정확도
    'inference_time': 8.5,      # 추론 시간 (ms)
    'predictions': tensor(...),  # [N, num_classes] 예측
    'labels': tensor(...)        # [N] 실제 레이블
}
```

---

## 6. 개선 필요사항

### 6.1 긴급 수정 필요 (P0)

1. **TaskStrategy 통합**
   ```python
   def __init__(self, device=None, task_strategy=None):
       self.task_strategy = task_strategy

   def train(self, ...):
       criterion = self.task_strategy.get_criterion()
       # 메트릭도 task_strategy에서 가져오기
   ```

2. **LoggingStrategy 통합**
   - print 대신 로깅 전략 사용
   - 설정 가능한 로그 레벨

### 6.2 성능 개선 (P1)

1. **GPU 동기화**
   ```python
   if self.device.type == 'cuda':
       torch.cuda.synchronize()
   epoch_time = time.time() - start_time
   ```

2. **메트릭 GPU 계산**
   - CPU 전송 최소화
   - 배치 단위 누적

### 6.3 기능 추가 (P2)

1. **Learning Rate Scheduling**
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='max', patience=5
   )
   ```

2. **Early Stopping**
   ```python
   if val_metric > best_metric:
       best_metric = val_metric
       patience_counter = 0
   else:
       patience_counter += 1
       if patience_counter >= patience:
           break
   ```

3. **Gradient Clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(
       model.parameters(), max_norm=1.0
   )
   ```

---

## 7. 테스트 요구사항

### 7.1 단위 테스트

```python
def test_vanilla_training_strategy():
    """VanillaTrainingStrategy 단위 테스트"""

    # 1. 초기화 테스트
    strategy = VanillaTrainingStrategy()
    assert strategy.device.type in ['cuda', 'cpu']

    # 2. Optimizer 생성 테스트
    for opt in ['adam', 'adamw', 'sgd']:
        config = {'optimizer': opt}
        # optimizer 생성 확인

    # 3. 디바이스 이동 테스트
    model = DummyModel()
    strategy._move_to_device(model)
    assert next(model.parameters()).device == strategy.device
```

### 7.2 통합 테스트

```python
def test_full_training_cycle():
    """전체 학습 사이클 테스트"""

    # 1. 데이터 준비
    train_loader = create_dummy_loader(100)
    val_loader = create_dummy_loader(20)
    test_loader = create_dummy_loader(30)

    # 2. 학습 실행
    strategy = VanillaTrainingStrategy()
    results = strategy.train(model, train_loader, val_loader, config)

    # 3. 검증
    assert 'history' in results
    assert len(results['history']['train_loss']) == config['max_epochs']
    assert results['best_val_acc'] > 0

    # 4. 평가
    test_results = strategy.evaluate(model, test_loader)
    assert test_results['test_acc'] > 0
    assert test_results['inference_time'] > 0
```

### 7.3 성능 테스트

```python
def test_training_performance():
    """학습 성능 벤치마크"""

    # GPU 메모리 사용량 측정
    # 학습 시간 측정
    # 배치 처리 속도 측정

    assert memory_usage < MAX_MEMORY_MB
    assert batch_time < MAX_BATCH_TIME_MS
```

### 7.4 엣지 케이스

- 빈 데이터로더 처리
- 1-에폭 학습
- 큰 배치 크기 (OOM 처리)
- NaN/Inf 손실 처리

---

## 부록: SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ⚠️ | 학습과 평가 담당, 하지만 메트릭 계산도 포함 |
| **O**pen/Closed | ❌ | CrossEntropyLoss 하드코딩으로 확장 불가 |
| **L**iskov Substitution | ✅ | TrainingStrategy 인터페이스 준수 |
| **I**nterface Segregation | ✅ | 최소 인터페이스 정의 |
| **D**ependency Inversion | ❌ | 구체적 손실 함수에 의존 |

현재 구현은 OCP와 DIP를 위반하고 있어 TaskStrategy 통합이 시급합니다.
# Task 전략 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [TaskStrategy 인터페이스](#2-taskstrategy-인터페이스)
3. [다중 분류 전략](#3-다중-분류-전략)
4. [이진 분류 전략](#4-이진-분류-전략)
5. [회귀 전략](#5-회귀-전략)
6. [API 명세](#6-api-명세)
7. [테스트 요구사항](#7-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Task 전략 시스템은 다양한 머신러닝 태스크(다중 분류, 이진 분류, 회귀)에 대한 통합 인터페이스를 제공합니다. Strategy Pattern을 사용하여 태스크별로 적절한 손실 함수, 활성화 함수, 메트릭을 자동으로 선택합니다.

### 1.2 설계 원칙

- **전략 패턴**: 태스크별 동작을 캡슐화
- **개방-폐쇄 원칙**: 새로운 태스크 추가 시 기존 코드 수정 불필요
- **하드코딩 지양**: 모든 상수는 클래스 상수로 정의

### 1.3 파일 구조

```
research/strategies/task/
├── __init__.py              # Strategy exports
├── base.py                  # TaskStrategy 추상 클래스 (이미 존재)
└── task_strategies.py       # 구체적 전략 구현 (구현 필요)
```

---

## 2. TaskStrategy 인터페이스

### 2.1 추상 클래스

이미 `research/strategies/task/base.py`에 구현되어 있음.

```python
class TaskStrategy(ABC):
    """Task strategy base class"""

    @abstractmethod
    def get_criterion(self):
        """Get loss function"""
        pass

    @abstractmethod
    def get_activation(self):
        """Get activation function"""
        pass

    @abstractmethod
    def calculate_metric(self, outputs, labels):
        """Calculate primary metric"""
        pass

    @abstractmethod
    def prepare_labels(self, labels, num_classes=None):
        """Prepare labels for the task"""
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        """Get task type name"""
        pass
```

---

## 3. 다중 분류 전략

### 3.1 MultiClassStrategy

**용도**: N개 클래스 중 하나를 선택하는 문제 (N ≥ 2)

**특징**:
- 손실 함수: CrossEntropyLoss
- 활성화 함수: Softmax (내장)
- 기본 메트릭: Accuracy
- 레이블 형식: Long tensor (class indices)

### 3.2 인터페이스

```python
class MultiClassStrategy(TaskStrategy):
    """Multi-class classification strategy"""

    TASK_TYPE = "multiclass"
    DEFAULT_NUM_CLASSES = 10

    def __init__(self, num_classes: int = None):
        """Initialize strategy

        Args:
            num_classes: Number of classes (required)
        """
        self.num_classes = num_classes or self.DEFAULT_NUM_CLASSES

    def get_criterion(self):
        """Get CrossEntropyLoss"""
        return nn.CrossEntropyLoss()

    def get_activation(self):
        """Get Softmax activation (dim=1)"""
        return nn.Softmax(dim=1)

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy

        Args:
            outputs: Model logits (batch_size, num_classes)
            labels: True labels (batch_size,)

        Returns:
            float: Accuracy (0.0 ~ 1.0)
        """
        pass

    def prepare_labels(self, labels: torch.Tensor, num_classes: int = None) -> torch.Tensor:
        """Prepare labels for multi-class classification

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for multi-class)

        Returns:
            torch.Tensor: Labels as Long tensor
        """
        pass

    def get_task_type(self) -> str:
        return self.TASK_TYPE
```

---

## 4. 이진 분류 전략

### 4.1 BinaryClassificationStrategy

**용도**: 0 또는 1을 선택하는 문제

**특징**:
- 손실 함수: BCEWithLogitsLoss
- 활성화 함수: Sigmoid
- 기본 메트릭: Accuracy
- 레이블 형식: Float tensor (0.0 or 1.0)

### 4.2 인터페이스

```python
class BinaryClassificationStrategy(TaskStrategy):
    """Binary classification strategy"""

    TASK_TYPE = "binary"
    THRESHOLD = 0.5  # Classification threshold

    def get_criterion(self):
        """Get BCEWithLogitsLoss"""
        return nn.BCEWithLogitsLoss()

    def get_activation(self):
        """Get Sigmoid activation"""
        return nn.Sigmoid()

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy

        Args:
            outputs: Model logits (batch_size, 1) or (batch_size,)
            labels: True labels (batch_size,) - 0 or 1

        Returns:
            float: Accuracy (0.0 ~ 1.0)
        """
        pass

    def prepare_labels(self, labels: torch.Tensor, num_classes: int = None) -> torch.Tensor:
        """Prepare labels for binary classification

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for binary)

        Returns:
            torch.Tensor: Labels as Float tensor
        """
        pass

    def get_task_type(self) -> str:
        return self.TASK_TYPE
```

---

## 5. 회귀 전략

### 5.1 RegressionStrategy

**용도**: 연속적인 값을 예측하는 문제

**특징**:
- 손실 함수: MSELoss
- 활성화 함수: None (선형 출력)
- 기본 메트릭: MSE
- 레이블 형식: Float tensor

### 5.2 인터페이스

```python
class RegressionStrategy(TaskStrategy):
    """Regression strategy"""

    TASK_TYPE = "regression"

    def get_criterion(self):
        """Get MSELoss"""
        return nn.MSELoss()

    def get_activation(self):
        """Get None (linear output)"""
        return None

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate MSE

        Args:
            outputs: Model predictions (batch_size, 1) or (batch_size,)
            labels: True values (batch_size,) or (batch_size, 1)

        Returns:
            float: MSE value
        """
        pass

    def prepare_labels(self, labels: torch.Tensor, num_classes: int = None) -> torch.Tensor:
        """Prepare labels for regression

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for regression)

        Returns:
            torch.Tensor: Labels as Float tensor
        """
        pass

    def get_task_type(self) -> str:
        return self.TASK_TYPE
```

---

## 6. API 명세

### 6.1 Import

```python
from research.strategies.task import (
    TaskStrategy,
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy
)
```

### 6.2 사용 예제

```python
# 다중 분류
strategy = MultiClassStrategy(num_classes=10)
criterion = strategy.get_criterion()
outputs = model(inputs)
loss = criterion(outputs, labels)
accuracy = strategy.calculate_metric(outputs, labels)

# 이진 분류
strategy = BinaryClassificationStrategy()
criterion = strategy.get_criterion()
outputs = model(inputs)
loss = criterion(outputs, labels)
accuracy = strategy.calculate_metric(outputs, labels)

# 회귀
strategy = RegressionStrategy()
criterion = strategy.get_criterion()
outputs = model(inputs)
loss = criterion(outputs, labels)
mse = strategy.calculate_metric(outputs, labels)
```

---

## 7. 테스트 요구사항

### 7.1 테스트 원칙

모든 테스트는 **Given-When-Then** 패턴을 따릅니다.

### 7.2 MultiClassStrategy 테스트

```python
def test_multiclass_get_criterion():
    """Given: MultiClassStrategy
    When: get_criterion() 호출
    Then: CrossEntropyLoss 반환"""

def test_multiclass_get_activation():
    """Given: MultiClassStrategy
    When: get_activation() 호출
    Then: Softmax 반환"""

def test_multiclass_calculate_metric_perfect():
    """Given: 완벽한 예측
    When: calculate_metric() 호출
    Then: 1.0 반환"""

def test_multiclass_prepare_labels():
    """Given: 레이블 텐서
    When: prepare_labels() 호출
    Then: Long tensor 반환"""

def test_multiclass_get_task_type():
    """Given: MultiClassStrategy
    When: get_task_type() 호출
    Then: 'multiclass' 반환"""
```

### 7.3 BinaryClassificationStrategy 테스트

```python
def test_binary_get_criterion():
    """Given: BinaryClassificationStrategy
    When: get_criterion() 호출
    Then: BCEWithLogitsLoss 반환"""

def test_binary_get_activation():
    """Given: BinaryClassificationStrategy
    When: get_activation() 호출
    Then: Sigmoid 반환"""

def test_binary_calculate_metric_perfect():
    """Given: 완벽한 예측
    When: calculate_metric() 호출
    Then: 1.0 반환"""

def test_binary_prepare_labels():
    """Given: 레이블 텐서
    When: prepare_labels() 호출
    Then: Float tensor 반환"""

def test_binary_threshold():
    """Given: BinaryClassificationStrategy
    When: 임계값 확인
    Then: 0.5"""
```

### 7.4 RegressionStrategy 테스트

```python
def test_regression_get_criterion():
    """Given: RegressionStrategy
    When: get_criterion() 호출
    Then: MSELoss 반환"""

def test_regression_get_activation():
    """Given: RegressionStrategy
    When: get_activation() 호출
    Then: None 반환"""

def test_regression_calculate_metric_perfect():
    """Given: 완벽한 예측
    When: calculate_metric() 호출
    Then: 0.0 반환 (MSE)"""

def test_regression_prepare_labels():
    """Given: 레이블 텐서
    When: prepare_labels() 호출
    Then: Float tensor 반환"""
```

### 7.5 전략 교체 가능성 테스트

```python
def test_strategies_are_interchangeable():
    """Given: 다양한 전략들
    When: TaskStrategy 인터페이스로 사용
    Then: 모두 동일한 메서드 제공"""

def test_strategies_polymorphism():
    """Given: 전략 리스트
    When: 각 전략의 메서드 호출
    Then: 다형성 동작 확인"""
```

---

## 8. 구현 체크리스트

### Phase 2-A: MultiClassStrategy
- [ ] get_criterion() 구현
- [ ] get_activation() 구현
- [ ] calculate_metric() 구현
- [ ] prepare_labels() 구현
- [ ] get_task_type() 구현
- [ ] 테스트 5개 이상 통과

### Phase 2-B: BinaryClassificationStrategy
- [ ] get_criterion() 구현
- [ ] get_activation() 구현
- [ ] calculate_metric() 구현
- [ ] prepare_labels() 구현
- [ ] get_task_type() 구현
- [ ] 테스트 5개 이상 통과

### Phase 2-C: RegressionStrategy
- [ ] get_criterion() 구현
- [ ] get_activation() 구현
- [ ] calculate_metric() 구현
- [ ] prepare_labels() 구현
- [ ] get_task_type() 구현
- [ ] 테스트 5개 이상 통과

### Phase 2-D: 통합 테스트
- [ ] 전략 교체 가능성 테스트
- [ ] 다형성 테스트
- [ ] 모든 테스트 통과

---

## 9. 성공 기준

### 9.1 기능적 요구사항
- 모든 전략이 TaskStrategy 인터페이스 준수
- 적절한 손실 함수 및 활성화 함수 반환
- 메트릭 계산 정확성

### 9.2 비기능적 요구사항
- 하드코딩 없음 (클래스 상수 사용)
- 테스트 커버리지 90% 이상
- Given-When-Then 패턴 준수

---

**문서 버전**: 1.0.0
**작성일**: 2025-11-03
**이전 문서**: 01_metrics_spec.md
**다음 문서**: 03_models_spec.md (예정)

# 메트릭 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [메트릭 인터페이스](#2-메트릭-인터페이스)
3. [분류 메트릭](#3-분류-메트릭)
4. [회귀 메트릭](#4-회귀-메트릭)
5. [MetricTracker](#5-metrictracker)
6. [API 명세](#6-api-명세)
7. [구현 가이드](#7-구현-가이드)
8. [테스트 요구사항](#8-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

메트릭 시스템은 모델 성능을 정량적으로 평가하기 위한 독립적인 모듈입니다. 분류 및 회귀 태스크를 위한 다양한 메트릭을 제공하며, 다중 메트릭 동시 추적 기능을 지원합니다.

### 1.2 설계 원칙

- **독립성**: 다른 모듈에 의존하지 않음 (PyTorch만 필요)
- **확장성**: 새로운 메트릭을 쉽게 추가 가능
- **일관성**: 모든 메트릭이 동일한 인터페이스 준수
- **하드코딩 지양**: 모든 상수는 클래스 상수 또는 설정 파라미터로 정의

### 1.3 파일 구조

```
research/metrics/
├── base.py              # BaseMetric 추상 클래스 (이미 구현)
├── classification.py    # 분류 메트릭 6종 (구현 필요)
├── regression.py        # 회귀 메트릭 3종 (구현 필요)
├── tracker.py          # MetricTracker (구현 필요)
└── __init__.py         # 통합 export
```

---

## 2. 메트릭 인터페이스

### 2.1 BaseMetric 추상 클래스

이미 `research/metrics/base.py`에 구현되어 있음.

```python
from abc import ABC, abstractmethod
import torch

class BaseMetric(ABC):
    """모든 메트릭의 기본 인터페이스"""

    @abstractmethod
    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """메트릭 계산"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """메트릭 이름 반환"""
        pass

    @abstractmethod
    def is_higher_better(self) -> bool:
        """높을수록 좋은 메트릭인지 여부"""
        pass
```

### 2.2 구현 요구사항

모든 메트릭 클래스는:
1. `BaseMetric`을 상속
2. 3개의 추상 메서드 구현
3. 메트릭 이름은 클래스 상수로 정의
4. 매직 넘버 사용 금지

---

## 3. 분류 메트릭

### 3.1 AccuracyMetric

**설명**: 전체 샘플 중 올바르게 예측한 비율

**공식**:
```
Accuracy = (올바른 예측 수) / (전체 샘플 수)
```

**인터페이스**:
```python
class AccuracyMetric(BaseMetric):
    METRIC_NAME = "accuracy"
    HIGHER_IS_BETTER = True

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 모델 예측 (logits 또는 class indices)
            targets: 실제 레이블 (class indices)

        Returns:
            float: 정확도 (0.0 ~ 1.0)
        """
        pass
```

**특징**:
- 다중 클래스 분류에 사용
- predictions가 2D인 경우 argmax 적용
- 범위: [0.0, 1.0]

---

### 3.2 PrecisionMetric

**설명**: 양성으로 예측한 것 중 실제 양성인 비율

**공식**:
```
Precision = TP / (TP + FP)
```

**인터페이스**:
```python
class PrecisionMetric(BaseMetric):
    METRIC_NAME = "precision"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted"]

    def __init__(self, average: str = None, num_classes: int = None):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
            num_classes: 클래스 수 (필수)
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(f"average must be one of {self.VALID_AVERAGES}")

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        pass
```

**평균 방식**:
- **macro**: 각 클래스별 정밀도의 산술 평균
- **micro**: 전체 샘플 기준으로 계산
- **weighted**: 각 클래스 샘플 수로 가중 평균

---

### 3.3 RecallMetric

**설명**: 실제 양성 중 올바르게 예측한 비율

**공식**:
```
Recall = TP / (TP + FN)
```

**인터페이스**:
```python
class RecallMetric(BaseMetric):
    METRIC_NAME = "recall"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted"]

    def __init__(self, average: str = None, num_classes: int = None):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
            num_classes: 클래스 수 (필수)
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(f"average must be one of {self.VALID_AVERAGES}")

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        pass
```

---

### 3.4 F1ScoreMetric

**설명**: Precision과 Recall의 조화 평균

**공식**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**인터페이스**:
```python
class F1ScoreMetric(BaseMetric):
    METRIC_NAME = "f1_score"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted"]

    def __init__(self, average: str = None, num_classes: int = None):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
            num_classes: 클래스 수 (필수)
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes
        self.precision_metric = PrecisionMetric(average, num_classes)
        self.recall_metric = RecallMetric(average, num_classes)

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(f"average must be one of {self.VALID_AVERAGES}")

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Precision과 Recall을 재사용하여 계산"""
        pass
```

---

### 3.5 Top5AccuracyMetric

**설명**: 상위 5개 예측 중 정답이 포함된 비율

**공식**:
```
Top5Accuracy = (상위 5개 중 정답 포함된 샘플 수) / (전체 샘플 수)
```

**인터페이스**:
```python
class Top5AccuracyMetric(BaseMetric):
    METRIC_NAME = "top5_accuracy"
    HIGHER_IS_BETTER = True
    TOP_K = 5

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 모델 logits (batch_size, num_classes)
            targets: 실제 레이블 (batch_size,)

        Returns:
            float: Top-5 정확도 (0.0 ~ 1.0)
        """
        pass
```

**특징**:
- predictions는 반드시 2D (logits)
- 클래스 수가 5개 미만일 경우 일반 accuracy와 동일

---

### 3.6 AUCMetric

**설명**: ROC 곡선 아래 면적 (이진 분류용)

**인터페이스**:
```python
class AUCMetric(BaseMetric):
    METRIC_NAME = "auc"
    HIGHER_IS_BETTER = True

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 확률 예측값 (0.0 ~ 1.0)
            targets: 실제 레이블 (0 또는 1)

        Returns:
            float: AUC 점수 (0.0 ~ 1.0)
        """
        pass
```

**특징**:
- sklearn.metrics.roc_auc_score 사용
- 이진 분류 전용

---

## 4. 회귀 메트릭

### 4.1 MSEMetric

**설명**: 평균 제곱 오차 (Mean Squared Error)

**공식**:
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**인터페이스**:
```python
class MSEMetric(BaseMetric):
    METRIC_NAME = "mse"
    HIGHER_IS_BETTER = False

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 예측값
            targets: 실제 값

        Returns:
            float: MSE 값 (0.0 이상)
        """
        pass
```

**특징**:
- 낮을수록 좋은 메트릭
- 이상치에 민감
- 범위: [0, ∞)

---

### 4.2 MAEMetric

**설명**: 평균 절대 오차 (Mean Absolute Error)

**공식**:
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**인터페이스**:
```python
class MAEMetric(BaseMetric):
    METRIC_NAME = "mae"
    HIGHER_IS_BETTER = False

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 예측값
            targets: 실제 값

        Returns:
            float: MAE 값 (0.0 이상)
        """
        pass
```

**특징**:
- 낮을수록 좋은 메트릭
- MSE보다 이상치에 덜 민감
- 범위: [0, ∞)

---

### 4.3 R2Metric

**설명**: 결정 계수 (Coefficient of Determination)

**공식**:
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - mean(y_true))²
```

**인터페이스**:
```python
class R2Metric(BaseMetric):
    METRIC_NAME = "r2_score"
    HIGHER_IS_BETTER = True
    PERFECT_SCORE = 1.0

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            predictions: 예측값
            targets: 실제 값

        Returns:
            float: R² 값
        """
        pass
```

**특징**:
- 높을수록 좋은 메트릭
- 완벽한 예측 시 1.0
- 범위: (-∞, 1]

---

## 5. MetricTracker

### 5.1 개요

다중 메트릭을 동시에 추적하고 히스토리를 관리하는 클래스입니다.

### 5.2 인터페이스

```python
class MetricTracker:
    """다중 메트릭 추적 및 히스토리 관리"""

    DEFAULT_WINDOW_SIZE = 10

    def __init__(self, metric_names: List[str], window_size: int = None):
        """
        Args:
            metric_names: 추적할 메트릭 이름 리스트
            window_size: 이동 평균 윈도우 크기 (기본값: 10)
        """
        self.metric_names = metric_names
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self.history = {name: [] for name in metric_names}

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               metrics: Dict[str, BaseMetric]) -> Dict[str, float]:
        """
        메트릭 계산 및 히스토리 업데이트

        Args:
            predictions: 모델 예측
            targets: 실제 레이블/값
            metrics: 메트릭 객체 딕셔너리 {name: metric_instance}

        Returns:
            Dict[str, float]: 계산된 메트릭 값들
        """
        pass

    def get_latest(self, metric_name: str = None) -> Union[float, Dict[str, float]]:
        """
        최신 메트릭 값 조회

        Args:
            metric_name: 특정 메트릭 이름 (None이면 전체)

        Returns:
            float 또는 Dict[str, float]: 최신 메트릭 값
        """
        pass

    def get_best(self, metric_name: str, higher_is_better: bool = True) -> float:
        """
        최고 성능 메트릭 값 조회

        Args:
            metric_name: 메트릭 이름
            higher_is_better: 높을수록 좋은지 여부

        Returns:
            float: 최고 메트릭 값
        """
        pass

    def get_history(self, metric_name: str = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        메트릭 히스토리 조회

        Args:
            metric_name: 특정 메트릭 이름 (None이면 전체)

        Returns:
            List[float] 또는 Dict: 메트릭 히스토리
        """
        pass

    def get_moving_average(self, metric_name: str) -> float:
        """
        이동 평균 계산

        Args:
            metric_name: 메트릭 이름

        Returns:
            float: 이동 평균 값
        """
        pass

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        전체 메트릭 요약 통계

        Returns:
            Dict: {metric_name: {mean, std, min, max, latest}}
        """
        pass

    def reset(self, metric_name: str = None):
        """
        메트릭 히스토리 초기화

        Args:
            metric_name: 특정 메트릭 이름 (None이면 전체)
        """
        pass
```

### 5.3 사용 예제

```python
# Given: 메트릭 추적기 생성
tracker = MetricTracker(['accuracy', 'precision', 'recall'])

# When: 메트릭 업데이트
metrics = {
    'accuracy': AccuracyMetric(),
    'precision': PrecisionMetric(num_classes=10),
    'recall': RecallMetric(num_classes=10)
}
results = tracker.update(predictions, targets, metrics)

# Then: 결과 조회
latest = tracker.get_latest('accuracy')
best = tracker.get_best('accuracy')
history = tracker.get_history('accuracy')
summary = tracker.summary()
```

---

## 6. API 명세

### 6.1 메트릭 Import

```python
from research.metrics import (
    # 베이스 클래스
    BaseMetric,

    # 분류 메트릭
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    Top5AccuracyMetric,
    AUCMetric,

    # 회귀 메트릭
    MSEMetric,
    MAEMetric,
    R2Metric,

    # 추적기
    MetricTracker
)
```

### 6.2 기본 사용법

```python
# 분류 메트릭 사용
accuracy = AccuracyMetric()
score = accuracy.calculate(predictions, targets)

# 회귀 메트릭 사용
mse = MSEMetric()
error = mse.calculate(predictions, targets)

# 다중 메트릭 추적
tracker = MetricTracker(['accuracy', 'f1_score'])
```

---

## 7. 구현 가이드

### 7.1 코딩 원칙

#### 하드코딩 지양

```python
# Bad: 매직 넘버
def calculate(self, predictions, targets):
    return (predictions == targets).float().mean().item() * 100  # 100은?

# Good: 클래스 상수
class AccuracyMetric(BaseMetric):
    PERCENTAGE_SCALE = 100.0
    METRIC_NAME = "accuracy"

    def calculate(self, predictions, targets):
        accuracy = (predictions == targets).float().mean().item()
        return accuracy  # 0.0~1.0 반환 (스케일링 불필요)
```

#### 설정 가능한 파라미터

```python
# Bad: 하드코딩된 평균 방식
class PrecisionMetric(BaseMetric):
    def calculate(self, predictions, targets):
        # macro 평균 하드코딩
        pass

# Good: 파라미터로 설정 가능
class PrecisionMetric(BaseMetric):
    DEFAULT_AVERAGE = "macro"

    def __init__(self, average: str = None, num_classes: int = None):
        self.average = average or self.DEFAULT_AVERAGE
```

### 7.2 커스텀 메트릭 추가

```python
from research.metrics import BaseMetric

class CustomMetric(BaseMetric):
    METRIC_NAME = "custom_metric"
    HIGHER_IS_BETTER = True

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        # 커스텀 계산 로직
        result = your_calculation(predictions, targets)
        return float(result)

    def get_name(self) -> str:
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        return self.HIGHER_IS_BETTER
```

---

## 8. 테스트 요구사항

### 8.1 테스트 원칙

모든 테스트는 **Given-When-Then** 패턴을 따릅니다.

```python
def test_accuracy_metric_perfect_prediction():
    # Given: 완벽한 예측 데이터
    predictions = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 4])
    metric = AccuracyMetric()

    # When: 메트릭 계산
    result = metric.calculate(predictions, targets)

    # Then: 100% 정확도 검증
    assert result == 1.0
```

### 8.2 필수 테스트 케이스

#### AccuracyMetric
- 완벽한 예측 (100% 정확도)
- 완전히 틀린 예측 (0% 정확도)
- 2D logits 입력 처리

#### PrecisionMetric, RecallMetric, F1ScoreMetric
- macro/micro/weighted 평균 방식
- 불균형 데이터셋
- 일부 클래스가 예측되지 않는 경우

#### Top5AccuracyMetric
- 상위 5개 내 정답 포함
- 클래스 수가 5개 미만인 경우

#### 회귀 메트릭 (MSE, MAE, R2)
- 완벽한 예측 (MSE=0, MAE=0, R²=1)
- 선형 관계 데이터
- 음수 값 처리

#### MetricTracker
- 다중 메트릭 동시 업데이트
- 히스토리 추적
- 최고/최신 값 조회
- 이동 평균 계산
- 초기화

### 8.3 테스트 파일 구조

```
tests/
├── conftest.py                  # pytest fixtures
└── unit/
    └── test_metrics.py          # 메트릭 테스트
```

### 8.4 필수 Fixtures

```python
# conftest.py

import pytest
import torch

# 테스트 상수
PERFECT_ACCURACY = 1.0
ZERO_ACCURACY = 0.0
NUM_SAMPLES = 32
NUM_CLASSES = 10

@pytest.fixture
def perfect_predictions():
    """완벽한 예측 데이터"""
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    predictions = targets.clone()
    return predictions, targets

@pytest.fixture
def random_predictions():
    """랜덤 예측 데이터"""
    predictions = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return predictions, targets
```

---

## 9. 구현 체크리스트

### Phase 1-A: 분류 메트릭 (classification.py)
- [ ] AccuracyMetric 구현
- [ ] PrecisionMetric 구현
- [ ] RecallMetric 구현
- [ ] F1ScoreMetric 구현
- [ ] Top5AccuracyMetric 구현
- [ ] AUCMetric 구현

### Phase 1-B: 회귀 메트릭 (regression.py)
- [ ] MSEMetric 구현
- [ ] MAEMetric 구현
- [ ] R2Metric 구현

### Phase 1-C: 메트릭 추적기 (tracker.py)
- [ ] MetricTracker 구현
- [ ] update() 메서드
- [ ] get_latest() 메서드
- [ ] get_best() 메서드
- [ ] get_history() 메서드
- [ ] get_moving_average() 메서드
- [ ] summary() 메서드
- [ ] reset() 메서드

### Phase 1-D: 테스트
- [ ] 모든 테스트 케이스 작성 (Given-When-Then)
- [ ] pytest 실행하여 통과 확인
- [ ] 코드 커버리지 90% 이상 확인

---

## 10. 성공 기준

### 10.1 기능적 요구사항
- 모든 메트릭이 BaseMetric 인터페이스 준수
- 예상된 입력에 대해 올바른 값 반환
- MetricTracker가 다중 메트릭 동시 추적

### 10.2 비기능적 요구사항
- 하드코딩 없음 (모든 상수는 클래스 상수로 정의)
- 테스트 커버리지 90% 이상
- 모든 테스트가 Given-When-Then 패턴 준수
- 타입 힌트 완전 적용

### 10.3 검증 방법
```bash
# 테스트 실행
pytest tests/unit/test_metrics.py -v

# 커버리지 확인
pytest tests/unit/test_metrics.py --cov=research/metrics --cov-report=term-missing

# 타입 체크
mypy research/metrics/
```

---

**문서 버전**: 1.0.0
**작성일**: 2025-11-03
**다음 문서**: 02_task_strategies_spec.md

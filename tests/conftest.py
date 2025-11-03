"""pytest configuration and shared fixtures for all tests"""

import pytest
import torch
import numpy as np
from typing import Tuple


# ============================================================================
# Test Constants - 모든 테스트에서 사용하는 상수 정의
# ============================================================================

# 정확도 관련 상수
PERFECT_ACCURACY = 1.0
ZERO_ACCURACY = 0.0
HALF_ACCURACY = 0.5

# 샘플 및 클래스 수
NUM_SAMPLES = 32
NUM_CLASSES = 10
BINARY_CLASSES = 2
SMALL_NUM_CLASSES = 3

# 회귀 관련 상수
PERFECT_MSE = 0.0
PERFECT_MAE = 0.0
PERFECT_R2 = 1.0

# MetricTracker 관련 상수
DEFAULT_WINDOW_SIZE = 10
SMALL_WINDOW_SIZE = 3

# Top-K 상수
TOP_K = 5

# Tolerance for float comparison
FLOAT_TOLERANCE = 1e-6


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """테스트에 사용할 디바이스 반환"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Classification Data Fixtures
# ============================================================================

@pytest.fixture
def perfect_predictions():
    """완벽한 예측 데이터 (100% 정확도)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    # Given: 타겟 생성 후 동일한 예측 생성
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    predictions = targets.clone()
    return predictions, targets


@pytest.fixture
def zero_predictions():
    """완전히 틀린 예측 데이터 (0% 정확도)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    # Given: 타겟과 완전히 다른 클래스로 예측
    # 타겟이 0이면 예측은 1, 타겟이 1이면 예측은 0 등
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    predictions = (targets + 1) % NUM_CLASSES
    return predictions, targets


@pytest.fixture
def random_predictions():
    """랜덤 예측 데이터

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    predictions = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    return predictions, targets


@pytest.fixture
def perfect_logits():
    """완벽한 예측 logits (2D)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (logits, targets)
    """
    # Given: 각 샘플의 정답 클래스에만 높은 logit
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)

    # 정답 클래스의 logit을 크게 설정
    for i, target in enumerate(targets):
        logits[i] = -10.0  # 모든 클래스를 낮게
        logits[i, target] = 10.0  # 정답 클래스만 높게

    return logits, targets


@pytest.fixture
def binary_perfect_predictions():
    """이진 분류 완벽한 예측

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    targets = torch.randint(0, BINARY_CLASSES, (NUM_SAMPLES,)).float()
    predictions = targets.clone()
    return predictions, targets


@pytest.fixture
def binary_probabilities():
    """이진 분류 확률 예측 (0.0 ~ 1.0)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (probabilities, targets)
    """
    targets = torch.randint(0, BINARY_CLASSES, (NUM_SAMPLES,)).float()
    probabilities = torch.rand(NUM_SAMPLES)
    return probabilities, targets


@pytest.fixture
def imbalanced_predictions():
    """불균형 데이터셋 예측

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    # Given: 클래스 0이 90%, 나머지가 10%
    num_class_0 = int(NUM_SAMPLES * 0.9)
    num_others = NUM_SAMPLES - num_class_0

    targets = torch.cat([
        torch.zeros(num_class_0, dtype=torch.long),
        torch.randint(1, SMALL_NUM_CLASSES, (num_others,))
    ])

    # 섞기
    indices = torch.randperm(NUM_SAMPLES)
    targets = targets[indices]
    predictions = targets.clone()

    return predictions, targets


@pytest.fixture
def top5_logits():
    """Top-5 정확도 테스트용 logits

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (logits, targets)
    """
    # Given: 정답이 상위 5개 안에 포함되도록 설정
    targets = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)

    # 정답을 상위 5개 안에 포함
    for i, target in enumerate(targets):
        # 정답 클래스의 logit을 상위 5개 중 하나로 설정
        logits[i, target] = torch.randn(1).item() + 5.0  # 충분히 높게

    return logits, targets


# ============================================================================
# Regression Data Fixtures
# ============================================================================

@pytest.fixture
def perfect_regression():
    """완벽한 회귀 예측 (MSE=0, MAE=0, R²=1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    targets = torch.randn(NUM_SAMPLES)
    predictions = targets.clone()
    return predictions, targets


@pytest.fixture
def linear_regression_data():
    """선형 관계 회귀 데이터 (y = 2x + 1)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    # Given: y = 2x + 1 관계
    x = torch.linspace(0, 10, NUM_SAMPLES)
    slope = 2.0
    intercept = 1.0
    targets = slope * x + intercept

    # 완벽한 예측
    predictions = targets.clone()

    return predictions, targets


@pytest.fixture
def noisy_regression_data():
    """노이즈가 있는 회귀 데이터

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    targets = torch.randn(NUM_SAMPLES)
    noise_level = 0.1
    predictions = targets + torch.randn(NUM_SAMPLES) * noise_level
    return predictions, targets


@pytest.fixture
def constant_predictions():
    """상수 예측 (평균값으로 예측) - R² = 0

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    targets = torch.randn(NUM_SAMPLES)
    predictions = torch.full((NUM_SAMPLES,), targets.mean().item())
    return predictions, targets


# ============================================================================
# MetricTracker Fixtures
# ============================================================================

@pytest.fixture
def metric_names():
    """테스트용 메트릭 이름 리스트

    Returns:
        List[str]: 메트릭 이름 리스트
    """
    return ['accuracy', 'precision', 'recall']


@pytest.fixture
def single_metric_name():
    """단일 메트릭 이름

    Returns:
        str: 메트릭 이름
    """
    return 'accuracy'


# ============================================================================
# Utility Functions
# ============================================================================

def assert_float_equal(actual: float, expected: float, tolerance: float = FLOAT_TOLERANCE):
    """부동소수점 비교 헬퍼 함수

    Args:
        actual: 실제 값
        expected: 예상 값
        tolerance: 허용 오차
    """
    assert abs(actual - expected) < tolerance, \
        f"Expected {expected}, but got {actual} (tolerance: {tolerance})"


def create_confusion_matrix_data(tp: int, fp: int, fn: int, tn: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Confusion matrix 기반 데이터 생성 (이진 분류)

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        tn: True Negatives

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
    """
    # Given: Confusion matrix 정의
    predictions = torch.cat([
        torch.ones(tp),   # TP: 1로 예측, 실제 1
        torch.ones(fp),   # FP: 1로 예측, 실제 0
        torch.zeros(fn),  # FN: 0으로 예측, 실제 1
        torch.zeros(tn)   # TN: 0으로 예측, 실제 0
    ]).long()

    targets = torch.cat([
        torch.ones(tp),   # TP
        torch.zeros(fp),  # FP
        torch.ones(fn),   # FN
        torch.zeros(tn)   # TN
    ]).long()

    # 섞기
    total = tp + fp + fn + tn
    indices = torch.randperm(total)

    return predictions[indices], targets[indices]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """pytest 설정"""
    # 마커 등록
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


@pytest.fixture(autouse=True)
def seed_everything():
    """모든 테스트에 랜덤 시드 고정 (재현 가능성)"""
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

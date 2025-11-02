"""
pytest 공통 fixtures 및 설정
"""

import pytest
import torch
import numpy as np
from typing import Dict

from research import (
    ExperimentResult,
    MetricTracker,
    AccuracyMetric,
    F1ScoreMetric,
    MSEMetric,
    MAEMetric,
)


@pytest.fixture
def device():
    """테스트용 디바이스 반환"""
    return torch.device('cpu')


@pytest.fixture
def dummy_multiclass_data():
    """다중 분류용 더미 데이터 반환"""
    batch_size = 32
    num_classes = 10

    X = torch.randn(batch_size, num_classes)
    y = torch.randint(0, num_classes, (batch_size,))

    return X, y


@pytest.fixture
def dummy_binary_data():
    """이진 분류용 더미 데이터 반환"""
    batch_size = 32

    X = torch.randn(batch_size, 20)
    y = torch.randint(0, 2, (batch_size,))

    return X, y


@pytest.fixture
def dummy_regression_data():
    """회귀용 더미 데이터 반환"""
    batch_size = 32

    X = torch.randn(batch_size, 10)
    y = torch.randn(batch_size)

    return X, y


@pytest.fixture
def classification_metric_tracker():
    """분류용 메트릭 트래커 반환"""
    return MetricTracker([
        AccuracyMetric(),
        F1ScoreMetric(average='macro')
    ])


@pytest.fixture
def regression_metric_tracker():
    """회귀용 메트릭 트래커 반환"""
    return MetricTracker([
        MSEMetric(),
        MAEMetric()
    ])


@pytest.fixture
def dummy_experiment_results() -> Dict[str, ExperimentResult]:
    """더미 실험 결과 3개 반환"""
    results = {}

    models_info = [
        {'name': 'Model_A', 'params': 10_000_000, 'inference_time': 0.05, 'base_acc': 0.85},
        {'name': 'Model_B', 'params': 20_000_000, 'inference_time': 0.10, 'base_acc': 0.88},
        {'name': 'Model_C', 'params': 5_000_000, 'inference_time': 0.03, 'base_acc': 0.82},
    ]

    for model_info in models_info:
        num_epochs = 5

        base_acc = model_info['base_acc']
        train_acc = [base_acc - 0.1 + i*0.02 for i in range(num_epochs)]
        val_acc = [base_acc - 0.05 + i*0.015 for i in range(num_epochs)]
        test_acc = [base_acc - 0.03 + i*0.01 for i in range(num_epochs)]

        train_loss = [1.5 - i*0.2 for i in range(num_epochs)]
        val_loss = [1.6 - i*0.18 for i in range(num_epochs)]
        test_loss = [1.65 - i*0.15 for i in range(num_epochs)]

        train_metrics = {'Accuracy': train_acc}
        val_metrics = {'Accuracy': val_acc}
        test_metrics = {'Accuracy': test_acc}

        epoch_times = [2.0 for _ in range(num_epochs)]

        result = ExperimentResult(
            model_name=model_info['name'],
            task_type='MultiClassStrategy',
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            primary_metric_name='Accuracy',
            best_test_metric=max(test_acc),
            parameters=model_info['params'],
            epoch_times=epoch_times,
            inference_time=model_info['inference_time']
        )

        results[model_info['name']] = result

    return results


@pytest.fixture
def seed_everything():
    """재현 가능한 결과를 위한 시드 설정"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

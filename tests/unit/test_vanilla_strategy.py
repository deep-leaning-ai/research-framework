"""
VanillaTrainingStrategy 통합 테스트
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research.strategies.training import VanillaTrainingStrategy
from research.strategies.task import MultiClassStrategy, RegressionStrategy
from research.strategies.logging import SimpleLoggingStrategy


class DummyModel(nn.Module):
    """테스트용 더미 모델"""
    def __init__(self, input_size=10, output_size=3):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_data():
    """더미 데이터셋 생성"""
    # 분류 데이터
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16)
    test_loader = DataLoader(dataset, batch_size=16)

    return train_loader, val_loader, test_loader


@pytest.fixture
def regression_data():
    """회귀 데이터셋 생성"""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)  # 회귀 타겟

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16)
    test_loader = DataLoader(dataset, batch_size=16)

    return train_loader, val_loader, test_loader


class TestVanillaStrategyWithTaskStrategy:
    """VanillaTrainingStrategy + TaskStrategy 통합 테스트"""

    def test_vanilla_with_multiclass_strategy(self, dummy_data):
        """MultiClassStrategy와의 통합 테스트"""
        train_loader, val_loader, test_loader = dummy_data

        # TaskStrategy 설정
        task_strategy = MultiClassStrategy(num_classes=3)

        # VanillaTrainingStrategy 생성
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        # 모델 생성
        model = DummyModel(input_size=10, output_size=3)

        # 학습 config
        config = {
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'adam'
        }

        # 학습 실행
        result = training_strategy.train(model, train_loader, val_loader, config)

        # 결과 검증
        assert 'training_time' in result
        # TaskStrategy가 반환하는 메트릭 이름은 'Accuracy' (대문자)
        assert ('best_val_Accuracy' in result or 'best_val_acc' in result)
        assert 'history' in result
        # 하위 호환성 키 확인
        assert 'best_val_acc' in result

        # 평가 실행
        test_result = training_strategy.evaluate(model, test_loader)
        # 'test_Accuracy' 또는 하위 호환성 'test_acc'
        assert ('test_Accuracy' in test_result or 'test_acc' in test_result)
        assert 'inference_time' in test_result

    def test_vanilla_with_regression_strategy(self, regression_data):
        """RegressionStrategy와의 통합 테스트"""
        train_loader, val_loader, test_loader = regression_data

        # TaskStrategy 설정
        task_strategy = RegressionStrategy()

        # VanillaTrainingStrategy 생성
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        # 모델 생성 (회귀용 - output_size=1)
        model = DummyModel(input_size=10, output_size=1)

        # 학습 config
        config = {
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'adam'
        }

        # 학습 실행
        result = training_strategy.train(model, train_loader, val_loader, config)

        # 결과 검증
        assert 'training_time' in result
        # 회귀의 경우 MSE 또는 R2
        assert 'history' in result

        # 평가 실행
        test_result = training_strategy.evaluate(model, test_loader)
        assert 'inference_time' in test_result
        # 하위 호환성 키 확인
        assert 'test_acc' in test_result

    def test_vanilla_without_task_strategy_backward_compatible(self, dummy_data):
        """TaskStrategy 없이도 작동하는지 (하위 호환성) 테스트"""
        train_loader, val_loader, test_loader = dummy_data

        # TaskStrategy 없이 생성
        training_strategy = VanillaTrainingStrategy(device='cpu')

        # 모델 생성
        model = DummyModel(input_size=10, output_size=3)

        # 학습 config
        config = {
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'adam'
        }

        # 학습 실행 (기본 CrossEntropyLoss 사용)
        result = training_strategy.train(model, train_loader, val_loader, config)

        # 결과 검증
        assert 'training_time' in result
        assert 'best_val_acc' in result
        assert 'history' in result

        # 평가 실행
        test_result = training_strategy.evaluate(model, test_loader)
        assert 'test_acc' in test_result


class TestAdvancedFeatures:
    """고급 기능 테스트"""

    def test_learning_rate_scheduler(self, dummy_data):
        """Learning rate scheduler 테스트"""
        train_loader, val_loader, test_loader = dummy_data

        task_strategy = MultiClassStrategy(num_classes=3)
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        model = DummyModel(input_size=10, output_size=3)

        config = {
            'max_epochs': 3,
            'learning_rate': 1e-2,
            'optimizer': 'adam',
            'use_scheduler': True
        }

        result = training_strategy.train(model, train_loader, val_loader, config)
        assert result is not None

    def test_early_stopping(self, dummy_data):
        """Early stopping 테스트"""
        train_loader, val_loader, test_loader = dummy_data

        task_strategy = MultiClassStrategy(num_classes=3)
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        model = DummyModel(input_size=10, output_size=3)

        config = {
            'max_epochs': 20,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'use_early_stopping': True,
            'patience': 3
        }

        result = training_strategy.train(model, train_loader, val_loader, config)
        # Early stopping으로 인해 20 에폭보다 적게 실행될 수 있음
        assert len(result['history']['train_loss']) <= 20

    def test_gradient_clipping(self, dummy_data):
        """Gradient clipping 테스트"""
        train_loader, val_loader, test_loader = dummy_data

        task_strategy = MultiClassStrategy(num_classes=3)
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        model = DummyModel(input_size=10, output_size=3)

        config = {
            'max_epochs': 2,
            'learning_rate': 1e-3,
            'optimizer': 'adam',
            'grad_clip_max_norm': 1.0
        }

        result = training_strategy.train(model, train_loader, val_loader, config)
        assert result is not None

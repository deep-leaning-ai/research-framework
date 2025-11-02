"""
Task Strategy Unit 테스트
"""

import pytest
import torch
import torch.nn as nn

from research import (
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy,
)


@pytest.mark.unit
class TestMultiClassStrategy:
    """MultiClassStrategy 테스트"""

    def test_criterion_type(self):
        """Criterion 타입 테스트"""
        strategy = MultiClassStrategy(num_classes=10)
        criterion = strategy.get_criterion()

        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_output_activation(self):
        """Output activation 테스트"""
        strategy = MultiClassStrategy(num_classes=10)
        activation = strategy.get_output_activation()

        assert activation is None

    def test_metric_calculation(self):
        """메트릭 계산 테스트"""
        strategy = MultiClassStrategy(num_classes=10)

        outputs = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))

        metric = strategy.calculate_metric(outputs, labels)

        assert 0 <= metric <= 100

    def test_metric_name(self):
        """메트릭 이름 테스트"""
        strategy = MultiClassStrategy(num_classes=10)

        assert strategy.get_metric_name() == "Accuracy"

    def test_prepare_labels(self):
        """레이블 준비 테스트"""
        strategy = MultiClassStrategy(num_classes=10)

        labels = torch.randint(0, 10, (32,))
        prepared = strategy.prepare_labels(labels)

        assert torch.equal(labels, prepared)


@pytest.mark.unit
class TestBinaryClassificationStrategy:
    """BinaryClassificationStrategy 테스트"""

    def test_criterion_type(self):
        """Criterion 타입 테스트"""
        strategy = BinaryClassificationStrategy()
        criterion = strategy.get_criterion()

        assert isinstance(criterion, nn.BCEWithLogitsLoss)

    def test_output_activation(self):
        """Output activation 테스트"""
        strategy = BinaryClassificationStrategy()
        activation = strategy.get_output_activation()

        assert isinstance(activation, nn.Sigmoid)

    def test_metric_calculation(self):
        """메트릭 계산 테스트"""
        strategy = BinaryClassificationStrategy()

        outputs = torch.randn(32, 1)
        labels = torch.randint(0, 2, (32, 1)).float()

        metric = strategy.calculate_metric(outputs, labels)

        assert 0 <= metric <= 100

    def test_prepare_labels(self):
        """레이블 준비 테스트"""
        strategy = BinaryClassificationStrategy()

        labels = torch.randint(0, 2, (32,))
        prepared = strategy.prepare_labels(labels)

        assert prepared.shape == (32, 1)
        assert prepared.dtype == torch.float32

    def test_loss_calculation(self):
        """손실 계산 테스트"""
        strategy = BinaryClassificationStrategy()
        criterion = strategy.get_criterion()

        outputs = torch.randn(32, 1)
        labels = torch.randint(0, 2, (32,))
        prepared_labels = strategy.prepare_labels(labels)

        loss = criterion(outputs, prepared_labels)

        assert loss.item() >= 0


@pytest.mark.unit
class TestRegressionStrategy:
    """RegressionStrategy 테스트"""

    def test_criterion_type(self):
        """Criterion 타입 테스트"""
        strategy = RegressionStrategy()
        criterion = strategy.get_criterion()

        assert isinstance(criterion, nn.MSELoss)

    def test_output_activation(self):
        """Output activation 테스트"""
        strategy = RegressionStrategy()
        activation = strategy.get_output_activation()

        assert activation is None

    def test_metric_calculation(self):
        """메트릭 계산 테스트"""
        strategy = RegressionStrategy()

        outputs = torch.randn(32, 1)
        labels = torch.randn(32, 1)

        metric = strategy.calculate_metric(outputs, labels)

        assert metric >= 0

    def test_metric_name(self):
        """메트릭 이름 테스트"""
        strategy = RegressionStrategy()

        assert strategy.get_metric_name() == "MSE"

    def test_prepare_labels(self):
        """레이블 준비 테스트"""
        strategy = RegressionStrategy()

        labels = torch.randn(32)
        prepared = strategy.prepare_labels(labels)

        assert prepared.shape == (32, 1)
        assert prepared.dtype == torch.float32

    def test_loss_calculation(self):
        """손실 계산 테스트"""
        strategy = RegressionStrategy()
        criterion = strategy.get_criterion()

        outputs = torch.randn(32, 1)
        labels = torch.randn(32)
        prepared_labels = strategy.prepare_labels(labels)

        loss = criterion(outputs, prepared_labels)

        assert loss.item() >= 0

"""Unit tests for task strategies

All tests follow the Given-When-Then pattern and avoid hardcoding.
"""

import pytest
import torch
import torch.nn as nn

# Test constants
NUM_SAMPLES = 32
NUM_CLASSES = 10
PERFECT_ACCURACY = 1.0
ZERO_ACCURACY = 0.0
PERFECT_MSE = 0.0
BINARY_THRESHOLD = 0.5


@pytest.mark.unit
class TestMultiClassStrategy:
    """MultiClassStrategy tests"""

    def test_get_criterion(self):
        """Given: MultiClassStrategy
        When: get_criterion() called
        Then: Return CrossEntropyLoss"""
        # Given
        from research.strategies.task import MultiClassStrategy
        strategy = MultiClassStrategy(num_classes=NUM_CLASSES)

        # When
        criterion = strategy.get_criterion()

        # Then
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_get_activation(self):
        """Given: MultiClassStrategy
        When: get_activation() called
        Then: Return Softmax"""
        # Given
        from research.strategies.task import MultiClassStrategy
        strategy = MultiClassStrategy(num_classes=NUM_CLASSES)

        # When
        activation = strategy.get_activation()

        # Then
        assert isinstance(activation, nn.Softmax)

    def test_calculate_metric_perfect(self):
        """Given: Perfect predictions
        When: calculate_metric() called
        Then: Return 1.0"""
        # Given
        from research.strategies.task import MultiClassStrategy
        strategy = MultiClassStrategy(num_classes=NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
        outputs = torch.zeros(NUM_SAMPLES, NUM_CLASSES)
        for i, label in enumerate(labels):
            outputs[i, label] = 10.0  # High logit for correct class

        # When
        accuracy = strategy.calculate_metric(outputs, labels)

        # Then
        assert accuracy == PERFECT_ACCURACY

    def test_prepare_labels(self):
        """Given: Label tensor
        When: prepare_labels() called
        Then: Return Long tensor"""
        # Given
        from research.strategies.task import MultiClassStrategy
        strategy = MultiClassStrategy(num_classes=NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,)).float()

        # When
        prepared = strategy.prepare_labels(labels)

        # Then
        assert prepared.dtype == torch.long

    def test_get_task_type(self):
        """Given: MultiClassStrategy
        When: get_task_type() called
        Then: Return 'multiclass'"""
        # Given
        from research.strategies.task import MultiClassStrategy
        strategy = MultiClassStrategy(num_classes=NUM_CLASSES)

        # When
        task_type = strategy.get_task_type()

        # Then
        assert task_type == "multiclass"


@pytest.mark.unit
class TestBinaryClassificationStrategy:
    """BinaryClassificationStrategy tests"""

    def test_get_criterion(self):
        """Given: BinaryClassificationStrategy
        When: get_criterion() called
        Then: Return BCEWithLogitsLoss"""
        # Given
        from research.strategies.task import BinaryClassificationStrategy
        strategy = BinaryClassificationStrategy()

        # When
        criterion = strategy.get_criterion()

        # Then
        assert isinstance(criterion, nn.BCEWithLogitsLoss)

    def test_get_activation(self):
        """Given: BinaryClassificationStrategy
        When: get_activation() called
        Then: Return Sigmoid"""
        # Given
        from research.strategies.task import BinaryClassificationStrategy
        strategy = BinaryClassificationStrategy()

        # When
        activation = strategy.get_activation()

        # Then
        assert isinstance(activation, nn.Sigmoid)

    def test_calculate_metric_perfect(self):
        """Given: Perfect predictions
        When: calculate_metric() called
        Then: Return 1.0"""
        # Given
        from research.strategies.task import BinaryClassificationStrategy
        strategy = BinaryClassificationStrategy()
        labels = torch.randint(0, 2, (NUM_SAMPLES,)).float()
        outputs = (labels * 10.0) - 5.0  # Positive logits for 1, negative for 0

        # When
        accuracy = strategy.calculate_metric(outputs, labels)

        # Then
        assert accuracy == PERFECT_ACCURACY

    def test_prepare_labels(self):
        """Given: Label tensor
        When: prepare_labels() called
        Then: Return Float tensor"""
        # Given
        from research.strategies.task import BinaryClassificationStrategy
        strategy = BinaryClassificationStrategy()
        labels = torch.randint(0, 2, (NUM_SAMPLES,))

        # When
        prepared = strategy.prepare_labels(labels)

        # Then
        assert prepared.dtype == torch.float32

    def test_threshold_constant(self):
        """Given: BinaryClassificationStrategy
        When: Check THRESHOLD constant
        Then: Value is 0.5"""
        # Given & When
        from research.strategies.task import BinaryClassificationStrategy

        # Then
        assert BinaryClassificationStrategy.THRESHOLD == BINARY_THRESHOLD

    def test_get_task_type(self):
        """Given: BinaryClassificationStrategy
        When: get_task_type() called
        Then: Return 'binary'"""
        # Given
        from research.strategies.task import BinaryClassificationStrategy
        strategy = BinaryClassificationStrategy()

        # When
        task_type = strategy.get_task_type()

        # Then
        assert task_type == "binary"


@pytest.mark.unit
class TestRegressionStrategy:
    """RegressionStrategy tests"""

    def test_get_criterion(self):
        """Given: RegressionStrategy
        When: get_criterion() called
        Then: Return MSELoss"""
        # Given
        from research.strategies.task import RegressionStrategy
        strategy = RegressionStrategy()

        # When
        criterion = strategy.get_criterion()

        # Then
        assert isinstance(criterion, nn.MSELoss)

    def test_get_activation(self):
        """Given: RegressionStrategy
        When: get_activation() called
        Then: Return None"""
        # Given
        from research.strategies.task import RegressionStrategy
        strategy = RegressionStrategy()

        # When
        activation = strategy.get_activation()

        # Then
        assert activation is None

    def test_calculate_metric_perfect(self):
        """Given: Perfect predictions
        When: calculate_metric() called
        Then: Return 0.0 (MSE)"""
        # Given
        from research.strategies.task import RegressionStrategy
        strategy = RegressionStrategy()
        labels = torch.randn(NUM_SAMPLES)
        outputs = labels.clone()

        # When
        mse = strategy.calculate_metric(outputs, labels)

        # Then
        assert abs(mse - PERFECT_MSE) < 1e-6

    def test_prepare_labels(self):
        """Given: Label tensor
        When: prepare_labels() called
        Then: Return Float tensor"""
        # Given
        from research.strategies.task import RegressionStrategy
        strategy = RegressionStrategy()
        labels = torch.randint(0, 100, (NUM_SAMPLES,))

        # When
        prepared = strategy.prepare_labels(labels)

        # Then
        assert prepared.dtype == torch.float32

    def test_get_task_type(self):
        """Given: RegressionStrategy
        When: get_task_type() called
        Then: Return 'regression'"""
        # Given
        from research.strategies.task import RegressionStrategy
        strategy = RegressionStrategy()

        # When
        task_type = strategy.get_task_type()

        # Then
        assert task_type == "regression"


@pytest.mark.unit
class TestStrategyInterchangeability:
    """Test strategy interchangeability (LSP)"""

    def test_all_strategies_have_same_interface(self):
        """Given: All task strategies
        When: Check methods
        Then: All have same interface"""
        # Given
        from research.strategies.task import (
            MultiClassStrategy,
            BinaryClassificationStrategy,
            RegressionStrategy
        )

        strategies = [
            MultiClassStrategy(num_classes=NUM_CLASSES),
            BinaryClassificationStrategy(),
            RegressionStrategy()
        ]

        # When & Then
        for strategy in strategies:
            assert hasattr(strategy, 'get_criterion')
            assert hasattr(strategy, 'get_activation')
            assert hasattr(strategy, 'calculate_metric')
            assert hasattr(strategy, 'prepare_labels')
            assert hasattr(strategy, 'get_task_type')

    def test_strategies_polymorphism(self):
        """Given: Strategy list
        When: Call get_task_type() on each
        Then: Return different task types"""
        # Given
        from research.strategies.task import (
            MultiClassStrategy,
            BinaryClassificationStrategy,
            RegressionStrategy
        )

        strategies = [
            MultiClassStrategy(num_classes=NUM_CLASSES),
            BinaryClassificationStrategy(),
            RegressionStrategy()
        ]

        # When
        task_types = [s.get_task_type() for s in strategies]

        # Then
        assert "multiclass" in task_types
        assert "binary" in task_types
        assert "regression" in task_types
        assert len(set(task_types)) == 3  # All unique

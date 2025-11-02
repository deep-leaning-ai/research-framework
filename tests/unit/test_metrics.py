"""
메트릭 시스템 Unit 테스트
"""

import pytest
import torch

from research import (
    MetricTracker,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    MSEMetric,
    MAEMetric,
)


@pytest.mark.unit
class TestAccuracyMetric:
    """AccuracyMetric 테스트"""

    def test_accuracy_perfect(self):
        """완벽한 예측 테스트"""
        metric = AccuracyMetric()

        outputs = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        labels = torch.tensor([0, 1, 2])

        accuracy = metric.calculate(outputs, labels)

        assert accuracy == 100.0, "Perfect predictions should give 100% accuracy"

    def test_accuracy_random(self):
        """랜덤 예측 테스트"""
        metric = AccuracyMetric()

        outputs = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))

        accuracy = metric.calculate(outputs, labels)

        assert 0 <= accuracy <= 100, "Accuracy should be between 0 and 100"

    def test_metric_name(self):
        """메트릭 이름 테스트"""
        metric = AccuracyMetric()
        assert metric.get_name() == "Accuracy"
        assert metric.name == "Accuracy"

    def test_higher_better(self):
        """높을수록 좋은 메트릭인지 테스트"""
        metric = AccuracyMetric()
        assert metric.is_higher_better() is True


@pytest.mark.unit
class TestF1ScoreMetric:
    """F1ScoreMetric 테스트"""

    def test_f1_macro(self):
        """Macro F1-Score 테스트"""
        metric = F1ScoreMetric(average='macro')

        outputs = torch.randn(32, 3)
        labels = torch.randint(0, 3, (32,))

        f1 = metric.calculate(outputs, labels)

        assert 0 <= f1 <= 100, "F1-Score should be between 0 and 100"

    def test_metric_name(self):
        """메트릭 이름 테스트"""
        metric = F1ScoreMetric(average='macro')
        assert metric.get_name() == "F1-Score (macro)"


@pytest.mark.unit
class TestMSEMetric:
    """MSEMetric 테스트"""

    def test_mse_zero(self):
        """완벽한 예측 (MSE=0) 테스트"""
        metric = MSEMetric()

        outputs = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([[1.0], [2.0], [3.0]])

        mse = metric.calculate(outputs, labels)

        assert mse < 1e-6, "Perfect predictions should give MSE close to 0"

    def test_mse_positive(self):
        """MSE는 항상 양수"""
        metric = MSEMetric()

        outputs = torch.randn(32, 1)
        labels = torch.randn(32, 1)

        mse = metric.calculate(outputs, labels)

        assert mse >= 0, "MSE should be non-negative"

    def test_higher_better(self):
        """낮을수록 좋은 메트릭인지 테스트"""
        metric = MSEMetric()
        assert metric.is_higher_better() is False


@pytest.mark.unit
class TestMetricTracker:
    """MetricTracker 테스트"""

    def test_tracker_initialization(self, classification_metric_tracker):
        """트래커 초기화 테스트"""
        tracker = classification_metric_tracker

        assert len(tracker.metrics) == 2
        assert len(tracker.history) == 2
        assert 'Accuracy' in tracker.history
        assert 'F1-Score (macro)' in tracker.history

    def test_tracker_update(self, classification_metric_tracker):
        """트래커 업데이트 테스트"""
        tracker = classification_metric_tracker

        outputs = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))

        metrics = tracker.update(outputs, labels)

        assert 'Accuracy' in metrics
        assert 'F1-Score (macro)' in metrics
        assert len(tracker.history['Accuracy']) == 1
        assert len(tracker.history['F1-Score (macro)']) == 1

    def test_tracker_multiple_updates(self, classification_metric_tracker):
        """여러 번 업데이트 테스트"""
        tracker = classification_metric_tracker

        for _ in range(5):
            outputs = torch.randn(32, 10)
            labels = torch.randint(0, 10, (32,))
            tracker.update(outputs, labels)

        assert len(tracker.history['Accuracy']) == 5
        assert len(tracker.history['F1-Score (macro)']) == 5

    def test_tracker_get_history(self, classification_metric_tracker):
        """히스토리 가져오기 테스트"""
        tracker = classification_metric_tracker

        outputs = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))
        tracker.update(outputs, labels)

        history = tracker.get_all_history()

        assert 'Accuracy' in history
        assert isinstance(history['Accuracy'], list)

        acc_history = tracker.get_history('Accuracy')
        assert isinstance(acc_history, list)
        assert len(acc_history) == 1

    def test_tracker_reset(self, classification_metric_tracker):
        """트래커 초기화 테스트"""
        tracker = classification_metric_tracker

        outputs = torch.randn(32, 10)
        labels = torch.randint(0, 10, (32,))
        tracker.update(outputs, labels)

        tracker.reset()

        assert len(tracker.history['Accuracy']) == 0
        assert len(tracker.history['F1-Score (macro)']) == 0
        assert 'Accuracy' in tracker.history
        assert 'F1-Score (macro)' in tracker.history

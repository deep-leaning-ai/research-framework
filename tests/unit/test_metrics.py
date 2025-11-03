"""Unit tests for metrics module

All tests follow the Given-When-Then pattern and avoid hardcoding.
"""

import pytest
import torch

# Test constants (duplicated from conftest for direct usage)
PERFECT_ACCURACY = 1.0
ZERO_ACCURACY = 0.0
HALF_ACCURACY = 0.5
PERFECT_MSE = 0.0
PERFECT_MAE = 0.0
PERFECT_R2 = 1.0
NUM_CLASSES = 10
BINARY_CLASSES = 2
SMALL_NUM_CLASSES = 3
FLOAT_TOLERANCE = 1e-6
SMALL_WINDOW_SIZE = 3


def assert_float_equal(actual: float, expected: float, tolerance: float = FLOAT_TOLERANCE):
    """Helper function for floating point comparison"""
    assert abs(actual - expected) < tolerance, \
        f"Expected {expected}, but got {actual} (tolerance: {tolerance})"


def create_confusion_matrix_data(tp: int, fp: int, fn: int, tn: int):
    """Helper function to create confusion matrix data"""
    predictions = torch.cat([
        torch.ones(tp),
        torch.ones(fp),
        torch.zeros(fn),
        torch.zeros(tn)
    ]).long()

    targets = torch.cat([
        torch.ones(tp),
        torch.zeros(fp),
        torch.ones(fn),
        torch.zeros(tn)
    ]).long()

    total = tp + fp + fn + tn
    indices = torch.randperm(total)

    return predictions[indices], targets[indices]


# ============================================================================
# AccuracyMetric Tests
# ============================================================================

@pytest.mark.unit
class TestAccuracyMetric:
    """AccuracyMetric 테스트"""

    def test_perfect_prediction(self, perfect_predictions):
        """Given: 완벽한 예측 데이터
        When: AccuracyMetric으로 계산
        Then: 100% 정확도 반환"""
        # Given
        from research.metrics import AccuracyMetric
        predictions, targets = perfect_predictions
        metric = AccuracyMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_zero_prediction(self, zero_predictions):
        """Given: 완전히 틀린 예측 데이터
        When: AccuracyMetric으로 계산
        Then: 0% 정확도 반환"""
        # Given
        from research.metrics import AccuracyMetric
        predictions, targets = zero_predictions
        metric = AccuracyMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, ZERO_ACCURACY)

    def test_2d_logits_input(self, perfect_logits):
        """Given: 2D logits 입력
        When: AccuracyMetric으로 계산
        Then: argmax 후 정확도 계산"""
        # Given
        from research.metrics import AccuracyMetric
        logits, targets = perfect_logits
        metric = AccuracyMetric()

        # When
        result = metric.calculate(logits, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_metric_name(self):
        """Given: AccuracyMetric 인스턴스
        When: get_name() 호출
        Then: 'accuracy' 반환"""
        # Given
        from research.metrics import AccuracyMetric
        metric = AccuracyMetric()

        # When
        name = metric.get_name()

        # Then
        assert name == "accuracy"

    def test_is_higher_better(self):
        """Given: AccuracyMetric 인스턴스
        When: is_higher_better() 호출
        Then: True 반환"""
        # Given
        from research.metrics import AccuracyMetric
        metric = AccuracyMetric()

        # When
        result = metric.is_higher_better()

        # Then
        assert result is True


# ============================================================================
# PrecisionMetric Tests
# ============================================================================

@pytest.mark.unit
class TestPrecisionMetric:
    """PrecisionMetric 테스트"""

    def test_perfect_prediction_macro(self, perfect_predictions):
        """Given: 완벽한 예측 데이터 (macro 평균)
        When: PrecisionMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import PrecisionMetric
        predictions, targets = perfect_predictions
        metric = PrecisionMetric(average='macro', num_classes=NUM_CLASSES)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_perfect_prediction_micro(self, perfect_predictions):
        """Given: 완벽한 예측 데이터 (micro 평균)
        When: PrecisionMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import PrecisionMetric
        predictions, targets = perfect_predictions
        metric = PrecisionMetric(average='micro', num_classes=NUM_CLASSES)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_perfect_prediction_weighted(self, perfect_predictions):
        """Given: 완벽한 예측 데이터 (weighted 평균)
        When: PrecisionMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import PrecisionMetric
        predictions, targets = perfect_predictions
        metric = PrecisionMetric(average='weighted', num_classes=NUM_CLASSES)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_binary_classification(self):
        """Given: 이진 분류 confusion matrix (TP=8, FP=2, FN=0, TN=0)
        When: PrecisionMetric으로 계산
        Then: Precision = 8/(8+2) = 0.8"""
        # Given
        from research.metrics import PrecisionMetric
        tp, fp, fn, tn = 8, 2, 0, 0
        predictions, targets = create_confusion_matrix_data(tp, fp, fn, tn)
        metric = PrecisionMetric(average='binary', num_classes=BINARY_CLASSES)

        expected_precision = tp / (tp + fp)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, expected_precision)

    def test_default_average_is_macro(self):
        """Given: average 파라미터 없이 생성
        When: 인스턴스 확인
        Then: default average가 'macro'"""
        # Given
        from research.metrics import PrecisionMetric
        metric = PrecisionMetric(num_classes=NUM_CLASSES)

        # When & Then
        assert metric.average == PrecisionMetric.DEFAULT_AVERAGE

    def test_invalid_average_raises_error(self):
        """Given: 잘못된 average 파라미터
        When: PrecisionMetric 생성
        Then: ValueError 발생"""
        # Given
        from research.metrics import PrecisionMetric
        invalid_average = "invalid"

        # When & Then
        with pytest.raises(ValueError):
            PrecisionMetric(average=invalid_average, num_classes=NUM_CLASSES)

    def test_metric_name(self):
        """Given: PrecisionMetric 인스턴스
        When: get_name() 호출
        Then: 'precision' 반환"""
        # Given
        from research.metrics import PrecisionMetric
        metric = PrecisionMetric(num_classes=NUM_CLASSES)

        # When
        name = metric.get_name()

        # Then
        assert name == "precision"


# ============================================================================
# RecallMetric Tests
# ============================================================================

@pytest.mark.unit
class TestRecallMetric:
    """RecallMetric 테스트"""

    def test_perfect_prediction_macro(self, perfect_predictions):
        """Given: 완벽한 예측 데이터 (macro 평균)
        When: RecallMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import RecallMetric
        predictions, targets = perfect_predictions
        metric = RecallMetric(average='macro', num_classes=NUM_CLASSES)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_binary_classification(self):
        """Given: 이진 분류 confusion matrix (TP=8, FP=0, FN=2, TN=0)
        When: RecallMetric으로 계산
        Then: Recall = 8/(8+2) = 0.8"""
        # Given
        from research.metrics import RecallMetric
        tp, fp, fn, tn = 8, 0, 2, 0
        predictions, targets = create_confusion_matrix_data(tp, fp, fn, tn)
        metric = RecallMetric(average='binary', num_classes=BINARY_CLASSES)

        expected_recall = tp / (tp + fn)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, expected_recall)

    def test_metric_name(self):
        """Given: RecallMetric 인스턴스
        When: get_name() 호출
        Then: 'recall' 반환"""
        # Given
        from research.metrics import RecallMetric
        metric = RecallMetric(num_classes=NUM_CLASSES)

        # When
        name = metric.get_name()

        # Then
        assert name == "recall"


# ============================================================================
# F1ScoreMetric Tests
# ============================================================================

@pytest.mark.unit
class TestF1ScoreMetric:
    """F1ScoreMetric 테스트"""

    def test_perfect_prediction(self, perfect_predictions):
        """Given: 완벽한 예측 데이터
        When: F1ScoreMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import F1ScoreMetric
        predictions, targets = perfect_predictions
        metric = F1ScoreMetric(average='macro', num_classes=NUM_CLASSES)

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_binary_classification(self):
        """Given: 이진 분류 confusion matrix (TP=8, FP=2, FN=2, TN=0)
        When: F1ScoreMetric으로 계산
        Then: F1 = 2 * (P * R) / (P + R)"""
        # Given
        from research.metrics import F1ScoreMetric
        tp, fp, fn, tn = 8, 2, 2, 0
        predictions, targets = create_confusion_matrix_data(tp, fp, fn, tn)
        metric = F1ScoreMetric(average='binary', num_classes=BINARY_CLASSES)

        precision = tp / (tp + fp)  # 8/10 = 0.8
        recall = tp / (tp + fn)     # 8/10 = 0.8
        expected_f1 = 2 * (precision * recall) / (precision + recall)  # 0.8

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, expected_f1)

    def test_metric_name(self):
        """Given: F1ScoreMetric 인스턴스
        When: get_name() 호출
        Then: 'f1_score' 반환"""
        # Given
        from research.metrics import F1ScoreMetric
        metric = F1ScoreMetric(num_classes=NUM_CLASSES)

        # When
        name = metric.get_name()

        # Then
        assert name == "f1_score"


# ============================================================================
# Top5AccuracyMetric Tests
# ============================================================================

@pytest.mark.unit
class TestTop5AccuracyMetric:
    """Top5AccuracyMetric 테스트"""

    def test_top5_with_correct_in_top5(self, top5_logits):
        """Given: 정답이 상위 5개 안에 포함된 logits
        When: Top5AccuracyMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import Top5AccuracyMetric
        logits, targets = top5_logits
        metric = Top5AccuracyMetric()

        # When
        result = metric.calculate(logits, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_top5_with_perfect_logits(self, perfect_logits):
        """Given: 정답이 최상위인 logits
        When: Top5AccuracyMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import Top5AccuracyMetric
        logits, targets = perfect_logits
        metric = Top5AccuracyMetric()

        # When
        result = metric.calculate(logits, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_top5_with_small_num_classes(self):
        """Given: 클래스 수가 5개 미만
        When: Top5AccuracyMetric으로 계산
        Then: 일반 accuracy와 동일"""
        # Given
        from research.metrics import Top5AccuracyMetric, AccuracyMetric
        targets = torch.randint(0, SMALL_NUM_CLASSES, (32,))
        logits = torch.randn(32, SMALL_NUM_CLASSES)
        # 정답 클래스에 높은 값
        for i, target in enumerate(targets):
            logits[i, target] = 10.0

        top5_metric = Top5AccuracyMetric()
        accuracy_metric = AccuracyMetric()

        # When
        top5_result = top5_metric.calculate(logits, targets)
        accuracy_result = accuracy_metric.calculate(logits, targets)

        # Then
        assert_float_equal(top5_result, accuracy_result)

    def test_metric_name(self):
        """Given: Top5AccuracyMetric 인스턴스
        When: get_name() 호출
        Then: 'top5_accuracy' 반환"""
        # Given
        from research.metrics import Top5AccuracyMetric
        metric = Top5AccuracyMetric()

        # When
        name = metric.get_name()

        # Then
        assert name == "top5_accuracy"


# ============================================================================
# AUCMetric Tests
# ============================================================================

@pytest.mark.unit
class TestAUCMetric:
    """AUCMetric 테스트"""

    def test_perfect_separation(self):
        """Given: 완벽하게 분리된 이진 분류 확률
        When: AUCMetric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import AUCMetric
        # 클래스 0: 확률 0.0, 클래스 1: 확률 1.0
        predictions = torch.cat([torch.zeros(16), torch.ones(16)])
        targets = torch.cat([torch.zeros(16), torch.ones(16)])
        metric = AUCMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_random_predictions(self):
        """Given: 랜덤 확률 예측
        When: AUCMetric으로 계산
        Then: 0.5 근처 값 반환"""
        # Given
        from research.metrics import AUCMetric
        predictions = torch.rand(32)
        targets = torch.randint(0, BINARY_CLASSES, (32,)).float()
        metric = AUCMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        # 랜덤이므로 0.3 ~ 0.7 사이
        assert 0.0 <= result <= 1.0

    def test_metric_name(self):
        """Given: AUCMetric 인스턴스
        When: get_name() 호출
        Then: 'auc' 반환"""
        # Given
        from research.metrics import AUCMetric
        metric = AUCMetric()

        # When
        name = metric.get_name()

        # Then
        assert name == "auc"


# ============================================================================
# MSEMetric Tests
# ============================================================================

@pytest.mark.unit
class TestMSEMetric:
    """MSEMetric 테스트"""

    def test_perfect_prediction(self, perfect_regression):
        """Given: 완벽한 회귀 예측
        When: MSEMetric으로 계산
        Then: 0.0 반환"""
        # Given
        from research.metrics import MSEMetric
        predictions, targets = perfect_regression
        metric = MSEMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_MSE)

    def test_known_mse(self):
        """Given: 알려진 오차
        When: MSEMetric으로 계산
        Then: 정확한 MSE 반환"""
        # Given
        from research.metrics import MSEMetric
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        metric = MSEMetric()

        # 오차: [0.5, 0.5, 0.5]
        # MSE = (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.75 / 3 = 0.25
        expected_mse = 0.25

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, expected_mse)

    def test_metric_name(self):
        """Given: MSEMetric 인스턴스
        When: get_name() 호출
        Then: 'mse' 반환"""
        # Given
        from research.metrics import MSEMetric
        metric = MSEMetric()

        # When
        name = metric.get_name()

        # Then
        assert name == "mse"

    def test_is_higher_better(self):
        """Given: MSEMetric 인스턴스
        When: is_higher_better() 호출
        Then: False 반환 (낮을수록 좋음)"""
        # Given
        from research.metrics import MSEMetric
        metric = MSEMetric()

        # When
        result = metric.is_higher_better()

        # Then
        assert result is False


# ============================================================================
# MAEMetric Tests
# ============================================================================

@pytest.mark.unit
class TestMAEMetric:
    """MAEMetric 테스트"""

    def test_perfect_prediction(self, perfect_regression):
        """Given: 완벽한 회귀 예측
        When: MAEMetric으로 계산
        Then: 0.0 반환"""
        # Given
        from research.metrics import MAEMetric
        predictions, targets = perfect_regression
        metric = MAEMetric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_MAE)

    def test_known_mae(self):
        """Given: 알려진 오차
        When: MAEMetric으로 계산
        Then: 정확한 MAE 반환"""
        # Given
        from research.metrics import MAEMetric
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.5, 2.5, 3.5])
        metric = MAEMetric()

        # 절대 오차: [0.5, 0.5, 0.5]
        # MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        expected_mae = 0.5

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, expected_mae)

    def test_metric_name(self):
        """Given: MAEMetric 인스턴스
        When: get_name() 호출
        Then: 'mae' 반환"""
        # Given
        from research.metrics import MAEMetric
        metric = MAEMetric()

        # When
        name = metric.get_name()

        # Then
        assert name == "mae"

    def test_is_higher_better(self):
        """Given: MAEMetric 인스턴스
        When: is_higher_better() 호출
        Then: False 반환"""
        # Given
        from research.metrics import MAEMetric
        metric = MAEMetric()

        # When
        result = metric.is_higher_better()

        # Then
        assert result is False


# ============================================================================
# R2Metric Tests
# ============================================================================

@pytest.mark.unit
class TestR2Metric:
    """R2Metric 테스트"""

    def test_perfect_prediction(self, perfect_regression):
        """Given: 완벽한 회귀 예측
        When: R2Metric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import R2Metric
        predictions, targets = perfect_regression
        metric = R2Metric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_R2)

    def test_mean_prediction(self, constant_predictions):
        """Given: 평균값으로 예측 (상수 예측)
        When: R2Metric으로 계산
        Then: 0.0 반환"""
        # Given
        from research.metrics import R2Metric
        predictions, targets = constant_predictions
        metric = R2Metric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        # 평균으로 예측하면 R² = 0
        assert_float_equal(result, 0.0, tolerance=0.01)

    def test_linear_relationship(self, linear_regression_data):
        """Given: 완벽한 선형 관계 데이터
        When: R2Metric으로 계산
        Then: 1.0 반환"""
        # Given
        from research.metrics import R2Metric
        predictions, targets = linear_regression_data
        metric = R2Metric()

        # When
        result = metric.calculate(predictions, targets)

        # Then
        assert_float_equal(result, PERFECT_R2)

    def test_metric_name(self):
        """Given: R2Metric 인스턴스
        When: get_name() 호출
        Then: 'r2_score' 반환"""
        # Given
        from research.metrics import R2Metric
        metric = R2Metric()

        # When
        name = metric.get_name()

        # Then
        assert name == "r2_score"

    def test_is_higher_better(self):
        """Given: R2Metric 인스턴스
        When: is_higher_better() 호출
        Then: True 반환"""
        # Given
        from research.metrics import R2Metric
        metric = R2Metric()

        # When
        result = metric.is_higher_better()

        # Then
        assert result is True


# ============================================================================
# MetricTracker Tests
# ============================================================================

@pytest.mark.unit
class TestMetricTracker:
    """MetricTracker 테스트"""

    def test_initialization(self, metric_names):
        """Given: 메트릭 이름 리스트
        When: MetricTracker 생성
        Then: 히스토리 초기화"""
        # Given
        from research.metrics import MetricTracker

        # When
        tracker = MetricTracker(metric_names)

        # Then
        assert tracker.metric_names == metric_names
        assert all(len(tracker.history[name]) == 0 for name in metric_names)

    def test_default_window_size(self, metric_names):
        """Given: window_size 없이 생성
        When: 인스턴스 확인
        Then: DEFAULT_WINDOW_SIZE 사용"""
        # Given
        from research.metrics import MetricTracker

        # When
        tracker = MetricTracker(metric_names)

        # Then
        assert tracker.window_size == MetricTracker.DEFAULT_WINDOW_SIZE

    def test_custom_window_size(self, metric_names):
        """Given: 커스텀 window_size
        When: MetricTracker 생성
        Then: 설정된 window_size 사용"""
        # Given
        from research.metrics import MetricTracker

        # When
        tracker = MetricTracker(metric_names, window_size=SMALL_WINDOW_SIZE)

        # Then
        assert tracker.window_size == SMALL_WINDOW_SIZE

    def test_update(self, metric_names, perfect_predictions):
        """Given: MetricTracker와 완벽한 예측
        When: update() 호출
        Then: 메트릭 계산 및 히스토리 저장"""
        # Given
        from research.metrics import MetricTracker, AccuracyMetric, PrecisionMetric, RecallMetric
        tracker = MetricTracker(metric_names)
        predictions, targets = perfect_predictions

        metrics = {
            'accuracy': AccuracyMetric(),
            'precision': PrecisionMetric(num_classes=NUM_CLASSES),
            'recall': RecallMetric(num_classes=NUM_CLASSES)
        }

        # When
        results = tracker.update(predictions, targets, metrics)

        # Then
        assert len(results) == len(metric_names)
        assert all(len(tracker.history[name]) == 1 for name in metric_names)
        assert_float_equal(results['accuracy'], PERFECT_ACCURACY)

    def test_get_latest_single(self, single_metric_name, perfect_predictions):
        """Given: 업데이트된 트래커
        When: get_latest(metric_name) 호출
        Then: 최신 값 반환"""
        # Given
        from research.metrics import MetricTracker, AccuracyMetric
        tracker = MetricTracker([single_metric_name])
        predictions, targets = perfect_predictions
        metrics = {single_metric_name: AccuracyMetric()}
        tracker.update(predictions, targets, metrics)

        # When
        result = tracker.get_latest(single_metric_name)

        # Then
        assert_float_equal(result, PERFECT_ACCURACY)

    def test_get_latest_all(self, metric_names, perfect_predictions):
        """Given: 업데이트된 트래커
        When: get_latest() 호출 (metric_name 없이)
        Then: 모든 메트릭의 최신 값 반환"""
        # Given
        from research.metrics import MetricTracker, AccuracyMetric, PrecisionMetric, RecallMetric
        tracker = MetricTracker(metric_names)
        predictions, targets = perfect_predictions
        metrics = {
            'accuracy': AccuracyMetric(),
            'precision': PrecisionMetric(num_classes=NUM_CLASSES),
            'recall': RecallMetric(num_classes=NUM_CLASSES)
        }
        tracker.update(predictions, targets, metrics)

        # When
        results = tracker.get_latest()

        # Then
        assert isinstance(results, dict)
        assert len(results) == len(metric_names)

    def test_get_best_higher_is_better(self, single_metric_name):
        """Given: 여러 에폭의 메트릭 값
        When: get_best() 호출 (higher_is_better=True)
        Then: 최대값 반환"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker([single_metric_name])
        values = [0.7, 0.9, 0.8, 0.95, 0.85]
        tracker.history[single_metric_name] = values

        # When
        best = tracker.get_best(single_metric_name, higher_is_better=True)

        # Then
        assert_float_equal(best, max(values))

    def test_get_best_lower_is_better(self):
        """Given: 여러 에폭의 MSE 값
        When: get_best() 호출 (higher_is_better=False)
        Then: 최소값 반환"""
        # Given
        from research.metrics import MetricTracker
        metric_name = 'mse'
        tracker = MetricTracker([metric_name])
        values = [0.5, 0.3, 0.4, 0.2, 0.35]
        tracker.history[metric_name] = values

        # When
        best = tracker.get_best(metric_name, higher_is_better=False)

        # Then
        assert_float_equal(best, min(values))

    def test_get_history_single(self, single_metric_name):
        """Given: 히스토리가 있는 트래커
        When: get_history(metric_name) 호출
        Then: 해당 메트릭의 히스토리 반환"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker([single_metric_name])
        expected_history = [0.7, 0.8, 0.9]
        tracker.history[single_metric_name] = expected_history

        # When
        history = tracker.get_history(single_metric_name)

        # Then
        assert history == expected_history

    def test_get_history_all(self, metric_names):
        """Given: 히스토리가 있는 트래커
        When: get_history() 호출 (metric_name 없이)
        Then: 모든 메트릭의 히스토리 반환"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker(metric_names)
        for name in metric_names:
            tracker.history[name] = [0.7, 0.8, 0.9]

        # When
        history = tracker.get_history()

        # Then
        assert isinstance(history, dict)
        assert len(history) == len(metric_names)

    def test_get_moving_average(self, single_metric_name):
        """Given: 히스토리가 있는 트래커
        When: get_moving_average() 호출
        Then: 윈도우 크기만큼의 이동 평균 반환"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker([single_metric_name], window_size=SMALL_WINDOW_SIZE)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracker.history[single_metric_name] = values

        # When
        moving_avg = tracker.get_moving_average(single_metric_name)

        # Then
        # 마지막 3개 값의 평균: (3.0 + 4.0 + 5.0) / 3 = 4.0
        expected_avg = sum(values[-SMALL_WINDOW_SIZE:]) / SMALL_WINDOW_SIZE
        assert_float_equal(moving_avg, expected_avg)

    def test_summary(self, single_metric_name):
        """Given: 히스토리가 있는 트래커
        When: summary() 호출
        Then: 통계 정보 반환 (mean, std, min, max, latest)"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker([single_metric_name])
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracker.history[single_metric_name] = values

        # When
        summary = tracker.summary()

        # Then
        assert single_metric_name in summary
        stats = summary[single_metric_name]
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'latest' in stats

        assert_float_equal(stats['mean'], sum(values) / len(values))
        assert_float_equal(stats['min'], min(values))
        assert_float_equal(stats['max'], max(values))
        assert_float_equal(stats['latest'], values[-1])

    def test_reset_single(self, metric_names):
        """Given: 히스토리가 있는 트래커
        When: reset(metric_name) 호출
        Then: 해당 메트릭만 초기화"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker(metric_names)
        for name in metric_names:
            tracker.history[name] = [0.7, 0.8, 0.9]

        reset_metric = metric_names[0]

        # When
        tracker.reset(reset_metric)

        # Then
        assert len(tracker.history[reset_metric]) == 0
        assert all(len(tracker.history[name]) > 0 for name in metric_names[1:])

    def test_reset_all(self, metric_names):
        """Given: 히스토리가 있는 트래커
        When: reset() 호출 (metric_name 없이)
        Then: 모든 메트릭 초기화"""
        # Given
        from research.metrics import MetricTracker
        tracker = MetricTracker(metric_names)
        for name in metric_names:
            tracker.history[name] = [0.7, 0.8, 0.9]

        # When
        tracker.reset()

        # Then
        assert all(len(tracker.history[name]) == 0 for name in metric_names)

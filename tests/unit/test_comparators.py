"""
Comparators 테스트
TDD 방식: Given-When-Then
"""
import pytest
import numpy as np
from research.comparison.comparators import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator
)
from research.experiment.result import ExperimentResult


def create_test_result(
    model_name: str,
    params: int,
    test_accuracy: float,
    inference_time: float = 0.01,
    epoch_times: list = None
) -> ExperimentResult:
    """테스트용 ExperimentResult 생성"""
    if epoch_times is None:
        epoch_times = [1.0, 1.0, 1.0]

    return ExperimentResult(
        model_name=model_name,
        task_type="MultiClass",
        parameters=params,
        train_metrics={"accuracy": [0.8]},
        val_metrics={"accuracy": [0.75]},
        test_metrics={"accuracy": [test_accuracy]},
        train_loss=[0.5],
        val_loss=[0.6],
        test_loss=[0.35],
        epoch_times=epoch_times,
        inference_time=inference_time,
        primary_metric_name="accuracy",
        best_test_metric=test_accuracy,
        final_overfitting_gap=0.05
    )


class TestPerformanceComparator:
    """PerformanceComparator 테스트"""

    def test_성능_비교_높을수록_좋음(self):
        """
        Given: 여러 모델의 결과
        When: PerformanceComparator로 비교 (higher_better=True)
        Then: 성능이 높은 순서대로 정렬
        """
        # Given
        results = {
            "model_a": create_test_result("model_a", 1000000, 0.85),
            "model_b": create_test_result("model_b", 2000000, 0.95),
            "model_c": create_test_result("model_c", 500000, 0.90),
        }
        comparator = PerformanceComparator("accuracy", higher_better=True)

        # When
        comparison = comparator.compare(results)

        # Then
        assert comparison["best_model"] == "model_b"
        assert comparison["best_score"] == 0.95
        assert comparison["ranking"][0][0] == "model_b"
        assert comparison["ranking"][1][0] == "model_c"
        assert comparison["ranking"][2][0] == "model_a"

    def test_성능_비교_낮을수록_좋음(self):
        """
        Given: 여러 모델의 결과
        When: PerformanceComparator로 비교 (higher_better=False)
        Then: 성능이 낮은 순서대로 정렬
        """
        # Given
        results = {
            "model_a": create_test_result("model_a", 1000000, 0.85),
            "model_b": create_test_result("model_b", 2000000, 0.95),
            "model_c": create_test_result("model_c", 500000, 0.90),
        }
        comparator = PerformanceComparator("accuracy", higher_better=False)

        # When
        comparison = comparator.compare(results)

        # Then
        assert comparison["best_model"] == "model_a"
        assert comparison["best_score"] == 0.85
        assert comparison["ranking"][0][0] == "model_a"

    def test_빈_결과_처리(self):
        """
        Given: 빈 결과 딕셔너리
        When: PerformanceComparator로 비교
        Then: None 반환
        """
        # Given
        results = {}
        comparator = PerformanceComparator("accuracy")

        # When
        comparison = comparator.compare(results)

        # Then
        assert comparison["best_model"] is None
        assert comparison["best_score"] is None
        assert comparison["ranking"] == []


class TestEfficiencyComparator:
    """EfficiencyComparator 테스트"""

    def test_효율성_공식_검증(self):
        """
        Given: 특정 파라미터와 성능을 가진 모델
        When: EfficiencyComparator로 효율성 계산
        Then: 수동 계산과 일치해야 함

        공식: efficiency = performance / log₁₀(params + 1)
        """
        # Given
        params = 10_000_000
        performance = 0.95
        expected_efficiency = performance / np.log10(params + EfficiencyComparator.EPSILON)

        results = {
            "test_model": create_test_result("test_model", params, performance)
        }
        comparator = EfficiencyComparator("accuracy")

        # When
        comparison = comparator.compare(results)

        # Then
        actual_efficiency = comparison["ranking"][0][1]["efficiency"]
        assert abs(actual_efficiency - expected_efficiency) < 1e-6

    def test_효율성_비교_작은_모델이_유리(self):
        """
        Given: 비슷한 성능의 큰 모델과 작은 모델
        When: EfficiencyComparator로 비교
        Then: 작은 모델이 더 효율적으로 판단
        """
        # Given
        results = {
            "large_model": create_test_result("large_model", 100_000_000, 0.96),
            "small_model": create_test_result("small_model", 10_000_000, 0.94),
        }
        comparator = EfficiencyComparator("accuracy")

        # When
        comparison = comparator.compare(results)

        # Then
        ranking = comparison["ranking"]
        # 효율성 계산
        large_eff = 0.96 / np.log10(100_000_000 + 1)
        small_eff = 0.94 / np.log10(10_000_000 + 1)

        # 작은 모델이 더 효율적인지 확인
        if small_eff > large_eff:
            assert ranking[0][0] == "small_model"
        else:
            assert ranking[0][0] == "large_model"

    def test_효율성_비교_ResNet_예시(self):
        """
        Given: 명세서의 ResNet 예시 (06번 명세서 line 212-219)
        When: EfficiencyComparator로 비교
        Then: 명세서와 동일한 효율성 계산

        명세서 예시:
        | 모델 | 파라미터 | Accuracy | log₁₀(params+1) | Efficiency |
        | ResNet18 | 11.7M | 0.94 | 7.07 | 0.133 |
        | ResNet50 | 25.6M | 0.96 | 7.41 | 0.130 |
        """
        # Given
        results = {
            "ResNet18": create_test_result("ResNet18", 11_700_000, 0.94),
            "ResNet50": create_test_result("ResNet50", 25_600_000, 0.96),
        }
        comparator = EfficiencyComparator("accuracy")

        # When
        comparison = comparator.compare(results)

        # Then
        # ResNet18이 더 효율적이어야 함
        assert comparison["best_model"] == "ResNet18"

        # 효율성 값 검증 (오차 허용)
        resnet18_ranking = [r for r in comparison["ranking"] if r[0] == "ResNet18"][0]
        resnet50_ranking = [r for r in comparison["ranking"] if r[0] == "ResNet50"][0]

        resnet18_eff = resnet18_ranking[1]["efficiency"]
        resnet50_eff = resnet50_ranking[1]["efficiency"]

        # 명세서 예상값과 비교 (소수점 3자리)
        assert abs(resnet18_eff - 0.133) < 0.01  # 오차 허용
        assert abs(resnet50_eff - 0.130) < 0.01  # 오차 허용

    def test_EPSILON_효과(self):
        """
        Given: 파라미터가 0인 모델
        When: EfficiencyComparator로 비교
        Then: EPSILON으로 인해 log(0) 에러가 발생하지 않음
        """
        # Given
        results = {
            "zero_param_model": create_test_result("zero_param_model", 0, 0.5)
        }
        comparator = EfficiencyComparator("accuracy")

        # When
        comparison = comparator.compare(results)

        # Then
        # 에러 없이 실행되어야 함
        assert comparison["ranking"][0][1]["efficiency"] == 0

    def test_클래스_상수_사용(self):
        """
        Given: EfficiencyComparator 클래스
        When: 클래스 상수 확인
        Then: LOG_BASE와 EPSILON이 정의되어 있어야 함
        """
        # Given & When & Then
        assert hasattr(EfficiencyComparator, 'LOG_BASE')
        assert hasattr(EfficiencyComparator, 'EPSILON')
        assert EfficiencyComparator.LOG_BASE == 10
        assert EfficiencyComparator.EPSILON == 1


class TestSpeedComparator:
    """SpeedComparator 테스트"""

    def test_속도_비교_추론시간_기준(self):
        """
        Given: 다양한 추론 시간을 가진 모델들
        When: SpeedComparator로 비교
        Then: 추론 시간이 짧은 순서대로 정렬
        """
        # Given
        results = {
            "fast_model": create_test_result("fast_model", 1000000, 0.9, inference_time=0.005),
            "medium_model": create_test_result("medium_model", 2000000, 0.92, inference_time=0.010),
            "slow_model": create_test_result("slow_model", 3000000, 0.95, inference_time=0.020),
        }
        comparator = SpeedComparator()

        # When
        comparison = comparator.compare(results)

        # Then
        assert comparison["fastest_model"] == "fast_model"
        assert comparison["fastest_time"] == 0.005
        assert comparison["ranking"][0][0] == "fast_model"
        assert comparison["ranking"][1][0] == "medium_model"
        assert comparison["ranking"][2][0] == "slow_model"

    def test_평균_에폭_시간_계산(self):
        """
        Given: 에폭 시간 정보가 있는 모델
        When: SpeedComparator로 비교
        Then: 평균 에폭 시간이 계산되어야 함
        """
        # Given
        epoch_times = [1.0, 1.2, 1.1, 1.3, 1.0]
        results = {
            "test_model": create_test_result(
                "test_model",
                1000000,
                0.9,
                inference_time=0.01,
                epoch_times=epoch_times
            )
        }
        comparator = SpeedComparator()

        # When
        comparison = comparator.compare(results)

        # Then
        avg_epoch_time = comparison["ranking"][0][1]["avg_epoch_time"]
        expected_avg = np.mean(epoch_times)
        assert abs(avg_epoch_time - expected_avg) < 1e-6

    def test_빈_결과_처리(self):
        """
        Given: 빈 결과 딕셔너리
        When: SpeedComparator로 비교
        Then: None 반환
        """
        # Given
        results = {}
        comparator = SpeedComparator()

        # When
        comparison = comparator.compare(results)

        # Then
        assert comparison["fastest_model"] is None
        assert comparison["fastest_time"] is None


class TestComparatorNames:
    """Comparator 이름 테스트"""

    def test_performance_comparator_name(self):
        """
        Given: PerformanceComparator
        When: get_comparison_name 호출
        Then: 적절한 이름 반환
        """
        # Given
        comparator = PerformanceComparator("accuracy")

        # When
        name = comparator.get_comparison_name()

        # Then
        assert "Performance" in name
        assert "accuracy" in name

    def test_efficiency_comparator_name(self):
        """
        Given: EfficiencyComparator
        When: get_comparison_name 호출
        Then: 적절한 이름 반환
        """
        # Given
        comparator = EfficiencyComparator("accuracy")

        # When
        name = comparator.get_comparison_name()

        # Then
        assert "Efficiency" in name
        assert "accuracy" in name

    def test_speed_comparator_name(self):
        """
        Given: SpeedComparator
        When: get_comparison_name 호출
        Then: 적절한 이름 반환
        """
        # Given
        comparator = SpeedComparator()

        # When
        name = comparator.get_comparison_name()

        # Then
        assert "Speed" in name

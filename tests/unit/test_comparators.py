"""
Comparator Unit 테스트
"""

import pytest

from research import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,
    ComparisonManager,
)


@pytest.mark.unit
class TestPerformanceComparator:
    """PerformanceComparator 테스트"""

    def test_comparison_name(self):
        """비교 이름 테스트"""
        comparator = PerformanceComparator('Accuracy', higher_better=True)
        assert comparator.get_comparison_name() == "Performance (Accuracy)"

    def test_compare_results(self, dummy_experiment_results):
        """비교 결과 테스트"""
        comparator = PerformanceComparator('Accuracy', higher_better=True)
        result = comparator.compare(dummy_experiment_results)

        assert 'ranking' in result
        assert 'best_model' in result
        assert 'best_score' in result
        assert 'metric' in result

        assert len(result['ranking']) == 3
        assert result['metric'] == 'Accuracy'
        assert result['best_model'] is not None

    def test_ranking_order(self, dummy_experiment_results):
        """랭킹 순서 테스트"""
        comparator = PerformanceComparator('Accuracy', higher_better=True)
        result = comparator.compare(dummy_experiment_results)

        ranking = result['ranking']

        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1], "Should be sorted in descending order"


@pytest.mark.unit
class TestEfficiencyComparator:
    """EfficiencyComparator 테스트"""

    def test_comparison_name(self):
        """비교 이름 테스트"""
        comparator = EfficiencyComparator('Accuracy')
        assert comparator.get_comparison_name() == "Efficiency (Accuracy)"

    def test_compare_results(self, dummy_experiment_results):
        """비교 결과 테스트"""
        comparator = EfficiencyComparator('Accuracy')
        result = comparator.compare(dummy_experiment_results)

        assert 'ranking' in result
        assert 'best_model' in result
        assert 'metric' in result

        assert len(result['ranking']) == 3
        assert result['metric'] == 'Accuracy'

    def test_efficiency_calculation(self, dummy_experiment_results):
        """효율성 계산 테스트"""
        comparator = EfficiencyComparator('Accuracy')
        result = comparator.compare(dummy_experiment_results)

        for model_name, scores in result['ranking']:
            assert 'efficiency' in scores
            assert 'performance' in scores
            assert 'parameters' in scores

            assert scores['efficiency'] > 0
            assert scores['performance'] > 0
            assert scores['parameters'] > 0


@pytest.mark.unit
class TestSpeedComparator:
    """SpeedComparator 테스트"""

    def test_comparison_name(self):
        """비교 이름 테스트"""
        comparator = SpeedComparator()
        assert comparator.get_comparison_name() == "Speed Comparison"

    def test_compare_results(self, dummy_experiment_results):
        """비교 결과 테스트"""
        comparator = SpeedComparator()
        result = comparator.compare(dummy_experiment_results)

        assert 'ranking' in result
        assert 'fastest_model' in result
        assert 'fastest_time' in result

        assert len(result['ranking']) == 3
        assert result['fastest_model'] is not None

    def test_speed_metrics(self, dummy_experiment_results):
        """속도 메트릭 테스트"""
        comparator = SpeedComparator()
        result = comparator.compare(dummy_experiment_results)

        for model_name, speeds in result['ranking']:
            assert 'inference_time' in speeds
            assert 'avg_epoch_time' in speeds

            assert speeds['inference_time'] > 0
            assert speeds['avg_epoch_time'] > 0

    def test_ranking_order(self, dummy_experiment_results):
        """랭킹 순서 테스트 (낮을수록 좋음)"""
        comparator = SpeedComparator()
        result = comparator.compare(dummy_experiment_results)

        ranking = result['ranking']

        for i in range(len(ranking) - 1):
            assert ranking[i][1]['inference_time'] <= ranking[i + 1][1]['inference_time']


@pytest.mark.unit
class TestComparisonManager:
    """ComparisonManager 테스트"""

    def test_manager_initialization(self):
        """매니저 초기화 테스트"""
        manager = ComparisonManager()

        assert len(manager.comparators) == 0
        assert manager.comparison_results == {}

    def test_add_comparator(self):
        """비교기 추가 테스트"""
        manager = ComparisonManager()
        comparator = PerformanceComparator('Accuracy', higher_better=True)

        manager.add_comparator(comparator)

        assert len(manager.comparators) == 1

    def test_run_all_comparisons(self, dummy_experiment_results):
        """모든 비교 실행 테스트"""
        manager = ComparisonManager()

        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
        manager.add_comparator(EfficiencyComparator('Accuracy'))
        manager.add_comparator(SpeedComparator())

        results = manager.run_all_comparisons(dummy_experiment_results)

        assert len(results) == 3
        assert 'Performance (Accuracy)' in results
        assert 'Efficiency (Accuracy)' in results
        assert 'Speed Comparison' in results

    def test_get_results(self, dummy_experiment_results):
        """결과 가져오기 테스트"""
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))

        manager.run_all_comparisons(dummy_experiment_results)
        results = manager.get_results()

        assert len(results) > 0
        assert 'Performance (Accuracy)' in results

    def test_clear(self, dummy_experiment_results):
        """초기화 테스트"""
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
        manager.run_all_comparisons(dummy_experiment_results)

        manager.clear()

        assert len(manager.comparators) == 0
        assert manager.comparison_results == {}

    def test_export_report(self, dummy_experiment_results, tmp_path):
        """리포트 내보내기 테스트"""
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
        manager.run_all_comparisons(dummy_experiment_results)

        report_path = tmp_path / "test_report.txt"
        manager.export_comparison_report(save_path=str(report_path))

        assert report_path.exists()
        assert report_path.stat().st_size > 0

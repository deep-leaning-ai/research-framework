"""
ComparisonManager 단위 테스트

이 테스트는 여러 비교자를 통한 모델 비교 관리 기능을 검증합니다.
"""

import pytest
from research.comparison.manager import ComparisonManager
from research.comparison.comparators import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator
)
from research.experiment.result import ExperimentResult


class TestComparisonManager:
    """ComparisonManager 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 테스트용 실험 결과 생성
        self.results = {
            'model_a': ExperimentResult(
                model_name='model_a',
                task_type='MultiClassStrategy',
                parameters=1000000,
                train_metrics={'accuracy': [0.7, 0.8, 0.85]},
                val_metrics={'accuracy': [0.65, 0.75, 0.8]},
                test_metrics={'accuracy': [0.6, 0.7, 0.75]},
                train_loss=[0.5, 0.4, 0.3],
                val_loss=[0.6, 0.5, 0.4],
                test_loss=[0.65, 0.55, 0.45],
                epoch_times=[10.0, 9.5, 9.0],
                inference_time=0.01,
                primary_metric_name='accuracy',
                best_test_metric=0.75
            ),
            'model_b': ExperimentResult(
                model_name='model_b',
                task_type='MultiClassStrategy',
                parameters=500000,
                train_metrics={'accuracy': [0.6, 0.7, 0.75]},
                val_metrics={'accuracy': [0.55, 0.65, 0.7]},
                test_metrics={'accuracy': [0.5, 0.6, 0.65]},
                train_loss=[0.6, 0.5, 0.4],
                val_loss=[0.7, 0.6, 0.5],
                test_loss=[0.75, 0.65, 0.55],
                epoch_times=[5.0, 4.5, 4.0],
                inference_time=0.005,
                primary_metric_name='accuracy',
                best_test_metric=0.65
            ),
            'model_c': ExperimentResult(
                model_name='model_c',
                task_type='MultiClassStrategy',
                parameters=10000000,
                train_metrics={'accuracy': [0.8, 0.85, 0.9]},
                val_metrics={'accuracy': [0.75, 0.8, 0.85]},
                test_metrics={'accuracy': [0.7, 0.75, 0.8]},
                train_loss=[0.4, 0.3, 0.2],
                val_loss=[0.5, 0.4, 0.3],
                test_loss=[0.55, 0.45, 0.35],
                epoch_times=[20.0, 19.0, 18.0],
                inference_time=0.02,
                primary_metric_name='accuracy',
                best_test_metric=0.8
            )
        }

    def test_add_comparator(self):
        """
        Given: ComparisonManager 인스턴스
        When: 비교자를 추가
        Then: 비교자가 정상적으로 추가됨
        """
        # Given
        manager = ComparisonManager()

        # When
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))
        manager.add_comparator(EfficiencyComparator('accuracy'))

        # Then
        assert len(manager.comparators) == 2

    def test_compare_models_performance(self):
        """
        Given: 여러 모델 결과와 PerformanceComparator
        When: 모델 비교 수행
        Then: 성능 기준 순위가 정확함
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))

        # When
        comparison_results = manager.compare(self.results)

        # Then
        perf_result = comparison_results['Performance (accuracy)']
        rankings = perf_result['ranking']  # 'rankings'가 아닌 'ranking'

        # model_c (0.8) > model_a (0.75) > model_b (0.65)
        assert rankings[0][0] == 'model_c'  # ranking은 튜플 리스트
        assert rankings[1][0] == 'model_a'
        assert rankings[2][0] == 'model_b'
        assert perf_result['best_model'] == 'model_c'

    def test_compare_models_efficiency(self):
        """
        Given: 여러 모델 결과와 EfficiencyComparator
        When: 모델 비교 수행
        Then: 효율성 기준 순위가 정확함
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(EfficiencyComparator('accuracy'))

        # When
        comparison_results = manager.compare(self.results)

        # Then
        eff_result = comparison_results['Efficiency (accuracy)']
        rankings = eff_result['ranking']

        # 효율성은 성능/log10(params)로 계산
        # model_a가 가장 효율적일 것으로 예상 (실제 계산 결과)
        assert rankings[0][0] == 'model_a'
        assert eff_result['best_model'] == 'model_a'

    def test_compare_models_speed(self):
        """
        Given: 여러 모델 결과와 SpeedComparator
        When: 모델 비교 수행
        Then: 속도 기준 순위가 정확함
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(SpeedComparator())

        # When
        comparison_results = manager.compare(self.results)

        # Then
        speed_result = comparison_results['Speed Comparison']
        rankings = speed_result['ranking']

        # model_b (0.005) > model_a (0.01) > model_c (0.02)
        assert rankings[0][0] == 'model_b'  # ranking은 튜플 리스트
        assert rankings[1][0] == 'model_a'
        assert rankings[2][0] == 'model_c'

    def test_multiple_comparators(self):
        """
        Given: 여러 비교자가 추가된 ComparisonManager
        When: 모델 비교 수행
        Then: 모든 비교 결과가 포함됨
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))
        manager.add_comparator(EfficiencyComparator('accuracy'))
        manager.add_comparator(SpeedComparator())

        # When
        results = manager.compare(self.results)

        # Then
        assert len(results) == 3
        assert 'Performance (accuracy)' in results
        assert 'Efficiency (accuracy)' in results
        assert 'Speed Comparison' in results

    def test_generate_report(self):
        """
        Given: 비교가 완료된 ComparisonManager
        When: 리포트 생성
        Then: 리포트가 정상적으로 생성됨
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))
        manager.add_comparator(EfficiencyComparator('accuracy'))
        manager.add_comparator(SpeedComparator())

        comparison_results = manager.compare(self.results)

        # When
        report = manager.generate_report(comparison_results)

        # Then
        assert isinstance(report, str)
        assert 'Model Comparison Report' in report
        assert 'Performance (accuracy)' in report
        assert 'Efficiency (accuracy)' in report
        assert 'Speed Comparison' in report
        # 리포트에 best_model만 표시되므로 model_a, model_c만 확인
        assert 'model_a' in report or 'model_c' in report

    def test_empty_results(self):
        """
        Given: 빈 결과 딕셔너리
        When: 모델 비교 수행
        Then: 빈 결과 반환 (에러 없음)
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))

        # When
        results = manager.compare({})

        # Then
        assert isinstance(results, dict)
        # 빈 결과여도 비교자 이름은 키로 존재할 수 있음

    def test_single_model(self):
        """
        Given: 단일 모델 결과
        When: 모델 비교 수행
        Then: 단일 모델 결과가 반환됨
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))

        single_result = {'model_a': self.results['model_a']}

        # When
        comparison = manager.compare(single_result)

        # Then
        perf_result = comparison['Performance (accuracy)']
        assert len(perf_result['ranking']) == 1
        assert perf_result['ranking'][0][0] == 'model_a'

    def test_no_comparators(self):
        """
        Given: 비교자가 없는 ComparisonManager
        When: 모델 비교 수행
        Then: 빈 결과 반환
        """
        # Given
        manager = ComparisonManager()

        # When
        results = manager.compare(self.results)

        # Then
        assert results == {}

    def test_print_summary(self, capsys):
        """
        Given: 비교가 완료된 ComparisonManager
        When: print_summary() 호출
        Then: 요약이 출력됨
        """
        # Given
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy', higher_better=True))

        comparison_results = manager.compare(self.results)

        # When
        manager.print_summary(comparison_results)

        # Then
        captured = capsys.readouterr()
        assert 'Comparison Summary' in captured.out
        assert 'Performance (accuracy)' in captured.out
        assert 'Best:' in captured.out
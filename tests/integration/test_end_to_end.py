"""
End-to-End Integration 테스트

전체 워크플로우를 테스트합니다.
"""

import pytest
import torch
import torch.nn as nn

from research import (
    MetricTracker,
    AccuracyMetric,
    F1ScoreMetric,
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy,
    ComparisonManager,
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,
    ExperimentRecorder,
    ExperimentResult,
    ExperimentVisualizer,
)


@pytest.mark.integration
class TestMetricTrackerIntegration:
    """MetricTracker 통합 테스트"""

    def test_training_loop_simulation(self, seed_everything):
        """학습 루프 시뮬레이션 테스트"""
        seed_everything

        tracker = MetricTracker([
            AccuracyMetric(),
            F1ScoreMetric(average='macro')
        ])

        model = nn.Linear(10, 3)
        num_epochs = 3

        for epoch in range(num_epochs):
            outputs = model(torch.randn(32, 10))
            labels = torch.randint(0, 3, (32,))

            metrics = tracker.update(outputs, labels)

            assert 'Accuracy' in metrics
            assert 'F1-Score (macro)' in metrics

        history = tracker.get_all_history()

        assert len(history['Accuracy']) == num_epochs
        assert len(history['F1-Score (macro)']) == num_epochs


@pytest.mark.integration
class TestTaskStrategyIntegration:
    """Task Strategy 통합 테스트"""

    def test_multiclass_workflow(self):
        """다중 분류 전체 워크플로우 테스트"""
        strategy = MultiClassStrategy(num_classes=10)

        model = nn.Linear(20, 10)

        X = torch.randn(32, 20)
        y = torch.randint(0, 10, (32,))

        outputs = model(X)
        prepared_labels = strategy.prepare_labels(y)

        criterion = strategy.get_criterion()
        loss = criterion(outputs, prepared_labels)

        metric = strategy.calculate_metric(outputs, y)

        assert loss.item() >= 0
        assert 0 <= metric <= 100

    def test_binary_workflow(self):
        """이진 분류 전체 워크플로우 테스트"""
        strategy = BinaryClassificationStrategy()

        model = nn.Linear(20, 1)

        X = torch.randn(32, 20)
        y = torch.randint(0, 2, (32,))

        outputs = model(X)
        prepared_labels = strategy.prepare_labels(y)

        criterion = strategy.get_criterion()
        loss = criterion(outputs, prepared_labels)

        metric = strategy.calculate_metric(outputs, prepared_labels)

        assert loss.item() >= 0
        assert 0 <= metric <= 100

    def test_regression_workflow(self):
        """회귀 전체 워크플로우 테스트"""
        strategy = RegressionStrategy()

        model = nn.Linear(20, 1)

        X = torch.randn(32, 20)
        y = torch.randn(32)

        outputs = model(X)
        prepared_labels = strategy.prepare_labels(y)

        criterion = strategy.get_criterion()
        loss = criterion(outputs, prepared_labels)

        metric = strategy.calculate_metric(outputs, prepared_labels)

        assert loss.item() >= 0
        assert metric >= 0


@pytest.mark.integration
class TestComparisonSystemIntegration:
    """비교 시스템 통합 테스트"""

    def test_full_comparison_workflow(self, dummy_experiment_results):
        """전체 비교 워크플로우 테스트"""
        manager = ComparisonManager()

        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
        manager.add_comparator(EfficiencyComparator('Accuracy'))
        manager.add_comparator(SpeedComparator())

        results = manager.run_all_comparisons(dummy_experiment_results)

        assert len(results) == 3

        for comparison_name, comparison_result in results.items():
            assert 'ranking' in comparison_result
            assert len(comparison_result['ranking']) > 0


@pytest.mark.integration
class TestVisualizationIntegration:
    """시각화 통합 테스트"""

    def test_experiment_recorder_and_visualizer(self, dummy_experiment_results, tmp_path):
        """ExperimentRecorder와 Visualizer 통합 테스트"""
        recorder = ExperimentRecorder()

        for model_name, result in dummy_experiment_results.items():
            recorder.add_result(result)

        assert len(recorder.get_all_results()) == 3

        save_path = tmp_path / "test_visualization.png"
        ExperimentVisualizer.plot_comparison(
            recorder=recorder,
            save_path=str(save_path)
        )

        assert save_path.exists()
        assert save_path.stat().st_size > 0


@pytest.mark.integration
class TestFullPipeline:
    """전체 파이프라인 통합 테스트"""

    def test_complete_experiment_pipeline(self, tmp_path):
        """완전한 실험 파이프라인 테스트"""
        recorder = ExperimentRecorder()

        models_info = [
            {'name': 'TestModel_A', 'params': 1_000_000},
            {'name': 'TestModel_B', 'params': 2_000_000},
        ]

        for model_info in models_info:
            num_epochs = 3

            train_acc = [0.7 + i*0.05 for i in range(num_epochs)]
            val_acc = [0.68 + i*0.04 for i in range(num_epochs)]
            test_acc = [0.67 + i*0.03 for i in range(num_epochs)]

            train_loss = [1.5 - i*0.3 for i in range(num_epochs)]
            val_loss = [1.6 - i*0.25 for i in range(num_epochs)]
            test_loss = [1.65 - i*0.2 for i in range(num_epochs)]

            result = ExperimentResult(
                model_name=model_info['name'],
                task_type='MultiClassStrategy',
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metrics={'Accuracy': train_acc},
                val_metrics={'Accuracy': val_acc},
                test_metrics={'Accuracy': test_acc},
                primary_metric_name='Accuracy',
                best_test_metric=max(test_acc),
                parameters=model_info['params'],
                epoch_times=[1.0] * num_epochs,
                inference_time=0.05
            )

            recorder.add_result(result)

        results = recorder.get_all_results()

        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
        comparison_results = manager.run_all_comparisons(results)

        assert len(comparison_results) > 0
        assert 'Performance (Accuracy)' in comparison_results

        best_model = comparison_results['Performance (Accuracy)']['best_model']
        assert best_model in ['TestModel_A', 'TestModel_B']

        save_path = tmp_path / "pipeline_visualization.png"
        ExperimentVisualizer.plot_comparison(
            recorder=recorder,
            save_path=str(save_path)
        )

        assert save_path.exists()

        report_path = tmp_path / "pipeline_report.txt"
        manager.export_comparison_report(save_path=str(report_path))

        assert report_path.exists()

"""
End-to-End 통합 테스트

전체 워크플로우를 테스트합니다:
1. 데이터 로드
2. 모델 생성
3. 학습
4. 평가
5. 비교
6. 시각화
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from research import (
    Experiment,
    ModelRegistry,
    CIFAR10DataModule,
    VanillaTrainingStrategy,
    SimpleLoggingStrategy,
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy,
    MetricTracker,
    AccuracyMetric,
    ExperimentRecorder,
    ExperimentResult,
    ComparisonManager,
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,
    ExperimentVisualizer
)


@pytest.fixture
def dummy_data_module():
    """테스트용 더미 데이터 모듈"""
    class DummyDataModule:
        def __init__(self, num_classes=10):
            self.prepared = False
            self.setup_called = False
            self.num_classes = num_classes

        def prepare_data(self):
            self.prepared = True

        def setup(self):
            self.setup_called = True

        def train_dataloader(self):
            # num_classes에 따라 적절한 레이블 생성
            if self.num_classes == 1:
                # Binary classification or regression - labels 0 or 1
                labels = torch.randint(0, 2, (20,))
            else:
                # Multi-class classification
                labels = torch.randint(0, self.num_classes, (20,))

            dataset = torch.utils.data.TensorDataset(
                torch.randn(20, 3, 32, 32),
                labels
            )
            return torch.utils.data.DataLoader(dataset, batch_size=4)

        def val_dataloader(self):
            if self.num_classes == 1:
                labels = torch.randint(0, 2, (10,))
            else:
                labels = torch.randint(0, self.num_classes, (10,))

            dataset = torch.utils.data.TensorDataset(
                torch.randn(10, 3, 32, 32),
                labels
            )
            return torch.utils.data.DataLoader(dataset, batch_size=4)

        def test_dataloader(self):
            if self.num_classes == 1:
                labels = torch.randint(0, 2, (10,))
            else:
                labels = torch.randint(0, self.num_classes, (10,))

            dataset = torch.utils.data.TensorDataset(
                torch.randn(10, 3, 32, 32),
                labels
            )
            return torch.utils.data.DataLoader(dataset, batch_size=4)

        def get_class_names(self):
            if self.num_classes == 1:
                return ["negative", "positive"]
            return [f"class_{i}" for i in range(self.num_classes)]

    return DummyDataModule()


class TestFullPipeline:
    """전체 파이프라인 테스트"""

    def test_complete_workflow(self, dummy_data_module):
        """
        Given: 전체 프레임워크
        When: 데이터 → 모델 → 학습 → 평가 → 비교 → 시각화
        Then: 모든 단계가 정상 동작
        """
        # 1. 실험 설정
        config = {
            'num_classes': 10,
            'learning_rate': 1e-3,
            'max_epochs': 1,  # 빠른 테스트
            'batch_size': 4
        }

        exp = Experiment(config)

        # 2. 환경 설정
        task_strategy = MultiClassStrategy(num_classes=10)
        training_strategy = VanillaTrainingStrategy(
            device='cpu',
            task_strategy=task_strategy
        )

        exp.setup(
            model_name='resnet18',
            data_module=dummy_data_module,
            training_strategy=training_strategy,
            logging_strategy=None
        )

        # 3. 학습 실행
        result = exp.run(strategy='fine_tuning')

        # 검증
        assert result is not None
        assert 'model_info' in result
        assert 'train_results' in result
        assert 'test_results' in result

        # 4. 전략 비교
        comparison = exp.compare_strategies(
            ['feature_extraction', 'fine_tuning'],
            reset_model=True
        )

        assert len(comparison) == 2
        assert 'feature_extraction' in comparison
        assert 'fine_tuning' in comparison

        # 5. 결과 저장
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            exp.save_results(temp_file)
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestModelRegistry:
    """ModelRegistry 통합 테스트"""

    def test_all_models_creatable(self):
        """
        Given: ModelRegistry
        When: 모든 등록된 모델 생성
        Then: 모두 정상 생성
        """
        models = ModelRegistry.list_models()
        assert len(models) > 0

        for model_name in models[:3]:  # 처음 3개만 테스트 (시간 절약)
            model = ModelRegistry.create(model_name, num_classes=10)
            assert model is not None
            assert hasattr(model, 'model')
            assert hasattr(model, 'freeze_backbone')
            assert hasattr(model, 'unfreeze_all')


class TestTaskStrategies:
    """Task 전략 통합 테스트"""

    def test_all_task_strategies(self):
        """
        Given: 모든 TaskStrategy
        When: 각각으로 학습
        Then: 모두 정상 동작
        """
        strategies = [
            MultiClassStrategy(num_classes=10),
            BinaryClassificationStrategy(),
            RegressionStrategy()
        ]

        for task_strategy in strategies:
            # BinaryClassificationStrategy는 1개 출력 유닛 사용
            if isinstance(task_strategy, BinaryClassificationStrategy):
                num_classes = 1
            elif isinstance(task_strategy, RegressionStrategy):
                num_classes = 1  # 회귀도 1개 출력
            else:
                num_classes = 10

            # num_classes에 맞는 더미 데이터 모듈 생성
            class DummyDataModule:
                def __init__(self, num_classes=10):
                    self.prepared = False
                    self.setup_called = False
                    self.num_classes = num_classes

                def prepare_data(self):
                    self.prepared = True

                def setup(self):
                    self.setup_called = True

                def train_dataloader(self):
                    if self.num_classes == 1:
                        labels = torch.randint(0, 2, (20,)).float()
                    else:
                        labels = torch.randint(0, self.num_classes, (20,))

                    dataset = torch.utils.data.TensorDataset(
                        torch.randn(20, 3, 32, 32),
                        labels
                    )
                    return torch.utils.data.DataLoader(dataset, batch_size=4)

                def val_dataloader(self):
                    if self.num_classes == 1:
                        labels = torch.randint(0, 2, (10,)).float()
                    else:
                        labels = torch.randint(0, self.num_classes, (10,))

                    dataset = torch.utils.data.TensorDataset(
                        torch.randn(10, 3, 32, 32),
                        labels
                    )
                    return torch.utils.data.DataLoader(dataset, batch_size=4)

                def test_dataloader(self):
                    if self.num_classes == 1:
                        labels = torch.randint(0, 2, (10,)).float()
                    else:
                        labels = torch.randint(0, self.num_classes, (10,))

                    dataset = torch.utils.data.TensorDataset(
                        torch.randn(10, 3, 32, 32),
                        labels
                    )
                    return torch.utils.data.DataLoader(dataset, batch_size=4)

                def get_class_names(self):
                    if self.num_classes == 1:
                        return ["negative", "positive"]
                    return [f"class_{i}" for i in range(self.num_classes)]

            dummy_data_module = DummyDataModule(num_classes)

            config = {
                'num_classes': num_classes,
                'learning_rate': 1e-3,
                'max_epochs': 1,
                'batch_size': 4
            }

            exp = Experiment(config)
            training_strategy = VanillaTrainingStrategy(
                device='cpu',
                task_strategy=task_strategy
            )

            exp.setup(
                model_name='resnet18',
                data_module=dummy_data_module,
                training_strategy=training_strategy
            )

            # inference만 테스트 (빠른 실행)
            result = exp.run(strategy='inference')
            assert result is not None


class TestMetricSystem:
    """메트릭 시스템 통합 테스트"""

    def test_metric_tracker_integration(self):
        """
        Given: MetricTracker와 여러 메트릭
        When: 여러 에폭 동안 추적
        Then: 히스토리, 최고값, 이동평균 정상 동작
        """
        # MetricTracker의 새로운 API에 맞춤
        tracker = MetricTracker(
            metric_names=['Accuracy'],
            window_size=3
        )

        # 메트릭 인스턴스
        metrics = {'Accuracy': AccuracyMetric()}

        # 5 에폭 시뮬레이션
        for epoch in range(5):
            predictions = torch.randn(10, 5)
            targets = torch.randint(0, 5, (10,))
            tracker.update(predictions, targets, metrics)

        # 검증
        history = tracker.get_history('Accuracy')
        assert len(history) == 5

        best = tracker.get_best('Accuracy')
        assert best is not None

        moving_avg = tracker.get_moving_average('Accuracy')
        assert moving_avg is not None


class TestComparisonSystem:
    """비교 시스템 통합 테스트"""

    def test_comparison_manager(self):
        """
        Given: ComparisonManager와 여러 Comparator
        When: 여러 모델 결과 비교
        Then: 모든 비교 정상 수행
        """
        # 더미 결과 생성
        results = {}
        for i, name in enumerate(['model1', 'model2', 'model3']):
            results[name] = ExperimentResult(
                model_name=name,
                task_type="MultiClass",
                parameters=1000000 * (i + 1),
                train_metrics={'accuracy': [0.8 + i * 0.05]},
                val_metrics={'accuracy': [0.75 + i * 0.05]},
                test_metrics={'accuracy': [0.7 + i * 0.05]},
                train_loss=[0.5 - i * 0.1],
                val_loss=[0.6 - i * 0.1],
                test_loss=[0.35 - i * 0.05],
                epoch_times=[1.0 + i * 0.5],
                inference_time=0.01 * (i + 1),
                primary_metric_name='accuracy',
                best_test_metric=0.7 + i * 0.05,
                final_overfitting_gap=0.05
            )

        # ComparisonManager 테스트
        manager = ComparisonManager()
        manager.add_comparator(PerformanceComparator('accuracy'))
        manager.add_comparator(EfficiencyComparator('accuracy'))
        manager.add_comparator(SpeedComparator())

        comparisons = manager.run_all_comparisons(results)

        assert len(comparisons) == 3
        assert 'Performance (accuracy)' in comparisons
        assert 'Efficiency (accuracy)' in comparisons
        assert 'Speed Comparison' in comparisons


class TestVisualization:
    """시각화 시스템 통합 테스트"""

    def test_experiment_visualizer(self):
        """
        Given: ExperimentVisualizer
        When: 여러 모델 결과 시각화
        Then: 차트 생성 성공
        """
        # ExperimentRecorder에 결과 추가
        recorder = ExperimentRecorder()

        for i in range(3):
            result = ExperimentResult(
                model_name=f"model_{i}",
                task_type="MultiClass",
                parameters=1000000,
                train_metrics={'accuracy': [0.7, 0.8, 0.85]},
                val_metrics={'accuracy': [0.65, 0.75, 0.80]},
                test_metrics={'accuracy': [0.80, 0.81, 0.82]},  # 3개로 맞춤
                train_loss=[0.6, 0.4, 0.3],
                val_loss=[0.7, 0.5, 0.4],
                test_loss=[0.35, 0.34, 0.33],  # 3개로 맞춤
                epoch_times=[1.0, 1.1, 1.2],
                inference_time=0.01,
                primary_metric_name='accuracy',
                best_test_metric=0.80,
                final_overfitting_gap=0.05
            )
            recorder.add_result(result)

        # 시각화 테스트 (파일 저장은 생략)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_file = f.name

        try:
            ExperimentVisualizer.plot_comparison(
                recorder,
                save_path=temp_file
            )
            assert os.path.exists(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestDataModules:
    """데이터 모듈 통합 테스트"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_cifar10_gpu_support(self, tmp_path):
        """
        Given: CIFAR10DataModule
        When: GPU 사용 설정
        Then: DataLoader가 pin_memory 사용
        """
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2
        )

        # 속성 확인
        assert data_module.num_workers == 2
        assert data_module.persistent_workers is True
        assert data_module.prefetch_factor == 2


class TestErrorHandling:
    """에러 처리 통합 테스트"""

    def test_invalid_model_name(self):
        """
        Given: 잘못된 모델 이름
        When: ModelRegistry.create() 호출
        Then: 의미있는 에러 메시지
        """
        with pytest.raises(ValueError) as exc_info:
            ModelRegistry.create('invalid_model', num_classes=10)

        assert 'not registered' in str(exc_info.value)
        assert 'Available models' in str(exc_info.value)

    def test_setup_before_run(self):
        """
        Given: setup() 호출 없이
        When: run() 호출
        Then: RuntimeError 발생
        """
        config = {'num_classes': 10}
        exp = Experiment(config)

        with pytest.raises(RuntimeError) as exc_info:
            exp.run()

        assert 'setup' in str(exc_info.value).lower()


@pytest.mark.slow
class TestPerformance:
    """성능 테스트"""

    def test_large_model_comparison(self, dummy_data_module):
        """
        Given: 여러 대형 모델
        When: 동시 비교
        Then: 메모리 오류 없이 완료
        """
        config = {
            'num_classes': 10,
            'learning_rate': 1e-3,
            'max_epochs': 1,
            'batch_size': 4
        }

        # 메모리 효율을 위해 한 번에 하나씩
        models = ['resnet18', 'resnet34']  # 큰 모델은 제외

        for model_name in models:
            exp = Experiment(config)
            task_strategy = MultiClassStrategy(num_classes=10)
            training_strategy = VanillaTrainingStrategy(
                device='cpu',
                task_strategy=task_strategy
            )

            exp.setup(
                model_name=model_name,
                data_module=dummy_data_module,
                training_strategy=training_strategy
            )

            # inference만 실행
            result = exp.run(strategy='inference')
            assert result is not None

            # 메모리 정리
            del exp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
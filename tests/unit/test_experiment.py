"""
Experiment Facade 테스트
TDD 방식: Given-When-Then
"""
import pytest
import torch
import torch.nn as nn
from research.core.experiment import Experiment
from research.strategies.training.vanilla_strategy import VanillaTrainingStrategy


class DummyDataModule:
    """테스트용 DataModule"""

    def __init__(self):
        self.prepared = False
        self.setup_called = False

    def prepare_data(self):
        self.prepared = True

    def setup(self):
        self.setup_called = True

    def train_dataloader(self):
        # 더미 데이터로더
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32),
            torch.randint(0, 10, (10,))
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32),
            torch.randint(0, 10, (10,))
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 3, 32, 32),
            torch.randint(0, 10, (10,))
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def get_class_names(self):
        return [f"class_{i}" for i in range(10)]


class TestClassConstants:
    """클래스 상수 테스트 - 하드코딩 제거 검증"""

    def test_기본_설정값_상수_정의(self):
        """
        Given: Experiment 클래스
        When: 클래스 상수 확인
        Then: 기본 설정값이 상수로 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(Experiment, 'DEFAULT_MAX_EPOCHS')
        assert hasattr(Experiment, 'DEFAULT_LEARNING_RATE')
        assert hasattr(Experiment, 'DEFAULT_BATCH_SIZE')
        assert hasattr(Experiment, 'DEFAULT_OPTIMIZER')
        assert hasattr(Experiment, 'DEFAULT_NUM_CLASSES')
        assert hasattr(Experiment, 'DEFAULT_IN_CHANNELS')

        assert Experiment.DEFAULT_MAX_EPOCHS == 100
        assert Experiment.DEFAULT_LEARNING_RATE == 1e-4
        assert Experiment.DEFAULT_BATCH_SIZE == 32
        assert Experiment.DEFAULT_OPTIMIZER == 'adam'
        assert Experiment.DEFAULT_NUM_CLASSES == 10
        assert Experiment.DEFAULT_IN_CHANNELS == 3

    def test_freeze_전략_매핑_상수_정의(self):
        """
        Given: Experiment 클래스
        When: FREEZE_STRATEGIES 상수 확인
        Then: 전략 매핑이 상수로 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(Experiment, 'FREEZE_STRATEGIES')
        assert isinstance(Experiment.FREEZE_STRATEGIES, dict)

        expected_strategies = {
            'feature_extraction': 'freeze_backbone',
            'fine_tuning': 'unfreeze_all',
            'inference': 'freeze_all'
        }
        assert Experiment.FREEZE_STRATEGIES == expected_strategies


class TestEncapsulation:
    """캡슐화 테스트"""

    def test_model_속성_읽기_전용(self):
        """
        Given: Experiment 인스턴스
        When: model 속성 접근
        Then: 읽기는 가능하지만 내부 속성은 _model로 보호됨
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)

        # When
        model = exp.model

        # Then
        assert model is None  # 아직 setup 호출 전
        assert hasattr(exp, '_model')
        assert hasattr(exp, '_model_name')
        assert hasattr(exp, '_initial_state')

    def test_추론용_모델_복사본_반환(self):
        """
        Given: setup된 Experiment
        When: get_model_for_inference 호출
        Then: 평가 모드의 모델 복사본 반환
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        model_copy = exp.get_model_for_inference()

        # Then
        assert model_copy is not None
        assert model_copy is not exp.model  # 다른 객체
        assert not model_copy.model.training  # eval 모드

    def test_setup_없이_추론용_모델_요청시_에러(self):
        """
        Given: setup 호출 전 Experiment
        When: get_model_for_inference 호출
        Then: RuntimeError 발생
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)

        # When & Then
        with pytest.raises(RuntimeError, match="No model available"):
            exp.get_model_for_inference()


class TestStateManagement:
    """상태 관리 테스트"""

    def test_초기_상태_저장(self):
        """
        Given: Experiment setup
        When: 모델 생성
        Then: 초기 상태가 저장되어야 함
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        # When
        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # Then
        assert exp._initial_state is not None
        assert isinstance(exp._initial_state, dict)
        assert exp._model_name == 'resnet18'

    def test_모델_리셋_가중치_복원(self):
        """
        Given: setup된 Experiment
        When: _reset_model 호출
        Then: 모델이 초기 상태로 복원되어야 함
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # 초기 상태 저장
        initial_params = {
            name: param.clone()
            for name, param in exp.model.model.named_parameters()
        }

        # 파라미터 변경 (학습 시뮬레이션)
        for param in exp.model.model.parameters():
            param.data += torch.randn_like(param.data) * 0.01

        # When
        exp._reset_model()

        # Then
        # 초기 상태로 복원되었는지 확인
        for name, param in exp.model.model.named_parameters():
            assert torch.allclose(param, initial_params[name], atol=1e-6)

    def test_setup_없이_리셋시_에러(self):
        """
        Given: setup 호출 전 Experiment
        When: _reset_model 호출
        Then: RuntimeError 발생
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)

        # When & Then
        with pytest.raises(RuntimeError, match="No initial state saved"):
            exp._reset_model()

    def test_compare_strategies_모델_리셋(self):
        """
        Given: setup된 Experiment
        When: compare_strategies 호출 (reset_model=True)
        Then: 각 전략마다 모델이 초기 상태로 리셋되어야 함
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        strategies = ['feature_extraction', 'fine_tuning']
        comparison = exp.compare_strategies(strategies, reset_model=True)

        # Then
        assert len(comparison) == 2
        assert 'feature_extraction' in comparison
        assert 'fine_tuning' in comparison


class TestSetup:
    """setup 메서드 테스트"""

    def test_기본_설정값_사용(self):
        """
        Given: num_classes만 지정한 config
        When: setup 호출
        Then: 클래스 상수의 기본값이 사용되어야 함
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        # When
        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # Then
        assert exp.model is not None
        assert exp._model_name == 'resnet18'
        assert data_module.prepared
        assert data_module.setup_called

    def test_커스텀_설정값_사용(self):
        """
        Given: 커스텀 설정이 포함된 config
        When: setup 호출
        Then: 커스텀 값이 우선 적용되어야 함
        """
        # Given
        config = {
            'num_classes': 5,
            'in_channels': 1,
            'learning_rate': 0.001
        }
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        # When
        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # Then
        assert exp.model is not None
        assert exp.config['num_classes'] == 5
        assert exp.config['in_channels'] == 1


class TestRun:
    """run 메서드 테스트"""

    def test_setup_없이_run_호출시_에러(self):
        """
        Given: setup 호출 전 Experiment
        When: run 호출
        Then: RuntimeError 발생
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)

        # When & Then
        with pytest.raises(RuntimeError, match="Call setup"):
            exp.run()

    def test_잘못된_전략_에러(self):
        """
        Given: setup된 Experiment
        When: 잘못된 전략으로 run 호출
        Then: ValueError 발생
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When & Then
        with pytest.raises(ValueError, match="Unknown strategy"):
            exp.run(strategy='invalid_strategy')

    def test_inference_전략_학습_생략(self):
        """
        Given: setup된 Experiment
        When: inference 전략으로 run 호출
        Then: 학습은 생략되고 평가만 수행
        """
        # Given
        config = {'num_classes': 10}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        result = exp.run(strategy='inference')

        # Then
        assert result is not None
        assert result['strategy'] == 'inference'
        assert result['train_results'] == {}  # 학습 결과 없음
        assert 'test_results' in result


class TestFreezeStrategies:
    """Freeze 전략 테스트"""

    def test_feature_extraction_전략(self):
        """
        Given: setup된 Experiment
        When: feature_extraction 전략으로 run
        Then: backbone이 동결되어야 함
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        exp.run(strategy='feature_extraction')

        # Then
        # Backbone 파라미터가 동결되었는지 확인
        backbone_params = exp.model.get_backbone_params()
        for param in backbone_params:
            assert not param.requires_grad

    def test_fine_tuning_전략(self):
        """
        Given: setup된 Experiment
        When: fine_tuning 전략으로 run
        Then: 모든 레이어가 학습 가능해야 함
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        exp.run(strategy='fine_tuning')

        # Then
        # 모든 파라미터가 학습 가능한지 확인
        for param in exp.model.model.parameters():
            assert param.requires_grad


class TestExperimentHistory:
    """실험 히스토리 테스트"""

    def test_실험_결과_히스토리_저장(self):
        """
        Given: setup된 Experiment
        When: 여러 번 run 호출
        Then: 모든 실험이 히스토리에 저장되어야 함
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        exp.run(strategy='inference')
        exp.run(strategy='inference')

        # Then
        history = exp.get_history()
        assert len(history) == 2

    def test_최근_결과_반환(self):
        """
        Given: 여러 실험을 실행한 Experiment
        When: get_latest_result 호출
        Then: 가장 최근 실험 결과 반환
        """
        # Given
        config = {'num_classes': 10, 'max_epochs': 1}
        exp = Experiment(config)
        data_module = DummyDataModule()
        strategy = VanillaTrainingStrategy()

        exp.setup(
            model_name='resnet18',
            data_module=data_module,
            training_strategy=strategy
        )

        # When
        exp.run(strategy='inference', run_name='run1')
        result2 = exp.run(strategy='inference', run_name='run2')
        latest = exp.get_latest_result()

        # Then
        assert latest == result2
        assert latest['strategy'] == 'inference'


class TestBackwardCompatibility:
    """하위 호환성 테스트"""

    def test_이전_방식_config_사용(self):
        """
        Given: 이전 버전 방식의 config
        When: Experiment 생성
        Then: 정상 동작해야 함
        """
        # Given
        config = {
            'num_classes': 10,
            'learning_rate': 1e-3,
            'max_epochs': 10,
            'batch_size': 32
        }

        # When
        exp = Experiment(config)

        # Then
        assert exp.config == config
        assert exp.model is None

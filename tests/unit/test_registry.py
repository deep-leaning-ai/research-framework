"""
ModelRegistry 단위 테스트

이 테스트는 모델 레지스트리의 등록, 생성, 리스팅 기능을 검증합니다.
"""

import pytest
import torch
import torch.nn as nn
from research.models.pretrained.registry import ModelRegistry
from research.core.base_model import BaseModel


class TestModelRegistry:
    """ModelRegistry 테스트 클래스"""

    def setup_method(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 테스트를 위해 레지스트리 클리어 (있다면)
        if hasattr(ModelRegistry, '_models'):
            ModelRegistry._models.clear()

    def test_register_model(self):
        """
        Given: 새로운 모델 클래스
        When: ModelRegistry.register 데코레이터로 등록
        Then: 레지스트리에 모델이 등록됨
        """
        # Given: 테스트용 모델 클래스
        @ModelRegistry.register('test_model', variant='v1')
        class TestModel(BaseModel):
            def _load_pretrained(self):
                return nn.Sequential(nn.Linear(10, 10))

            def _modify_classifier(self):
                pass

            def get_backbone_params(self):
                return []

        # When: 모델 리스트 확인
        models = ModelRegistry.list_models()

        # Then: 등록된 모델 확인 (variant 없이 저장됨)
        assert 'test_model' in models

    def test_create_registered_model(self):
        """
        Given: 레지스트리에 등록된 모델
        When: create() 메서드로 모델 생성
        Then: 정상적으로 모델 인스턴스가 생성됨
        """
        # Given: 모델 등록
        @ModelRegistry.register('create_test', variant='v1')
        class CreateTestModel(BaseModel):
            def __init__(self, num_classes=10, **kwargs):
                # BaseModel.__init__ 호출 전에 num_classes 설정해야 _load_pretrained에서 사용 가능
                super().__init__()
                # 생성 후 num_classes 검증을 위해 속성으로 저장
                self._num_classes = num_classes
                # 분류기를 다시 생성하여 올바른 num_classes로 설정
                if hasattr(self, 'model') and len(self.model) > 0:
                    self.model[-1] = nn.Linear(20, num_classes)

            @property
            def num_classes(self):
                return self._num_classes if hasattr(self, '_num_classes') else 10

            @num_classes.setter
            def num_classes(self, value):
                self._num_classes = value

            def _load_pretrained(self):
                return nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, self.num_classes)
                )

            def _modify_classifier(self):
                pass

            def get_backbone_params(self):
                return self.model[0].parameters()

        # When: 모델 생성 (variant 없이)
        model = ModelRegistry.create('create_test', num_classes=5)

        # Then: 모델 검증
        assert model is not None
        assert isinstance(model, BaseModel)
        assert model.num_classes == 5

    def test_create_nonexistent_model(self):
        """
        Given: 레지스트리에 없는 모델 이름
        When: create() 메서드 호출
        Then: ValueError 발생
        """
        # Given & When & Then
        with pytest.raises((ValueError, KeyError)):
            ModelRegistry.create('nonexistent_model')

    def test_list_models(self):
        """
        Given: 여러 모델이 등록된 레지스트리
        When: list_models() 호출
        Then: 모든 등록된 모델 이름이 반환됨
        """
        # Given: 여러 모델 등록
        @ModelRegistry.register('model_a', variant='v1')
        class ModelA(BaseModel):
            def _load_pretrained(self):
                return nn.Linear(10, 10)
            def _modify_classifier(self):
                pass
            def get_backbone_params(self):
                return []

        @ModelRegistry.register('model_b', variant='v2')
        class ModelB(BaseModel):
            def _load_pretrained(self):
                return nn.Linear(10, 10)
            def _modify_classifier(self):
                pass
            def get_backbone_params(self):
                return []

        # When: 모델 리스트 조회
        models = ModelRegistry.list_models()

        # Then: 등록된 모델들 확인
        assert len(models) >= 2  # 최소 2개 이상
        # 모델 이름 확인 (variant가 이름에 포함되지 않음)
        assert 'model_a' in models
        assert 'model_b' in models

    def test_register_duplicate_model(self):
        """
        Given: 이미 등록된 모델 이름
        When: 같은 이름으로 다시 등록 시도
        Then: 마지막 등록이 유효함 (덮어쓰기)
        """
        # Given: 첫 번째 모델 등록
        @ModelRegistry.register('duplicate', variant='v1')
        class FirstModel(BaseModel):
            version = 1
            def _load_pretrained(self):
                return nn.Linear(10, 10)
            def _modify_classifier(self):
                pass
            def get_backbone_params(self):
                return []

        # When: 같은 이름으로 두 번째 모델 등록
        @ModelRegistry.register('duplicate', variant='v1')
        class SecondModel(BaseModel):
            version = 2
            def _load_pretrained(self):
                return nn.Linear(20, 20)
            def _modify_classifier(self):
                pass
            def get_backbone_params(self):
                return []

        # Then: 마지막 등록된 모델이 생성됨
        model = ModelRegistry.create('duplicate')
        assert model.version == 2

    def test_model_with_pretrained_weights(self):
        """
        Given: pretrained 가중치를 사용하는 모델
        When: 모델 생성
        Then: 가중치가 로드된 모델이 생성됨
        """
        # Given: pretrained 모델 시뮬레이션
        @ModelRegistry.register('pretrained_test', variant='v1')
        class PretrainedTestModel(BaseModel):
            def _load_pretrained(self):
                # 실제로는 torchvision.models에서 로드
                model = nn.Sequential(
                    nn.Conv2d(3, 64, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10)
                )
                # 가중치 초기화 (pretrained 시뮬레이션)
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                return model

            def _modify_classifier(self):
                # 마지막 레이어 수정
                self.model[-1] = nn.Linear(64, self.num_classes)

            def get_backbone_params(self):
                # 마지막 레이어 제외
                return [p for m in self.model[:-1] for p in m.parameters()]

        # When: 모델 생성
        model = ModelRegistry.create('pretrained_test', num_classes=100)

        # Then: 모델 구조 확인
        assert isinstance(model.model[-1], nn.Linear)
        assert model.model[-1].out_features == 100
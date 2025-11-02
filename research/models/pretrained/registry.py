"""
ModelRegistry: Factory 패턴을 사용한 모델 등록 및 생성
데코레이터를 사용하여 자동으로 모델을 등록하고, 이름으로 모델을 생성
"""
from typing import Dict, Type, Any
from ...core.base_model import BaseModel


class ModelRegistry:
    """
    모델 레지스트리 (Factory Pattern)

    사용법:
        # 모델 등록 (데코레이터 사용)
        @ModelRegistry.register('resnet50')
        class ResNetModel(BaseModel):
            ...

        # 모델 생성
        model = ModelRegistry.create('resnet50', num_classes=10)

        # 등록된 모델 목록 확인
        print(ModelRegistry.list_models())
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, **default_kwargs):
        """
        모델을 레지스트리에 등록하는 데코레이터

        Args:
            name: 모델 등록 이름
            **default_kwargs: 기본 인자 (variant 등)

        Returns:
            데코레이터 함수
        """
        def decorator(model_class: Type[BaseModel]):
            if name in cls._models:
                print(f"Warning: Model '{name}' already registered. Overwriting...")

            # 기본 kwargs를 저장
            cls._models[name] = (model_class, default_kwargs)

            # 원래 클래스를 그대로 반환 (기능 변경 없음)
            return model_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """
        등록된 모델 생성

        Args:
            name: 등록된 모델 이름
            **kwargs: 모델 생성 인자 (num_classes, pretrained 등)

        Returns:
            생성된 모델 인스턴스

        Raises:
            ValueError: 등록되지 않은 모델 이름
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{name}' not registered. "
                f"Available models: {available}"
            )

        model_class, default_kwargs = cls._models[name]

        # 기본 kwargs와 사용자 kwargs 병합 (사용자 kwargs가 우선)
        merged_kwargs = {**default_kwargs, **kwargs}

        return model_class(**merged_kwargs)

    @classmethod
    def list_models(cls) -> list:
        """
        등록된 모든 모델 이름 반환

        Returns:
            모델 이름 리스트
        """
        return sorted(cls._models.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        모델 정보 반환 (클래스, 기본 인자)

        Args:
            name: 등록된 모델 이름

        Returns:
            모델 정보 딕셔너리
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered")

        model_class, default_kwargs = cls._models[name]
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'default_kwargs': default_kwargs
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        모델이 등록되어 있는지 확인

        Args:
            name: 모델 이름

        Returns:
            등록 여부
        """
        return name in cls._models

    @classmethod
    def unregister(cls, name: str):
        """
        모델 등록 해제 (테스트용)

        Args:
            name: 모델 이름
        """
        if name in cls._models:
            del cls._models[name]
            print(f"[OK] Model '{name}' unregistered")
        else:
            print(f"Warning: Model '{name}' not found in registry")

    @classmethod
    def clear(cls):
        """모든 모델 등록 해제 (테스트용)"""
        cls._models.clear()
        print("[OK] All models cleared from registry")

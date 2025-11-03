"""
ModelRegistry: Factory + Registry 패턴을 사용한 모델 관리 시스템

데코레이터를 사용하여 자동으로 모델을 등록하고, 이름으로 모델을 생성합니다.
현재 13개의 사전학습 모델이 등록되어 있습니다.

주요 기능:
    - 데코레이터 기반 자동 모델 등록
    - 문자열 이름으로 모델 인스턴스 생성
    - 등록된 모델 목록 조회
    - 모델 존재 여부 확인

사용 예시:
    >>> from research.models.pretrained.registry import ModelRegistry
    >>>
    >>> # 모델 생성
    >>> model = ModelRegistry.create('resnet50', num_classes=10, in_channels=3)
    >>>
    >>> # 사용 가능한 모델 목록
    >>> models = ModelRegistry.list_models()
    >>> print(f"Available models: {models}")
    >>>
    >>> # 모델 등록 확인
    >>> if ModelRegistry.is_registered('resnet50'):
    >>>     print("ResNet50 is available")
"""
from typing import Dict, Type, Any
from ...core.base_model import BaseModel


class ModelRegistry:
    """
    모델 레지스트리 (Factory + Registry Pattern)

    이 클래스는 Factory 패턴과 Registry 패턴을 결합하여
    모델의 등록과 생성을 중앙에서 관리합니다.

    Attributes:
        _models (Dict[str, Type[BaseModel]]): 등록된 모델 클래스 저장소

    Class Methods:
        register(name, **kwargs): 데코레이터로 모델 등록
        create(name, **kwargs): 이름으로 모델 인스턴스 생성
        list_models(): 등록된 모든 모델 이름 반환
        is_registered(name): 모델 등록 여부 확인
        get_model_info(name): 모델 정보 반환
        unregister(name): 모델 등록 해제
        clear(): 모든 모델 등록 해제

    등록된 모델:
        - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
        - VGG: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
    """

    # Registry storage
    _models: Dict[str, Type[BaseModel]] = {}

    # Error messages - avoid hardcoding
    ERROR_NOT_REGISTERED = "Model '{name}' not registered. Available models: {available}"
    ERROR_ALREADY_REGISTERED = "Warning: Model '{name}' already registered. Overwriting..."

    # Success messages
    SUCCESS_UNREGISTER = "[OK] Model '{name}' unregistered"
    SUCCESS_CLEAR = "[OK] All models cleared from registry"

    # Warning messages
    WARNING_NOT_FOUND = "Warning: Model '{name}' not found in registry"

    # Separator for model list
    MODEL_LIST_SEPARATOR = ", "

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
                print(cls.ERROR_ALREADY_REGISTERED.format(name=name))

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
            available = cls.MODEL_LIST_SEPARATOR.join(cls._models.keys())
            raise ValueError(
                cls.ERROR_NOT_REGISTERED.format(name=name, available=available)
            )

        model_class, default_kwargs = cls._models[name]

        # 기본 kwargs와 사용자 kwargs 병합 (사용자 kwargs가 우선)
        merged_kwargs = {**default_kwargs, **kwargs}

        # variant는 레지스트리에서만 사용하므로 제거
        merged_kwargs.pop('variant', None)

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
            print(cls.SUCCESS_UNREGISTER.format(name=name))
        else:
            print(cls.WARNING_NOT_FOUND.format(name=name))

    @classmethod
    def clear(cls):
        """모든 모델 등록 해제 (테스트용)"""
        cls._models.clear()
        print(cls.SUCCESS_CLEAR)

"""
BaseModel: 전이학습 모델의 추상 베이스 클래스

Template Method 패턴을 사용하여 공통 로직은 베이스에서 처리하고,
모델별 차이는 서브클래스에서 구현합니다.

주요 기능:
    - 사전학습 모델 로드 및 분류기 수정
    - Freeze 전략: feature_extraction, fine_tuning, inference
    - 파라미터 관리: freeze/unfreeze 메서드
    - 모델 정보 제공: get_model_info()

사용 예시:
    >>> from research.models.pretrained.resnet import ResNetModel
    >>> model = ResNetModel(variant='resnet50', num_classes=10)
    >>> model.freeze_backbone()  # Feature extraction
    >>> model.unfreeze_all()     # Fine-tuning
    >>> info = model.get_model_info()  # 모델 정보 확인
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator
import torch
import torch.nn as nn


class BaseModel(ABC):
    """
    모든 전이학습 모델의 베이스 클래스

    이 클래스는 Template Method 디자인 패턴을 구현하여
    모든 전이학습 모델에 공통된 구조와 동작을 제공합니다.

    Attributes:
        num_classes (int): 출력 클래스 수
        pretrained (bool): 사전학습 가중치 사용 여부
        model (nn.Module): 실제 PyTorch 모델

    Abstract Methods:
        _load_pretrained(): 사전학습 모델 로드
        _modify_classifier(): 분류기 레이어 수정
        get_backbone_params(): 백본 파라미터 반환

    Public Methods:
        freeze_backbone(): 백본 동결 (feature extraction)
        unfreeze_all(): 모든 레이어 해제 (fine-tuning)
        freeze_all(): 모든 레이어 동결 (inference)
        partial_unfreeze(): 일부 레이어만 해제
        get_model_info(): 모델 정보 반환
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        Args:
            num_classes: 분류할 클래스 수
            pretrained: 사전학습 가중치 사용 여부
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None

        # 초기화 (Template Method 패턴)
        self._initialize_model()

    def _initialize_model(self):
        """모델 초기화 (템플릿 메서드)"""
        self.model = self._load_pretrained()
        self._modify_classifier()

    @abstractmethod
    def _load_pretrained(self) -> nn.Module:
        """
        사전학습 모델 로드 (서브클래스에서 구현)

        Returns:
            로드된 모델
        """
        pass

    @abstractmethod
    def _modify_classifier(self):
        """
        분류기 레이어를 num_classes에 맞게 수정 (서브클래스에서 구현)

        ResNet: model.fc 수정
        VGG: model.classifier[6] 수정
        EfficientNet: model.classifier[1] 수정
        """
        pass

    @abstractmethod
    def get_backbone_params(self) -> Iterator[nn.Parameter]:
        """
        백본(특성 추출기) 파라미터 반환 (서브클래스에서 구현)

        ResNet: model.fc를 제외한 모든 파라미터
        VGG: model.features의 파라미터
        EfficientNet: model.classifier를 제외한 모든 파라미터

        Returns:
            백본 파라미터 Iterator
        """
        pass

    # 공통 메서드 (모든 모델에서 재사용)

    def freeze_backbone(self):
        """Feature Extraction 전략: 백본 동결, 분류기만 학습"""
        for param in self.get_backbone_params():
            param.requires_grad = False
        print(f"[OK] Backbone frozen for feature extraction")

    def unfreeze_all(self):
        """Fine-tuning 전략: 모든 레이어 학습"""
        for param in self.model.parameters():
            param.requires_grad = True
        print(f"[OK] All layers unfrozen for fine-tuning")

    def freeze_all(self):
        """모든 레이어 동결 (추론 전용)"""
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[OK] All layers frozen for inference")

    def partial_unfreeze(self, num_layers: int):
        """
        일부 레이어만 해제 (점진적 Fine-tuning)

        Args:
            num_layers: 해제할 레이어 수 (뒤에서부터)
        """
        # 기본적으로 모두 동결
        self.freeze_all()

        # 마지막 num_layers개만 해제
        all_params = list(self.model.parameters())
        for param in all_params[-num_layers:]:
            param.requires_grad = True

        print(f"[OK] Last {num_layers} layers unfrozen")

    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 수집 (공통 로직)

        Returns:
            모델 정보 딕셔너리
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'frozen_backbone': trainable_params < total_params
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (간단한 래퍼)

        Args:
            x: 입력 텐서

        Returns:
            모델 출력
        """
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """모델을 함수처럼 호출 가능하게"""
        return self.forward(x)

    def to(self, device):
        """디바이스 이동 (GPU/CPU)"""
        self.model = self.model.to(device)
        return self

    def eval(self):
        """평가 모드"""
        self.model.eval()
        return self

    def train(self):
        """학습 모드"""
        self.model.train()
        return self

    def measure_inference_time(self, dataloader, device='cpu', num_batches=10):
        """
        모델의 추론 시간 측정

        Args:
            dataloader: 테스트 데이터 로더
            device: 실행 디바이스
            num_batches: 측정할 배치 수

        Returns:
            배치당 평균 추론 시간 (초)
        """
        import time
        import torch

        self.model.eval()
        self.model = self.model.to(device)

        # 워밍업
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _ = self.model(inputs)
                break

        # 실제 측정
        total_time = 0
        batch_count = 0

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)

                # GPU 동기화
                if torch.cuda.is_available() and device != 'cpu':
                    torch.cuda.synchronize()

                start_time = time.time()
                _ = self.model(inputs)

                # GPU 동기화
                if torch.cuda.is_available() and device != 'cpu':
                    torch.cuda.synchronize()

                total_time += time.time() - start_time
                batch_count += 1

                if batch_count >= num_batches:
                    break

        return total_time / batch_count if batch_count > 0 else 0.0

    def state_dict(self):
        """모델 가중치 반환"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """모델 가중치 로드"""
        self.model.load_state_dict(state_dict)

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"{info['model_name']}(\n"
            f"  num_classes={info['num_classes']},\n"
            f"  total_params={info['total_parameters']:,},\n"
            f"  trainable_params={info['trainable_parameters']:,},\n"
            f"  trainable_ratio={info['trainable_ratio']:.2%}\n"
            f")"
        )

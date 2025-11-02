"""
ResNetModel: ResNet 계열 모델 구현
"""
import torch.nn as nn
from torchvision import models
from .registry import ModelRegistry
from ...core.base_model import BaseModel


@ModelRegistry.register('resnet18', variant='resnet18')
@ModelRegistry.register('resnet34', variant='resnet34')
@ModelRegistry.register('resnet50', variant='resnet50')
@ModelRegistry.register('resnet101', variant='resnet101')
@ModelRegistry.register('resnet152', variant='resnet152')
class ResNetModel(BaseModel):
    """
    ResNet 모델 클래스

    지원 모델:
    - resnet18, resnet34, resnet50, resnet101, resnet152

    사용법:
        # Registry로 생성
        model = ModelRegistry.create('resnet50', num_classes=10)

        # 직접 생성
        model = ResNetModel(variant='resnet50', num_classes=10)

        # Feature Extraction
        model.freeze_backbone()

        # Fine-tuning
        model.unfreeze_all()
    """

    def __init__(self, variant: str = 'resnet50', num_classes: int = 10, pretrained: bool = True, in_channels: int = 3):
        """
        Args:
            variant: ResNet 변형 (resnet18, resnet34, resnet50, resnet101, resnet152)
            num_classes: 분류할 클래스 수
            pretrained: 사전학습 가중치 사용 여부
            in_channels: 입력 채널 수 (기본값 3=RGB, 1=Grayscale/Mel-Spectrogram 등)
        """
        self.variant = variant
        self.in_channels = in_channels
        super().__init__(num_classes=num_classes, pretrained=pretrained)

    def _load_pretrained(self) -> nn.Module:
        """사전학습 ResNet 모델 로드"""

        # Variant에 따라 모델 선택
        model_dict = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
        }

        if self.variant not in model_dict:
            raise ValueError(
                f"Unknown ResNet variant: {self.variant}. "
                f"Available: {list(model_dict.keys())}"
            )

        model_fn, weights = model_dict[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
            print(f"Loaded pretrained {self.variant} (ImageNet)")
        else:
            model = model_fn(weights=None)
            print(f"Initialized {self.variant} (random weights)")

        # 입력 채널 수가 3이 아닐 경우 conv1 레이어 수정
        if self.in_channels != 3:
            import torch
            original_conv1_weight = model.conv1.weight.data.clone()

            # 새로운 conv1 레이어 생성
            model.conv1 = nn.Conv2d(
                self.in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

            # Pretrained weights 재사용 (가능한 경우)
            if self.pretrained and self.in_channels == 1:
                # RGB 3채널을 평균 내서 1채널로 변환
                model.conv1.weight.data = original_conv1_weight.mean(dim=1, keepdim=True)
                print(f"Conv1 modified: {self.in_channels} channel input (pretrained weights averaged)")
            else:
                print(f"Conv1 modified: {self.in_channels} channel input (random weights)")

        return model

    def _modify_classifier(self):
        """분류기를 num_classes에 맞게 수정"""
        # ResNet의 마지막 레이어는 'fc'
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

        print(f"[OK] Classifier modified: fc({in_features} → {self.num_classes})")

    def get_backbone_params(self):
        """백본 파라미터 반환 (fc 제외)"""
        # fc를 제외한 모든 파라미터
        backbone_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                backbone_params.append(param)

        return iter(backbone_params)

    def get_layer_groups(self):
        """
        레이어 그룹 반환 (점진적 Fine-tuning용)

        Returns:
            dict: 레이어 그룹 딕셔너리
        """
        return {
            'conv1': self.model.conv1,
            'bn1': self.model.bn1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4,
            'fc': self.model.fc
        }

    def freeze_until_layer(self, layer_name: str):
        """
        특정 레이어까지 동결

        Args:
            layer_name: 동결 마지막 레이어 ('layer1', 'layer2', 'layer3', 'layer4')
        """
        layer_order = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']

        if layer_name not in layer_order:
            raise ValueError(f"Invalid layer name. Choose from: {layer_order}")

        # 모두 동결
        self.freeze_all()

        # layer_name 이후는 해제
        freeze_until_idx = layer_order.index(layer_name)
        layer_groups = self.get_layer_groups()

        for layer in layer_order[freeze_until_idx + 1:]:
            if layer in layer_groups:
                for param in layer_groups[layer].parameters():
                    param.requires_grad = True

        # fc는 항상 해제
        for param in self.model.fc.parameters():
            param.requires_grad = True

        print(f"[OK] Frozen until {layer_name}, rest unfrozen")

    def __repr__(self):
        info = self.get_model_info()
        return (
            f"ResNetModel(\n"
            f"  variant='{self.variant}',\n"
            f"  num_classes={info['num_classes']},\n"
            f"  total_params={info['total_parameters']:,},\n"
            f"  trainable_params={info['trainable_parameters']:,},\n"
            f"  trainable_ratio={info['trainable_ratio']:.2%}\n"
            f")"
        )

"""
VGGModel: VGG 계열 모델 구현
"""
import torch.nn as nn
from torchvision import models
from .registry import ModelRegistry
from ...core.base_model import BaseModel


@ModelRegistry.register('vgg11', variant='vgg11')
@ModelRegistry.register('vgg11_bn', variant='vgg11_bn')
@ModelRegistry.register('vgg13', variant='vgg13')
@ModelRegistry.register('vgg13_bn', variant='vgg13_bn')
@ModelRegistry.register('vgg16', variant='vgg16')
@ModelRegistry.register('vgg16_bn', variant='vgg16_bn')
@ModelRegistry.register('vgg19', variant='vgg19')
@ModelRegistry.register('vgg19_bn', variant='vgg19_bn')
class VGGModel(BaseModel):
    """
    VGG 모델 클래스

    지원 모델:
    - vgg11, vgg11_bn, vgg13, vgg13_bn
    - vgg16, vgg16_bn, vgg19, vgg19_bn

    사용법:
        # Registry로 생성
        model = ModelRegistry.create('vgg16', num_classes=10)

        # 직접 생성
        model = VGGModel(variant='vgg16', num_classes=10)

        # Feature Extraction (features 동결)
        model.freeze_backbone()

        # Fine-tuning (전체 학습)
        model.unfreeze_all()
    """

    # Model variants - avoid hardcoding
    SUPPORTED_VARIANTS = [
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
        'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ]
    DEFAULT_VARIANT = 'vgg16'

    # Input configuration
    DEFAULT_NUM_CLASSES = 10
    DEFAULT_IN_CHANNELS = 3
    VALID_IN_CHANNELS = [1, 3, 4]

    # Architecture constants
    CLASSIFIER_FC_INDEX = 6  # classifier[6]이 마지막 Linear
    FEATURES_FIRST_CONV_INDEX = 0

    # Block configuration
    MIN_BLOCKS = 1
    MAX_BLOCKS = 5

    # Channel dimension for averaging RGB to grayscale
    RGB_TO_GRAY_DIM = 1
    KEEPDIM_TRUE = True
    CONV_BIAS = False

    def __init__(self, variant: str = None, num_classes: int = None, pretrained: bool = True, in_channels: int = None):
        """
        Args:
            variant: VGG 변형 (vgg11, vgg13, vgg16, vgg19, *_bn)
            num_classes: 분류할 클래스 수
            pretrained: 사전학습 가중치 사용 여부
            in_channels: 입력 채널 수 (기본값 3=RGB, 1=Grayscale/Mel-Spectrogram 등)
        """
        self.variant = variant or self.DEFAULT_VARIANT
        self.in_channels = in_channels or self.DEFAULT_IN_CHANNELS
        num_classes = num_classes or self.DEFAULT_NUM_CLASSES
        super().__init__(num_classes=num_classes, pretrained=pretrained)

    def _load_pretrained(self) -> nn.Module:
        """사전학습 VGG 모델 로드"""

        # Variant에 따라 모델 선택
        model_dict = {
            'vgg11': (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
            'vgg11_bn': (models.vgg11_bn, models.VGG11_BN_Weights.IMAGENET1K_V1),
            'vgg13': (models.vgg13, models.VGG13_Weights.IMAGENET1K_V1),
            'vgg13_bn': (models.vgg13_bn, models.VGG13_BN_Weights.IMAGENET1K_V1),
            'vgg16': (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
            'vgg16_bn': (models.vgg16_bn, models.VGG16_BN_Weights.IMAGENET1K_V1),
            'vgg19': (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1),
            'vgg19_bn': (models.vgg19_bn, models.VGG19_BN_Weights.IMAGENET1K_V1),
        }

        if self.variant not in model_dict:
            raise ValueError(
                f"Unknown VGG variant: {self.variant}. "
                f"Available: {list(model_dict.keys())}"
            )

        model_fn, weights = model_dict[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
            print(f"Loaded pretrained {self.variant} (ImageNet)")
        else:
            model = model_fn(weights=None)
            print(f"Initialized {self.variant} (random weights)")

        # 입력 채널 수가 DEFAULT_IN_CHANNELS이 아닐 경우 첫 번째 conv 레이어 수정
        if self.in_channels != self.DEFAULT_IN_CHANNELS:
            import torch
            # VGG의 첫 번째 레이어는 features[FEATURES_FIRST_CONV_INDEX]
            original_conv1 = model.features[self.FEATURES_FIRST_CONV_INDEX]
            original_weight = original_conv1.weight.data.clone()

            # 새로운 첫 번째 conv 레이어 생성
            model.features[self.FEATURES_FIRST_CONV_INDEX] = nn.Conv2d(
                self.in_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=self.CONV_BIAS
            )

            # Pretrained weights 재사용 (가능한 경우)
            if self.pretrained and self.in_channels == self.VALID_IN_CHANNELS[0]:  # 1 channel
                # RGB 3채널을 평균 내서 1채널로 변환
                model.features[self.FEATURES_FIRST_CONV_INDEX].weight.data = original_weight.mean(
                    dim=self.RGB_TO_GRAY_DIM, keepdim=self.KEEPDIM_TRUE
                )
                print(f"First conv modified: {self.in_channels} channel input (pretrained weights averaged)")
            else:
                print(f"First conv modified: {self.in_channels} channel input (random weights)")

        return model

    def _modify_classifier(self):
        """분류기를 num_classes에 맞게 수정"""
        # VGG의 classifier는 Sequential이고, 마지막 레이어는 인덱스 CLASSIFIER_FC_INDEX
        # classifier: [0] Linear(25088, 4096), [1] ReLU, [2] Dropout,
        #            [3] Linear(4096, 4096), [4] ReLU, [5] Dropout,
        #            [6] Linear(4096, 1000) <- 이걸 수정

        in_features = self.model.classifier[self.CLASSIFIER_FC_INDEX].in_features
        self.model.classifier[self.CLASSIFIER_FC_INDEX] = nn.Linear(in_features, self.num_classes)

        print(f"[OK] Classifier modified: classifier[{self.CLASSIFIER_FC_INDEX}]({in_features} → {self.num_classes})")

    def get_backbone_params(self):
        """백본 파라미터 반환 (features 부분)"""
        # VGG는 features와 classifier로 명확히 구분됨
        return iter(self.model.features.parameters())

    def get_layer_groups(self):
        """
        레이어 그룹 반환 (점진적 Fine-tuning용)

        VGG는 Conv block으로 구성:
        - Block 1: conv1-1, conv1-2
        - Block 2: conv2-1, conv2-2
        - Block 3: conv3-1, conv3-2, conv3-3 (vgg16, vgg19)
        - Block 4: conv4-1, conv4-2, conv4-3
        - Block 5: conv5-1, conv5-2, conv5-3
        """
        return {
            'features': self.model.features,
            'avgpool': self.model.avgpool,
            'classifier': self.model.classifier
        }

    def freeze_features(self):
        """Features 부분만 동결 (Feature Extraction 전략)"""
        for param in self.model.features.parameters():
            param.requires_grad = False

        print("[OK] Features frozen (Feature Extraction mode)")

    def unfreeze_classifier_only(self):
        """Classifier만 해제 (나머지는 동결)"""
        self.freeze_all()

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print("[OK] Only classifier unfrozen")

    def partial_unfreeze_features(self, num_blocks: int):
        """
        Features의 마지막 N개 블록만 해제

        Args:
            num_blocks: 해제할 블록 수 (MIN_BLOCKS-MAX_BLOCKS)
        """
        # VGG16 features는 총 31개 레이어 (MaxPool 포함)
        # Block 구분: MaxPool로 구분 가능

        if not self.MIN_BLOCKS <= num_blocks <= self.MAX_BLOCKS:
            raise ValueError(f"num_blocks must be {self.MIN_BLOCKS}-{self.MAX_BLOCKS}, got {num_blocks}")

        # 모두 동결
        self.freeze_all()

        # features의 레이어를 block 단위로 그룹화
        # VGG16 기준:
        # Block 1: 0-4 (conv, relu, conv, relu, maxpool)
        # Block 2: 5-9
        # Block 3: 10-16
        # Block 4: 17-23
        # Block 5: 24-30

        block_ranges = []
        maxpool_indices = []

        for i, layer in enumerate(self.model.features):
            if isinstance(layer, nn.MaxPool2d):
                maxpool_indices.append(i)

        # Block 범위 계산
        start = 0
        for idx in maxpool_indices:
            block_ranges.append((start, idx + 1))
            start = idx + 1

        # 마지막 num_blocks개만 해제
        for start, end in block_ranges[-num_blocks:]:
            for i in range(start, end):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True

        # Classifier는 항상 해제
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print(f"[OK] Last {num_blocks} feature blocks unfrozen")

    def get_architecture_info(self):
        """VGG 아키텍처 정보 반환"""
        num_conv_layers = sum(1 for m in self.model.features if isinstance(m, nn.Conv2d))
        num_fc_layers = sum(1 for m in self.model.classifier if isinstance(m, nn.Linear))

        return {
            'variant': self.variant,
            'num_conv_layers': num_conv_layers,
            'num_fc_layers': num_fc_layers,
            'has_batch_norm': 'bn' in self.variant,
            'total_depth': num_conv_layers + num_fc_layers
        }

    def __repr__(self):
        info = self.get_model_info()
        arch_info = self.get_architecture_info()

        return (
            f"VGGModel(\n"
            f"  variant='{self.variant}',\n"
            f"  num_classes={info['num_classes']},\n"
            f"  depth={arch_info['total_depth']} ({arch_info['num_conv_layers']} conv + {arch_info['num_fc_layers']} fc),\n"
            f"  total_params={info['total_parameters']:,},\n"
            f"  trainable_params={info['trainable_parameters']:,},\n"
            f"  trainable_ratio={info['trainable_ratio']:.2%}\n"
            f")"
        )

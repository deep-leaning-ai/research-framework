# Model 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [BaseModel 인터페이스](#2-basemodel-인터페이스)
3. [ResNet 모델](#3-resnet-모델)
4. [VGG 모델](#4-vgg-모델)
5. [Simple 모델](#5-simple-모델)
6. [ModelRegistry](#6-modelregistry)
7. [API 명세](#7-api-명세)
8. [테스트 요구사항](#8-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Model 시스템은 전이학습(Transfer Learning) 및 간단한 커스텀 모델을 위한 통합 인터페이스를 제공합니다. Factory Pattern과 Template Method Pattern을 사용하여 모델 생성 및 학습 전략을 체계적으로 관리합니다.

### 1.2 설계 원칙

- **Template Method Pattern**: BaseModel에서 공통 로직 제공
- **Factory Pattern**: ModelRegistry를 통한 모델 생성
- **개방-폐쇄 원칙**: 새로운 모델 추가 시 기존 코드 수정 불필요
- **하드코딩 지양**: 모든 상수는 클래스 상수로 정의

### 1.3 파일 구조

```
research/models/
├── pretrained/
│   ├── __init__.py
│   ├── registry.py          # ModelRegistry (Factory)
│   ├── resnet.py            # ResNetModel (구현 완료, 검증 필요)
│   └── vgg.py               # VGGModel (구현 완료, 검증 필요)
└── simple/
    ├── __init__.py
    ├── base.py              # Simple models용 BaseModel
    ├── cnn.py               # CNN (구현 완료, 검증 필요)
    └── fully_connected.py   # FullyConnectedNN (구현 완료, 검증 필요)
```

---

## 2. BaseModel 인터페이스

### 2.1 추상 클래스

이미 `research/core/base_model.py`에 구현되어 있음.

```python
class BaseModel(ABC):
    """모든 전이학습 모델의 베이스 클래스"""

    @abstractmethod
    def _load_pretrained(self) -> nn.Module:
        """사전학습 모델 로드"""
        pass

    @abstractmethod
    def _modify_classifier(self):
        """분류기 레이어 수정"""
        pass

    @abstractmethod
    def get_backbone_params(self) -> Iterator[nn.Parameter]:
        """백본 파라미터 반환"""
        pass

    # 공통 메서드
    def freeze_backbone(self):
        """백본 동결 (Feature Extraction)"""
        pass

    def unfreeze_all(self):
        """전체 해제 (Fine-tuning)"""
        pass

    def freeze_all(self):
        """전체 동결 (Inference)"""
        pass
```

---

## 3. ResNet 모델

### 3.1 ResNetModel

**용도**: 전이학습을 위한 ResNet 계열 모델

**지원 변형**:
- resnet18 (11.7M params)
- resnet34 (21.8M params)
- resnet50 (25.6M params)
- resnet101 (44.5M params)
- resnet152 (60.2M params)

**특징**:
- ImageNet 사전학습 가중치 사용
- 1채널 입력 지원 (grayscale, mel-spectrogram)
- 백본 동결/해제 지원
- 점진적 Fine-tuning 지원

### 3.2 클래스 상수

```python
class ResNetModel(BaseModel):
    # Model variants
    SUPPORTED_VARIANTS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    DEFAULT_VARIANT = 'resnet50'

    # Input configuration
    DEFAULT_IN_CHANNELS = 3
    VALID_IN_CHANNELS = [1, 3, 4]

    # Architecture constants
    CONV1_KERNEL_SIZE = 7
    CONV1_STRIDE = 2
    CONV1_PADDING = 3
    CONV1_OUT_CHANNELS = 64

    # Layer names
    BACKBONE_PREFIX = 'fc'
    LAYER_GROUPS = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
```

### 3.3 주요 메서드

```python
def __init__(
    self,
    variant: str = DEFAULT_VARIANT,
    num_classes: int = 10,
    pretrained: bool = True,
    in_channels: int = DEFAULT_IN_CHANNELS
):
    """ResNet 모델 초기화

    Args:
        variant: ResNet 변형
        num_classes: 분류 클래스 수
        pretrained: 사전학습 가중치 사용 여부
        in_channels: 입력 채널 수
    """
    pass

def _load_pretrained(self) -> nn.Module:
    """사전학습 ResNet 모델 로드

    - variant에 따라 적절한 모델 선택
    - in_channels != 3이면 conv1 수정
    - pretrained=True이면 ImageNet 가중치 로드
    """
    pass

def _modify_classifier(self):
    """분류기를 num_classes에 맞게 수정

    - ResNet의 fc 레이어 수정
    """
    pass

def get_backbone_params(self) -> Iterator[nn.Parameter]:
    """백본 파라미터 반환 (fc 제외)"""
    pass

def freeze_until_layer(self, layer_name: str):
    """특정 레이어까지 동결

    Args:
        layer_name: 'layer1', 'layer2', 'layer3', 'layer4'
    """
    pass
```

---

## 4. VGG 모델

### 4.1 VGGModel

**용도**: 전이학습을 위한 VGG 계열 모델

**지원 변형**:
- vgg11, vgg11_bn (132.9M params)
- vgg13, vgg13_bn (133.0M params)
- vgg16, vgg16_bn (138.4M params)
- vgg19, vgg19_bn (143.7M params)

**특징**:
- ImageNet 사전학습 가중치 사용
- Batch Normalization 변형 지원
- Features와 Classifier 명확히 분리
- Block 단위 Fine-tuning 지원

### 4.2 클래스 상수

```python
class VGGModel(BaseModel):
    # Model variants
    SUPPORTED_VARIANTS = [
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
        'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
    ]
    DEFAULT_VARIANT = 'vgg16'

    # Input configuration
    DEFAULT_IN_CHANNELS = 3
    VALID_IN_CHANNELS = [1, 3, 4]

    # Architecture constants
    CLASSIFIER_FC_INDEX = 6  # classifier[6]이 마지막 Linear
    FEATURES_FIRST_CONV_INDEX = 0

    # Block configuration
    MIN_BLOCKS = 1
    MAX_BLOCKS = 5
```

### 4.3 주요 메서드

```python
def __init__(
    self,
    variant: str = DEFAULT_VARIANT,
    num_classes: int = 10,
    pretrained: bool = True,
    in_channels: int = DEFAULT_IN_CHANNELS
):
    """VGG 모델 초기화"""
    pass

def _load_pretrained(self) -> nn.Module:
    """사전학습 VGG 모델 로드"""
    pass

def _modify_classifier(self):
    """분류기를 num_classes에 맞게 수정

    - VGG의 classifier[6] 수정
    """
    pass

def get_backbone_params(self) -> Iterator[nn.Parameter]:
    """백본 파라미터 반환 (features 부분)"""
    pass

def partial_unfreeze_features(self, num_blocks: int):
    """Features의 마지막 N개 블록만 해제

    Args:
        num_blocks: 1-5
    """
    pass
```

---

## 5. Simple 모델

### 5.1 CNN

**용도**: 간단한 이미지 분류 (MNIST 등)

**아키텍처**:
- Conv1: 1 → 16 channels
- Conv2: 16 → 32 channels
- FC: 1568 → num_classes

**클래스 상수**:

```python
class CNN(BaseModel):
    # Architecture constants
    INPUT_CHANNELS = 1
    CONV1_OUT_CHANNELS = 16
    CONV2_OUT_CHANNELS = 32

    KERNEL_SIZE = 5
    STRIDE = 1
    PADDING = 2
    POOL_SIZE = 2

    # MNIST input: 28x28
    INPUT_SIZE = 28
    FC_INPUT_SIZE = 32 * 7 * 7  # 1568

    DEFAULT_OUTPUT_DIM = 10
```

### 5.2 FullyConnectedNN

**용도**: 간단한 완전연결 네트워크

**아키텍처**:
- FC1: 784 → hidden_size
- FC2: hidden_size → hidden_size
- FC3: hidden_size → num_classes

**클래스 상수**:

```python
class FullyConnectedNN(BaseModel):
    # Architecture constants
    INPUT_SIZE = 784  # 28x28
    DEFAULT_HIDDEN_SIZE = 128
    DEFAULT_OUTPUT_DIM = 10

    # Layer configuration
    NUM_HIDDEN_LAYERS = 2
```

---

## 6. ModelRegistry

### 6.1 개요

Factory Pattern을 사용한 모델 등록 및 생성 시스템.

이미 `research/models/pretrained/registry.py`에 구현되어 있음.

### 6.2 클래스 상수

```python
class ModelRegistry:
    # Registry storage
    _models: Dict[str, Type[BaseModel]] = {}

    # Error messages
    ERROR_NOT_REGISTERED = "Model '{name}' not registered. Available models: {available}"
    ERROR_ALREADY_REGISTERED = "Warning: Model '{name}' already registered. Overwriting..."

    # Success messages
    SUCCESS_UNREGISTER = "[OK] Model '{name}' unregistered"
    SUCCESS_CLEAR = "[OK] All models cleared from registry"

    # Warning messages
    WARNING_NOT_FOUND = "Warning: Model '{name}' not found in registry"
```

### 6.3 주요 메서드

```python
@classmethod
def register(cls, name: str, **default_kwargs):
    """모델을 레지스트리에 등록하는 데코레이터"""
    pass

@classmethod
def create(cls, name: str, **kwargs) -> BaseModel:
    """등록된 모델 생성"""
    pass

@classmethod
def list_models(cls) -> list:
    """등록된 모든 모델 이름 반환"""
    pass

@classmethod
def is_registered(cls, name: str) -> bool:
    """모델이 등록되어 있는지 확인"""
    pass
```

---

## 7. API 명세

### 7.1 Import

```python
from research.models.pretrained import ModelRegistry, ResNetModel, VGGModel
from research.models.simple import CNN, FullyConnectedNN
```

### 7.2 사용 예제

```python
# Registry를 통한 생성 (권장)
model = ModelRegistry.create('resnet50', num_classes=10, in_channels=3)

# 직접 생성
model = ResNetModel(variant='resnet50', num_classes=10, in_channels=1)

# Feature Extraction 전략
model.freeze_backbone()

# Fine-tuning 전략
model.unfreeze_all()

# 모델 정보
info = model.get_model_info()
print(info['total_parameters'])

# 등록된 모델 목록
available_models = ModelRegistry.list_models()
```

---

## 8. 테스트 요구사항

### 8.1 테스트 원칙

모든 테스트는 **Given-When-Then** 패턴을 따릅니다.

### 8.2 ResNetModel 테스트

```python
def test_resnet_model_creation():
    """Given: ResNet18 variant
    When: ResNetModel 생성
    Then: 모델 정상 생성 및 num_classes 확인"""

def test_resnet_freeze_backbone():
    """Given: ResNet 모델
    When: freeze_backbone() 호출
    Then: 백본 파라미터 requires_grad=False"""

def test_resnet_1_channel_input():
    """Given: in_channels=1
    When: ResNet 모델 생성
    Then: conv1 입력 채널이 1로 수정됨"""

def test_resnet_forward_pass():
    """Given: ResNet 모델과 입력 텐서
    When: forward() 호출
    Then: 올바른 출력 shape 반환"""

def test_resnet_all_variants():
    """Given: 모든 ResNet 변형
    When: 각 변형으로 모델 생성
    Then: 모든 변형이 정상 동작"""
```

### 8.3 VGGModel 테스트

```python
def test_vgg_model_creation():
    """Given: VGG16 variant
    When: VGGModel 생성
    Then: 모델 정상 생성"""

def test_vgg_freeze_features():
    """Given: VGG 모델
    When: freeze_backbone() 호출
    Then: features 파라미터 requires_grad=False"""

def test_vgg_partial_unfreeze():
    """Given: VGG 모델
    When: partial_unfreeze_features(2) 호출
    Then: 마지막 2개 블록만 requires_grad=True"""

def test_vgg_all_variants():
    """Given: 모든 VGG 변형
    When: 각 변형으로 모델 생성
    Then: 모든 변형이 정상 동작"""
```

### 8.4 Simple Models 테스트

```python
def test_cnn_creation():
    """Given: CNN 파라미터
    When: CNN 생성
    Then: 모델 정상 생성"""

def test_cnn_forward_pass():
    """Given: CNN과 MNIST 형태 입력
    When: forward() 호출
    Then: 올바른 출력 shape"""

def test_fc_creation():
    """Given: FullyConnectedNN 파라미터
    When: FullyConnectedNN 생성
    Then: 모델 정상 생성"""

def test_fc_forward_pass():
    """Given: FullyConnectedNN과 입력
    When: forward() 호출
    Then: 올바른 출력 shape"""
```

### 8.5 ModelRegistry 테스트

```python
def test_registry_create_resnet():
    """Given: 'resnet18' 등록됨
    When: ModelRegistry.create('resnet18') 호출
    Then: ResNetModel 인스턴스 반환"""

def test_registry_create_vgg():
    """Given: 'vgg16' 등록됨
    When: ModelRegistry.create('vgg16') 호출
    Then: VGGModel 인스턴스 반환"""

def test_registry_list_models():
    """Given: 여러 모델 등록됨
    When: list_models() 호출
    Then: 정렬된 모델 이름 리스트 반환"""

def test_registry_unregistered_model():
    """Given: 등록되지 않은 모델
    When: create('unknown') 호출
    Then: ValueError 발생"""

def test_registry_is_registered():
    """Given: 등록된/미등록 모델
    When: is_registered() 호출
    Then: 올바른 boolean 반환"""
```

### 8.6 통합 테스트

```python
def test_all_models_have_same_interface():
    """Given: 모든 모델 클래스
    When: 인터페이스 확인
    Then: freeze_backbone, unfreeze_all 등 공통 메서드 존재"""

def test_model_info_format():
    """Given: 모든 모델
    When: get_model_info() 호출
    Then: 일관된 딕셔너리 형식 반환"""
```

---

## 9. 구현 체크리스트

### Phase 3-A: ResNet 검증
- [ ] ResNet 모델 생성 테스트
- [ ] ResNet freeze/unfreeze 테스트
- [ ] ResNet 1채널 입력 테스트
- [ ] ResNet forward pass 테스트
- [ ] ResNet 모든 변형 테스트
- [ ] 클래스 상수 추가

### Phase 3-B: VGG 검증
- [ ] VGG 모델 생성 테스트
- [ ] VGG freeze/unfreeze 테스트
- [ ] VGG partial unfreeze 테스트
- [ ] VGG forward pass 테스트
- [ ] VGG 모든 변형 테스트
- [ ] 클래스 상수 추가

### Phase 3-C: Simple Models 검증
- [ ] CNN 생성 및 forward 테스트
- [ ] FullyConnectedNN 생성 및 forward 테스트
- [ ] 클래스 상수 추가

### Phase 3-D: ModelRegistry 검증
- [ ] Registry create 테스트
- [ ] Registry list_models 테스트
- [ ] Registry 에러 처리 테스트
- [ ] 클래스 상수 추가

### Phase 3-E: 통합 테스트
- [ ] 모든 모델 인터페이스 일관성 테스트
- [ ] 모든 테스트 통과

---

## 10. 성공 기준

### 10.1 기능적 요구사항
- 모든 ResNet 변형 (5개) 정상 동작
- 모든 VGG 변형 (8개) 정상 동작
- Simple 모델 (2개) 정상 동작
- ModelRegistry를 통한 모델 생성 성공
- 1채널 입력 지원

### 10.2 비기능적 요구사항
- 하드코딩 없음 (클래스 상수 사용)
- 테스트 커버리지 90% 이상
- Given-When-Then 패턴 준수
- 모든 모델이 BaseModel 인터페이스 준수

---

**문서 버전**: 1.0.0
**작성일**: 2025-11-03
**이전 문서**: 02_task_strategies_spec.md
**다음 문서**: 04_training_spec.md (예정)

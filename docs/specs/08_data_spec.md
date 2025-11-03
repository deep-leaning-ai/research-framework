# Data 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [CIFAR10DataModule](#2-cifar10datamodule)
3. [Transform 파이프라인](#3-transform-파이프라인)
4. [DataLoaderFactory](#4-dataloaderfactory)
5. [정규화 전략](#5-정규화-전략)
6. [API 명세](#6-api-명세)
7. [테스트 요구사항](#7-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Data 시스템은 머신러닝 실험을 위한 데이터 로딩, 전처리, 증강을 담당합니다. CIFAR-10 데이터셋을 기본으로 지원하며, 전이학습 모델을 위한 적절한 전처리를 제공합니다.

### 1.2 설계 원칙

- **모듈화**: 데이터 로딩과 전처리 분리
- **재현성**: 시드 제어를 통한 재현 가능한 분할
- **성능 최적화**: 효율적인 데이터 로딩
- **하드코딩 지양**: 모든 설정은 파라미터화

### 1.3 파일 구조

```
research/data/
├── __init__.py          # Data exports
├── cifar10.py           # CIFAR10DataModule (구현 완료)
├── loaders.py           # DataLoaderFactory (구현 완료)
└── transforms.py        # Transform utilities (선택)
```

---

## 2. CIFAR10DataModule

### 2.1 개요

CIFAR-10 데이터셋을 위한 완전한 데이터 모듈입니다. PyTorch Lightning 스타일의 인터페이스를 제공합니다.

### 2.2 클래스 구조

```python
class CIFAR10DataModule:
    """CIFAR-10 데이터 모듈"""

    # 클래스 상수
    DEFAULT_DATA_DIR = "./data"
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_NUM_WORKERS = 2
    DEFAULT_IMAGE_SIZE = 224  # ResNet/VGG 입력 크기

    # CIFAR-10 메타데이터
    NUM_CLASSES = 10
    TRAIN_SIZE = 50000
    TEST_SIZE = 10000
    ORIGINAL_IMAGE_SIZE = 32
    NUM_CHANNELS = 3

    # 클래스 이름
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self,
                 data_dir: str = DEFAULT_DATA_DIR,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 num_workers: int = DEFAULT_NUM_WORKERS,
                 image_size: int = DEFAULT_IMAGE_SIZE):
        """초기화

        Args:
            data_dir: 데이터 저장 디렉토리
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
            image_size: 리사이즈 타겟 크기 (전이학습용)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
```

### 2.3 주요 메서드

#### prepare_data()

```python
def prepare_data(self):
    """데이터 다운로드 (한 번만 실행)

    프로세스:
    1. CIFAR-10 다운로드 확인
    2. 필요시 자동 다운로드
    3. 파일 무결성 검증

    Note: 멀티프로세싱 환경에서 한 번만 실행
    """
    CIFAR10(self.data_dir, train=True, download=True)
    CIFAR10(self.data_dir, train=False, download=True)
```

#### setup()

```python
def setup(self, stage: str = None, val_split: float = 0.2):
    """데이터셋 준비 및 분할

    프로세스:
    1. Transform 정의 (train/test 별도)
    2. CIFAR-10 로드
    3. Train을 train/val로 분할 (80/20 기본)
    4. 분할 정보 출력

    Args:
        stage: 'fit', 'test', 또는 None
        val_split: 검증 데이터 비율

    결과:
        self.train_dataset: 40,000 샘플
        self.val_dataset: 10,000 샘플
        self.test_dataset: 10,000 샘플
    """
```

**현재 문제점**:
- `random_split` 사용으로 클래스 불균형 가능
- Stratified splitting 미지원

#### DataLoader 메서드들

```python
def train_dataloader(self) -> DataLoader:
    """학습 데이터로더 반환

    설정:
    - shuffle=True
    - pin_memory=True (GPU 전송 최적화)
    - drop_last=False
    """
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        pin_memory=True
    )

def val_dataloader(self) -> DataLoader:
    """검증 데이터로더 반환

    설정:
    - shuffle=False (재현성)
    - pin_memory=True
    """

def test_dataloader(self) -> DataLoader:
    """테스트 데이터로더 반환

    설정:
    - shuffle=False
    - pin_memory=True
    """
```

---

## 3. Transform 파이프라인

### 3.1 Training Transform

```python
self.train_transform = transforms.Compose([
    # 1. 크기 조정 (32x32 → 224x224)
    transforms.Resize((self.image_size, self.image_size)),

    # 2. 데이터 증강
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),

    # 3. 텐서 변환
    transforms.ToTensor(),

    # 4. 정규화 (ImageNet 통계)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3.2 Test Transform

```python
self.test_transform = transforms.Compose([
    # 1. 크기 조정
    transforms.Resize((self.image_size, self.image_size)),

    # 2. 텐서 변환
    transforms.ToTensor(),

    # 3. 정규화 (ImageNet 통계)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3.3 데이터 증강 전략

| 증강 기법 | 파라미터 | 목적 |
|----------|----------|------|
| RandomHorizontalFlip | p=0.5 | 좌우 대칭 불변성 |
| RandomRotation | degrees=10 | 회전 불변성 |

**누락된 증강 기법**:
- ColorJitter (색상 변화)
- RandomCrop (크롭)
- RandomErasing (일부 지우기)
- MixUp/CutMix (고급 증강)

---

## 4. DataLoaderFactory

### 4.1 개요

일반적인 데이터셋에서 DataLoader를 생성하는 정적 팩토리 클래스입니다.

### 4.2 클래스 구조

```python
class DataLoaderFactory:
    """DataLoader 생성 팩토리"""

    # 클래스 상수
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_TRAIN_RATIO = 0.8
    DEFAULT_SEED = 42

    @staticmethod
    def create_loaders(train_dataset: Dataset,
                      test_dataset: Optional[Dataset] = None,
                      train_ratio: float = DEFAULT_TRAIN_RATIO,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      num_workers: int = DEFAULT_NUM_WORKERS,
                      seed: int = DEFAULT_SEED) -> tuple:
        """DataLoader 트리플렛 생성"""
```

### 4.3 create_loaders() 메서드

```python
def create_loaders(...) -> tuple:
    """Train/Val/Test 데이터로더 생성

    프로세스:
    1. Train 데이터셋 크기 계산
    2. random_split으로 train/val 분할
    3. Generator로 시드 제어
    4. 3개 DataLoader 생성

    Args:
        train_dataset: 학습용 데이터셋
        test_dataset: 테스트 데이터셋 (선택)
        train_ratio: 학습 데이터 비율
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        seed: 랜덤 시드

    Returns:
        (train_loader, val_loader, test_loader) 튜플
    """

    # 분할 크기 계산
    total_size = len(train_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    # 시드 제어 분할
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=generator
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    # ... val_loader, test_loader
```

### 4.4 설계 결정

1. **pin_memory=True**: GPU 전송 최적화
2. **shuffle 정책**: train만 True, val/test는 False
3. **시드 제어**: 재현 가능한 분할

---

## 5. 정규화 전략

### 5.1 ImageNet 정규화 사용 이유

```python
# ImageNet 통계
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 네이티브 통계 (참고용)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
```

**ImageNet 정규화를 사용하는 이유**:
1. **전이학습 모델 호환성**: ResNet, VGG 등은 ImageNet으로 학습
2. **사전학습 가중치 활용**: ImageNet 통계 기대
3. **성능 향상**: 전이학습시 더 나은 결과

**문제점**:
- CIFAR-10 전용 모델에는 부적절
- 통계 차이로 인한 성능 저하 가능

### 5.2 개선 제안

```python
class AdaptiveNormalization:
    """데이터셋별 적응형 정규화"""

    @staticmethod
    def get_normalization(dataset_name: str, use_pretrained: bool):
        if use_pretrained:
            # 전이학습: ImageNet 통계
            return IMAGENET_MEAN, IMAGENET_STD
        else:
            # 처음부터 학습: 네이티브 통계
            if dataset_name == 'cifar10':
                return CIFAR10_MEAN, CIFAR10_STD
            # ... 다른 데이터셋
```

---

## 6. API 명세

### 6.1 CIFAR10DataModule 사용

```python
from research.data import CIFAR10DataModule

# 데이터 모듈 생성
data_module = CIFAR10DataModule(
    data_dir='./data',
    batch_size=32,
    num_workers=4,
    image_size=224  # 전이학습용
)

# 데이터 준비
data_module.prepare_data()  # 다운로드
data_module.setup()          # 분할

# DataLoader 가져오기
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# 메타데이터 접근
class_names = data_module.get_class_names()
print(f"Classes: {class_names}")
```

### 6.2 DataLoaderFactory 사용

```python
from research.data import DataLoaderFactory
from torchvision.datasets import CIFAR10

# 데이터셋 준비
transform = transforms.ToTensor()
train_dataset = CIFAR10('./data', train=True, transform=transform)
test_dataset = CIFAR10('./data', train=False, transform=transform)

# DataLoader 생성
train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_ratio=0.8,
    batch_size=64,
    num_workers=4,
    seed=42
)
```

### 6.3 커스텀 데이터셋 통합

```python
class CustomDataModule:
    """커스텀 데이터 모듈 예제"""

    def __init__(self, data_path: str, **kwargs):
        self.data_path = data_path
        # 커스텀 설정

    def setup(self):
        # 커스텀 데이터 로딩
        self.train_dataset = CustomDataset(
            self.data_path,
            split='train',
            transform=self.get_train_transform()
        )

    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 커스텀 정규화
        ])
```

---

## 7. 테스트 요구사항

### 7.1 단위 테스트

```python
def test_cifar10_datamodule():
    """CIFAR10DataModule 단위 테스트"""

    # 1. 초기화 테스트
    dm = CIFAR10DataModule(batch_size=16)
    assert dm.batch_size == 16

    # 2. 데이터 준비
    dm.prepare_data()
    dm.setup()

    # 3. 분할 크기 확인
    assert len(dm.train_dataset) == 40000
    assert len(dm.val_dataset) == 10000
    assert len(dm.test_dataset) == 10000

    # 4. DataLoader 생성
    train_loader = dm.train_dataloader()
    assert len(train_loader) == 40000 // 16
```

### 7.2 Transform 테스트

```python
def test_transforms():
    """Transform 파이프라인 테스트"""

    dm = CIFAR10DataModule(image_size=224)
    dm.setup()

    # 샘플 가져오기
    sample, label = dm.train_dataset[0]

    # 크기 확인
    assert sample.shape == (3, 224, 224)

    # 정규화 확인
    assert -3 < sample.min() < sample.max() < 3
```

### 7.3 재현성 테스트

```python
def test_reproducibility():
    """분할 재현성 테스트"""

    # 동일 시드로 두 번 생성
    loaders1 = DataLoaderFactory.create_loaders(
        train_dataset, seed=42
    )
    loaders2 = DataLoaderFactory.create_loaders(
        train_dataset, seed=42
    )

    # 첫 배치 비교
    batch1 = next(iter(loaders1[0]))
    batch2 = next(iter(loaders2[0]))

    assert torch.equal(batch1[0], batch2[0])
```

### 7.4 성능 테스트

```python
def test_loading_performance():
    """데이터 로딩 성능 테스트"""

    import time

    dm = CIFAR10DataModule(
        batch_size=128,
        num_workers=4
    )
    dm.setup()

    loader = dm.train_dataloader()

    # 10 배치 로딩 시간 측정
    start = time.time()
    for i, (data, labels) in enumerate(loader):
        if i >= 10:
            break
    elapsed = time.time() - start

    assert elapsed < 1.0  # 1초 이내
```

### 7.5 Edge Case 테스트

- 빈 데이터셋 처리
- num_workers=0 동작
- 매우 큰 batch_size
- 이미지 크기 극단값 (8x8, 1024x1024)

---

## 부록: 성능 최적화

### 1. DataLoader 최적화

| 설정 | 권장값 | 효과 |
|------|--------|------|
| num_workers | CPU 코어수 | 병렬 로딩 |
| pin_memory | True | GPU 전송 가속 |
| persistent_workers | True | 워커 재사용 |
| prefetch_factor | 2 | 미리 로드 |

### 2. 메모리 최적화

```python
# 대용량 데이터셋용
class EfficientDataLoader:
    def __init__(self, ...):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # 워커 재사용
            prefetch_factor=2,        # 미리 로드
            drop_last=True            # 불완전 배치 제거
        )
```

### 3. 분산 학습 지원

```python
from torch.utils.data.distributed import DistributedSampler

def create_distributed_loader(dataset, world_size, rank):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    return DataLoader(dataset, sampler=sampler, ...)
```

---

## 개선 로드맵

### 우선순위 P0 (긴급)

1. **Stratified Splitting**
   - 클래스 균형 유지 분할
   - sklearn.model_selection.train_test_split 활용

2. **정규화 옵션 제공**
   - ImageNet/CIFAR-10 선택 가능
   - 자동 감지 모드

### 우선순위 P1 (중요)

1. **고급 증강 기법**
   - AutoAugment
   - MixUp/CutMix
   - RandAugment

2. **동적 배치 크기**
   - GPU 메모리 기반 자동 조정

### 우선순위 P2 (개선)

1. **다른 데이터셋 지원**
   - ImageNet
   - COCO
   - Custom datasets

2. **온라인 증강**
   - GPU 기반 증강
   - NVIDIA DALI 통합

---

## SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ✅ | 데이터 로딩만 담당 |
| **O**pen/Closed | ⚠️ | 새 데이터셋 추가시 수정 필요 |
| **L**iskov Substitution | ✅ | DataModule 인터페이스 준수 |
| **I**nterface Segregation | ✅ | 최소 인터페이스 |
| **D**ependency Inversion | ✅ | Dataset 추상화 사용 |

OCP 개선을 위해 DataModule 추상 클래스 도입이 필요합니다.
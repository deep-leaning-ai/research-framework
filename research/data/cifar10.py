"""
CIFAR-10 데이터셋 모듈

개선사항:
- 하드코딩 제거: 정규화 값 클래스 상수화
- 성능 최적화: persistent_workers, prefetch_factor 추가
- num_workers 자동 설정
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple, Optional
import os


class CIFAR10DataModule:
    """
    CIFAR-10 데이터셋 관리 클래스 (PyTorch Lightning 호환)

    개선사항:
    - 성능 최적화: persistent_workers, prefetch_factor
    - 하드코딩 제거: 정규화 상수화
    - num_workers 자동 설정
    """

    # 클래스 상수 - ImageNet 정규화 (전이학습용)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # CIFAR-10 native 정규화 (참고용)
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2470, 0.2435, 0.2616]

    # 데이터 증강 파라미터
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_DEGREES = 10

    # 데이터셋 분할 비율
    TRAIN_VAL_SPLIT_RATIO = 0.8

    # 성능 최적화 파라미터
    DEFAULT_NUM_WORKERS = 4  # 기본값 증가
    DEFAULT_PREFETCH_FACTOR = 2
    MIN_BATCHES_FOR_PERSISTENT = 10  # persistent_workers 활성화 최소 배치 수

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        image_size: int = 224,
        use_imagenet_norm: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None
    ):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            batch_size: 배치 크기
            num_workers: DataLoader 워커 수 (None=자동 설정)
            image_size: 이미지 크기 (ResNet/VGG: 224)
            use_imagenet_norm: ImageNet 정규화 사용 여부 (전이학습 시 True)
            persistent_workers: 워커 재사용 (None=자동)
            prefetch_factor: 배치 prefetch 개수 (None=기본값)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_imagenet_norm = use_imagenet_norm

        # num_workers 자동 설정
        if num_workers is None:
            # CPU 코어 수에 따라 자동 설정 (최대 8)
            self.num_workers = min(os.cpu_count() or self.DEFAULT_NUM_WORKERS, 8)
        else:
            self.num_workers = num_workers

        # persistent_workers 자동 설정
        if persistent_workers is None:
            # 워커가 있고 충분한 배치가 있을 때만 활성화
            total_samples = 50000  # CIFAR-10 train size
            num_batches = total_samples // batch_size
            self.persistent_workers = (
                self.num_workers > 0 and
                num_batches >= self.MIN_BATCHES_FOR_PERSISTENT
            )
        else:
            self.persistent_workers = persistent_workers and self.num_workers > 0

        # prefetch_factor 설정
        if prefetch_factor is None:
            self.prefetch_factor = self.DEFAULT_PREFETCH_FACTOR if self.num_workers > 0 else None
        else:
            self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

        # 데이터셋 속성 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

        # 정규화 값 선택
        if use_imagenet_norm:
            norm_mean = self.IMAGENET_MEAN
            norm_std = self.IMAGENET_STD
        else:
            norm_mean = self.CIFAR10_MEAN
            norm_std = self.CIFAR10_STD

        # Transform 정의
        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=self.HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(self.ROTATION_DEGREES),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def prepare_data(self):
        """데이터셋 다운로드"""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)
        print("[OK] CIFAR-10 dataset downloaded")

    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정 (train/val/test 분할)"""
        # 훈련용 데이터셋
        full_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform_train
        )

        # 훈련/검증 분할 (8:2)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # 테스트 데이터셋
        self.test_dataset = datasets.CIFAR10(
            self.data_dir, train=False, transform=self.transform_test
        )

        print(f"[OK] CIFAR-10 splits: Train={train_size}, Val={val_size}, Test={len(self.test_dataset)}")

    def train_dataloader(self):
        """훈련 데이터로더"""
        if self.train_dataset is None:
            self.setup()

        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
        }

        # persistent_workers와 prefetch_factor는 num_workers > 0일 때만 적용
        if self.num_workers > 0:
            if self.persistent_workers:
                loader_kwargs['persistent_workers'] = True
            if self.prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = self.prefetch_factor

        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self):
        """검증 데이터로더"""
        if self.val_dataset is None:
            self.setup()

        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': True,
        }

        # persistent_workers와 prefetch_factor는 num_workers > 0일 때만 적용
        if self.num_workers > 0:
            if self.persistent_workers:
                loader_kwargs['persistent_workers'] = True
            if self.prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = self.prefetch_factor

        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self):
        """테스트 데이터로더"""
        if self.test_dataset is None:
            self.setup()

        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': True,
        }

        # persistent_workers와 prefetch_factor는 num_workers > 0일 때만 적용
        if self.num_workers > 0:
            if self.persistent_workers:
                loader_kwargs['persistent_workers'] = True
            if self.prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = self.prefetch_factor

        return DataLoader(self.test_dataset, **loader_kwargs)

    def get_class_names(self) -> list:
        """클래스 이름 반환"""
        return self.class_names

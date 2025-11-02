"""
CIFAR-10 데이터셋 모듈
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from typing import Tuple, Optional


class CIFAR10DataModule:
    """CIFAR-10 데이터셋 관리 클래스 (PyTorch Lightning 호환)"""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 2,
        image_size: int = 224
    ):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
            batch_size: 배치 크기
            num_workers: DataLoader 워커 수
            image_size: 이미지 크기 (ResNet/VGG: 224)
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # 데이터셋 속성 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

        # ImageNet 정규화 (전이학습용)
        self.transform_train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
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

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """검증 데이터로더"""
        if self.val_dataset is None:
            self.setup()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """테스트 데이터로더"""
        if self.test_dataset is None:
            self.setup()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_class_names(self) -> list:
        """클래스 이름 반환"""
        return self.class_names

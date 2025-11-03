"""
데이터 로더 팩토리 모듈

이 모듈은 다양한 데이터셋에 대한 DataLoader를 쉽게 생성할 수 있는
팩토리 클래스를 제공합니다. 자동 train/validation 분할과 최적화된
설정을 포함합니다.

사용 예시:
    >>> from torchvision import datasets, transforms
    >>> from research.data.loaders import DataLoaderFactory
    >>>
    >>> # CIFAR10 데이터셋 준비
    >>> transform = transforms.Compose([
    ...     transforms.ToTensor(),
    ...     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ... ])
    >>> train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    >>> test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    >>>
    >>> # DataLoader 생성
    >>> train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
    ...     train_dataset=train_dataset,
    ...     test_dataset=test_dataset,
    ...     train_ratio=0.8,
    ...     batch_size=32,
    ...     num_workers=4
    ... )
    >>>
    >>> print(f"Train batches: {len(train_loader)}")
    >>> print(f"Val batches: {len(val_loader)}")
    >>> print(f"Test batches: {len(test_loader)}")

Custom Dataset 예시:
    >>> class CustomDataset(Dataset):
    ...     def __init__(self, data, labels):
    ...         self.data = data
    ...         self.labels = labels
    ...
    ...     def __len__(self):
    ...         return len(self.data)
    ...
    ...     def __getitem__(self, idx):
    ...         return self.data[idx], self.labels[idx]
    >>>
    >>> # 커스텀 데이터셋으로 로더 생성
    >>> custom_dataset = CustomDataset(data, labels)
    >>> train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
    ...     train_dataset=custom_dataset,
    ...     train_ratio=0.9,  # 90% train, 10% validation
    ...     batch_size=64
    ... )
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DataLoaderFactory:
    """
    DataLoader 생성을 위한 팩토리 클래스

    이 클래스는 PyTorch Dataset을 입력받아 최적화된 설정으로
    train, validation, test DataLoader를 생성합니다.

    Features:
        - 자동 train/validation 분할
        - pin_memory 최적화 (GPU 학습 시 성능 향상)
        - 재현 가능한 랜덤 시드 지원
        - 유연한 배치 크기와 워커 수 설정

    Design Pattern:
        - Factory Pattern: DataLoader 생성 로직 캡슐화
        - Static Factory Method: 인스턴스 생성 없이 사용 가능
    """

    @staticmethod
    def create_loaders(
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        train_ratio: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        범용 DataLoader 생성 함수

        Args:
            train_dataset: 학습용 Dataset
            test_dataset: 테스트용 Dataset (선택)
            train_ratio: 학습/검증 분할 비율
            batch_size: 배치 크기
            num_workers: 워커 수
            seed: 랜덤 시드

        Returns:
            train_loader, val_loader, test_loader
        """
        # Train/Val 분할
        train_size = int(train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_split, val_split = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        # DataLoader 생성
        train_loader = DataLoader(
            train_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_split,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        return train_loader, val_loader, test_loader

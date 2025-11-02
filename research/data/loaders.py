"""
데이터 로더 팩토리 모듈
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DataLoaderFactory:
    """다양한 데이터셋에 대한 DataLoader 생성 팩토리"""

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

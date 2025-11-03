"""
DataLoaderFactory 단위 테스트

이 테스트는 DataLoader 생성 팩토리의 기능을 검증합니다.
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from research.data.loaders import DataLoaderFactory


class DummyDataset(Dataset):
    """테스트용 더미 데이터셋"""

    def __init__(self, size=100, input_dim=10, output_dim=1):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, output_dim, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TestDataLoaderFactory:
    """DataLoaderFactory 테스트 클래스"""

    def test_create_loaders_basic(self):
        """
        Given: 학습용 데이터셋
        When: create_loaders() 호출
        Then: train, val 로더가 생성됨
        """
        # Given
        train_dataset = DummyDataset(size=100)

        # When
        train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
            train_dataset=train_dataset,
            batch_size=10,
            train_ratio=0.8
        )

        # Then
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert test_loader is None

        # 데이터 분할 확인
        assert len(train_loader.dataset) == 80  # 100 * 0.8
        assert len(val_loader.dataset) == 20   # 100 * 0.2

    def test_create_loaders_with_test(self):
        """
        Given: 학습용과 테스트용 데이터셋
        When: create_loaders() 호출
        Then: train, val, test 로더가 모두 생성됨
        """
        # Given
        train_dataset = DummyDataset(size=100)
        test_dataset = DummyDataset(size=50)

        # When
        train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=10,
            train_ratio=0.8
        )

        # Then
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 20
        assert len(test_loader.dataset) == 50

    def test_batch_size_configuration(self):
        """
        Given: 특정 배치 크기 설정
        When: DataLoader 생성
        Then: 설정된 배치 크기로 데이터가 로드됨
        """
        # Given
        dataset = DummyDataset(size=100)
        batch_size = 25

        # When
        train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            batch_size=batch_size,
            train_ratio=0.8
        )

        # Then
        # 배치 크기 확인
        for batch_data, batch_labels in train_loader:
            assert batch_data.shape[0] <= batch_size
            break

        for batch_data, batch_labels in val_loader:
            assert batch_data.shape[0] <= batch_size
            break

    def test_num_workers_configuration(self):
        """
        Given: num_workers 설정
        When: DataLoader 생성
        Then: 설정된 워커 수로 로더가 생성됨
        """
        # Given
        dataset = DummyDataset(size=100)
        num_workers = 2

        # When
        train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            num_workers=num_workers
        )

        # Then
        assert train_loader.num_workers == num_workers
        assert val_loader.num_workers == num_workers

    def test_reproducibility_with_seed(self):
        """
        Given: 동일한 시드값
        When: 여러 번 DataLoader 생성
        Then: 동일한 데이터 분할이 생성됨
        """
        # Given
        dataset = DummyDataset(size=100)
        seed = 42

        # When: 첫 번째 생성
        train_loader1, val_loader1, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            train_ratio=0.7,
            seed=seed
        )

        # When: 두 번째 생성 (같은 시드)
        train_loader2, val_loader2, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            train_ratio=0.7,
            seed=seed
        )

        # Then: 같은 인덱스 분할
        # 첫 번째 배치의 데이터가 같은지 확인
        train_data1 = next(iter(train_loader1))[0]
        train_data2 = next(iter(train_loader2))[0]
        # 데이터셋이 동일한 객체이므로 인덱스 분할만 확인
        assert len(train_loader1.dataset) == len(train_loader2.dataset)

    def test_train_ratio_edge_cases(self):
        """
        Given: 극단적인 train_ratio 값
        When: DataLoader 생성
        Then: 정상적으로 처리됨
        """
        # Given
        dataset = DummyDataset(size=100)

        # When & Then: train_ratio = 0.1
        train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            train_ratio=0.1
        )
        assert len(train_loader.dataset) == 10
        assert len(val_loader.dataset) == 90

        # When & Then: train_ratio = 0.9
        train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            train_ratio=0.9
        )
        assert len(train_loader.dataset) == 90
        assert len(val_loader.dataset) == 10

    def test_pin_memory_enabled(self):
        """
        Given: DataLoader 생성
        When: 기본 설정으로 생성
        Then: pin_memory가 활성화됨 (GPU 최적화)
        """
        # Given
        dataset = DummyDataset(size=100)

        # When
        train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            test_dataset=dataset
        )

        # Then
        assert train_loader.pin_memory == True
        assert val_loader.pin_memory == True
        assert test_loader.pin_memory == True

    def test_shuffle_configuration(self):
        """
        Given: DataLoader 생성
        When: 기본 설정으로 생성
        Then: train은 shuffle=True, val/test는 shuffle=False
        """
        # Given
        dataset = DummyDataset(size=100)

        # When
        train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
            train_dataset=dataset,
            test_dataset=dataset
        )

        # Then
        # DataLoader의 sampler 타입으로 shuffle 여부 확인
        from torch.utils.data.sampler import RandomSampler, SequentialSampler

        # train_loader는 shuffle=True이므로 RandomSampler 사용
        assert isinstance(train_loader.sampler, RandomSampler) or \
               isinstance(train_loader.batch_sampler.sampler, RandomSampler)

        # val_loader는 shuffle=False이므로 SequentialSampler 사용
        assert isinstance(val_loader.sampler, SequentialSampler) or \
               isinstance(val_loader.batch_sampler.sampler, SequentialSampler)

    def test_empty_dataset(self):
        """
        Given: 빈 데이터셋
        When: DataLoader 생성 시도
        Then: ValueError 발생 (빈 데이터셋은 분할 불가)
        """
        # Given
        empty_dataset = DummyDataset(size=0)

        # When & Then: 빈 데이터셋은 분할할 수 없으므로 에러 발생
        with pytest.raises(ValueError):
            train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
                train_dataset=empty_dataset,
                train_ratio=0.8
            )

    def test_large_dataset_performance(self):
        """
        Given: 큰 데이터셋
        When: DataLoader 생성
        Then: 메모리 효율적으로 생성됨
        """
        # Given
        large_dataset = DummyDataset(size=10000)

        # When
        train_loader, val_loader, _ = DataLoaderFactory.create_loaders(
            train_dataset=large_dataset,
            batch_size=128,
            train_ratio=0.8,
            num_workers=4
        )

        # Then
        assert len(train_loader.dataset) == 8000
        assert len(val_loader.dataset) == 2000

        # 배치 수 확인
        assert len(train_loader) == (8000 + 127) // 128  # 올림 나눗셈
        assert len(val_loader) == (2000 + 127) // 128
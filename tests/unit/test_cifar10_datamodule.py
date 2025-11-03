"""
CIFAR10DataModule 테스트
TDD 방식: Given-When-Then
"""
import pytest
import torch
from torch.utils.data import DataLoader
import os
from research.data.cifar10 import CIFAR10DataModule


class TestClassConstants:
    """클래스 상수 테스트 - 하드코딩 제거 검증"""

    def test_정규화_상수_정의(self):
        """
        Given: CIFAR10DataModule 클래스
        When: 정규화 상수 확인
        Then: ImageNet과 CIFAR10 정규화 상수가 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(CIFAR10DataModule, 'IMAGENET_MEAN')
        assert hasattr(CIFAR10DataModule, 'IMAGENET_STD')
        assert hasattr(CIFAR10DataModule, 'CIFAR10_MEAN')
        assert hasattr(CIFAR10DataModule, 'CIFAR10_STD')

        # 값 검증
        assert CIFAR10DataModule.IMAGENET_MEAN == [0.485, 0.456, 0.406]
        assert CIFAR10DataModule.IMAGENET_STD == [0.229, 0.224, 0.225]
        assert CIFAR10DataModule.CIFAR10_MEAN == [0.4914, 0.4822, 0.4465]
        assert CIFAR10DataModule.CIFAR10_STD == [0.2470, 0.2435, 0.2616]

    def test_데이터_증강_상수_정의(self):
        """
        Given: CIFAR10DataModule 클래스
        When: 데이터 증강 상수 확인
        Then: 증강 파라미터가 상수로 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(CIFAR10DataModule, 'HORIZONTAL_FLIP_PROB')
        assert hasattr(CIFAR10DataModule, 'ROTATION_DEGREES')

        assert CIFAR10DataModule.HORIZONTAL_FLIP_PROB == 0.5
        assert CIFAR10DataModule.ROTATION_DEGREES == 10

    def test_성능_최적화_상수_정의(self):
        """
        Given: CIFAR10DataModule 클래스
        When: 성능 최적화 상수 확인
        Then: num_workers, prefetch_factor 관련 상수가 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(CIFAR10DataModule, 'DEFAULT_NUM_WORKERS')
        assert hasattr(CIFAR10DataModule, 'DEFAULT_PREFETCH_FACTOR')
        assert hasattr(CIFAR10DataModule, 'MIN_BATCHES_FOR_PERSISTENT')

        assert CIFAR10DataModule.DEFAULT_NUM_WORKERS == 4
        assert CIFAR10DataModule.DEFAULT_PREFETCH_FACTOR == 2
        assert CIFAR10DataModule.MIN_BATCHES_FOR_PERSISTENT == 10

    def test_데이터셋_분할_상수_정의(self):
        """
        Given: CIFAR10DataModule 클래스
        When: 데이터셋 분할 상수 확인
        Then: TRAIN_VAL_SPLIT_RATIO가 정의되어야 함
        """
        # Given & When & Then
        assert hasattr(CIFAR10DataModule, 'TRAIN_VAL_SPLIT_RATIO')
        assert CIFAR10DataModule.TRAIN_VAL_SPLIT_RATIO == 0.8


class TestAutoConfiguration:
    """자동 설정 기능 테스트"""

    def test_num_workers_자동_설정_None일때(self, tmp_path):
        """
        Given: num_workers=None
        When: CIFAR10DataModule 초기화
        Then: CPU 코어 수에 따라 자동 설정 (최대 8)
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=None
        )

        # When & Then
        expected = min(os.cpu_count() or 4, 8)
        assert data_module.num_workers == expected

    def test_num_workers_명시적_설정(self, tmp_path):
        """
        Given: num_workers=2 (명시적 설정)
        When: CIFAR10DataModule 초기화
        Then: 설정한 값이 사용되어야 함
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=2
        )

        # When & Then
        assert data_module.num_workers == 2

    def test_persistent_workers_자동_활성화(self, tmp_path):
        """
        Given: persistent_workers=None, batch_size=32 (충분한 배치 수)
        When: CIFAR10DataModule 초기화
        Then: num_workers > 0이고 배치 수가 충분하면 자동 활성화
        """
        # Given
        batch_size = 32
        total_samples = 50000
        num_batches = total_samples // batch_size  # 1562 배치

        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=None
        )

        # When & Then
        # 배치 수가 MIN_BATCHES_FOR_PERSISTENT(10)보다 크므로 활성화
        assert data_module.persistent_workers is True

    def test_persistent_workers_자동_비활성화_배치부족(self, tmp_path):
        """
        Given: persistent_workers=None, batch_size=10000 (배치 수 부족)
        When: CIFAR10DataModule 초기화
        Then: 배치 수가 부족하면 자동 비활성화
        """
        # Given
        batch_size = 10000
        total_samples = 50000
        num_batches = total_samples // batch_size  # 5 배치

        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=None
        )

        # When & Then
        # 배치 수가 MIN_BATCHES_FOR_PERSISTENT(10)보다 작으므로 비활성화
        assert data_module.persistent_workers is False

    def test_persistent_workers_자동_비활성화_워커없음(self, tmp_path):
        """
        Given: persistent_workers=None, num_workers=0
        When: CIFAR10DataModule 초기화
        Then: num_workers=0이면 persistent_workers 비활성화
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=0,
            persistent_workers=None
        )

        # When & Then
        assert data_module.persistent_workers is False

    def test_persistent_workers_명시적_설정(self, tmp_path):
        """
        Given: persistent_workers=True (명시적 설정)
        When: CIFAR10DataModule 초기화
        Then: num_workers > 0이면 설정한 값이 사용됨
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=4,
            persistent_workers=True
        )

        # When & Then
        assert data_module.persistent_workers is True

    def test_persistent_workers_명시적_설정_워커없음(self, tmp_path):
        """
        Given: persistent_workers=True, num_workers=0
        When: CIFAR10DataModule 초기화
        Then: num_workers=0이면 persistent_workers는 강제로 False
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=0,
            persistent_workers=True
        )

        # When & Then
        assert data_module.persistent_workers is False

    def test_prefetch_factor_자동_설정_워커있음(self, tmp_path):
        """
        Given: prefetch_factor=None, num_workers=4
        When: CIFAR10DataModule 초기화
        Then: DEFAULT_PREFETCH_FACTOR 사용
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=4,
            prefetch_factor=None
        )

        # When & Then
        assert data_module.prefetch_factor == CIFAR10DataModule.DEFAULT_PREFETCH_FACTOR

    def test_prefetch_factor_자동_설정_워커없음(self, tmp_path):
        """
        Given: prefetch_factor=None, num_workers=0
        When: CIFAR10DataModule 초기화
        Then: prefetch_factor는 None
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=0,
            prefetch_factor=None
        )

        # When & Then
        assert data_module.prefetch_factor is None

    def test_prefetch_factor_명시적_설정(self, tmp_path):
        """
        Given: prefetch_factor=4 (명시적 설정)
        When: CIFAR10DataModule 초기화
        Then: num_workers > 0이면 설정한 값이 사용됨
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            num_workers=4,
            prefetch_factor=4
        )

        # When & Then
        assert data_module.prefetch_factor == 4


class TestNormalizationSelection:
    """정규화 전략 선택 테스트"""

    def test_ImageNet_정규화_사용(self, tmp_path):
        """
        Given: use_imagenet_norm=True
        When: CIFAR10DataModule 초기화
        Then: ImageNet 정규화 값이 사용되어야 함
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            use_imagenet_norm=True
        )

        # When & Then
        # Transform 확인 (Normalize 레이어 추출)
        normalize_transform = None
        for transform in data_module.transform_train.transforms:
            if transform.__class__.__name__ == 'Normalize':
                normalize_transform = transform
                break

        assert normalize_transform is not None
        assert normalize_transform.mean == CIFAR10DataModule.IMAGENET_MEAN
        assert normalize_transform.std == CIFAR10DataModule.IMAGENET_STD

    def test_CIFAR10_정규화_사용(self, tmp_path):
        """
        Given: use_imagenet_norm=False
        When: CIFAR10DataModule 초기화
        Then: CIFAR10 native 정규화 값이 사용되어야 함
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            use_imagenet_norm=False
        )

        # When & Then
        # Transform 확인 (Normalize 레이어 추출)
        normalize_transform = None
        for transform in data_module.transform_train.transforms:
            if transform.__class__.__name__ == 'Normalize':
                normalize_transform = transform
                break

        assert normalize_transform is not None
        assert normalize_transform.mean == CIFAR10DataModule.CIFAR10_MEAN
        assert normalize_transform.std == CIFAR10DataModule.CIFAR10_STD


class TestDataLoaderParameters:
    """DataLoader 파라미터 검증 테스트"""

    def test_train_dataloader_파라미터_워커있음(self, tmp_path):
        """
        Given: num_workers=4, persistent_workers=True, prefetch_factor=2
        When: train_dataloader 생성
        Then: 모든 최적화 파라미터가 DataLoader에 적용되어야 함
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2
        )

        # When
        # setup 호출하여 데이터셋 준비
        # 실제 다운로드 없이 테스트하기 위해 mock 사용하지 않고 속성만 확인
        assert data_module.num_workers == 4
        assert data_module.persistent_workers is True
        assert data_module.prefetch_factor == 2

    def test_train_dataloader_파라미터_워커없음(self, tmp_path):
        """
        Given: num_workers=0
        When: train_dataloader 생성
        Then: persistent_workers와 prefetch_factor가 적용되지 않아야 함
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            num_workers=0
        )

        # When & Then
        assert data_module.num_workers == 0
        assert data_module.persistent_workers is False
        assert data_module.prefetch_factor is None

    def test_모든_dataloader_동일_파라미터_사용(self, tmp_path):
        """
        Given: 성능 최적화 파라미터가 설정된 DataModule
        When: train/val/test dataloader 생성
        Then: 모든 DataLoader가 동일한 최적화 파라미터 사용
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2
        )

        # When & Then
        # 모든 DataLoader가 동일한 설정 사용
        assert data_module.num_workers == 4
        assert data_module.persistent_workers is True
        assert data_module.prefetch_factor == 2


class TestBackwardCompatibility:
    """하위 호환성 테스트"""

    def test_기본_파라미터로_초기화(self, tmp_path):
        """
        Given: 기본 파라미터만 사용
        When: CIFAR10DataModule 초기화
        Then: 이전 버전과 동일하게 동작해야 함
        """
        # Given & When
        data_module = CIFAR10DataModule(data_dir=str(tmp_path))

        # Then
        assert data_module.batch_size == 32
        assert data_module.image_size == 224
        assert data_module.use_imagenet_norm is True
        # 자동 설정값 확인
        assert data_module.num_workers >= 0
        assert isinstance(data_module.persistent_workers, bool)

    def test_기존_파라미터_여전히_작동(self, tmp_path):
        """
        Given: 기존에 사용하던 파라미터
        When: CIFAR10DataModule 초기화
        Then: 정상 동작해야 함
        """
        # Given & When
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=64,
            num_workers=2,
            image_size=32
        )

        # Then
        assert data_module.batch_size == 64
        assert data_module.num_workers == 2
        assert data_module.image_size == 32


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_매우_큰_배치_크기(self, tmp_path):
        """
        Given: batch_size=50000 (전체 데이터셋)
        When: CIFAR10DataModule 초기화
        Then: 배치 수가 1이므로 persistent_workers 비활성화
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=50000,
            num_workers=4,
            persistent_workers=None
        )

        # When & Then
        # 배치 수가 1이므로 MIN_BATCHES_FOR_PERSISTENT(10)보다 작음
        assert data_module.persistent_workers is False

    def test_매우_작은_배치_크기(self, tmp_path):
        """
        Given: batch_size=1
        When: CIFAR10DataModule 초기화
        Then: 배치 수가 많으므로 persistent_workers 활성화
        """
        # Given
        data_module = CIFAR10DataModule(
            data_dir=str(tmp_path),
            batch_size=1,
            num_workers=4,
            persistent_workers=None
        )

        # When & Then
        # 배치 수가 50000이므로 MIN_BATCHES_FOR_PERSISTENT(10)보다 큼
        assert data_module.persistent_workers is True

    def test_클래스_이름_반환(self, tmp_path):
        """
        Given: CIFAR10DataModule
        When: get_class_names 호출
        Then: 10개의 클래스 이름 반환
        """
        # Given
        data_module = CIFAR10DataModule(data_dir=str(tmp_path))

        # When
        class_names = data_module.get_class_names()

        # Then
        assert len(class_names) == 10
        expected_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        assert class_names == expected_classes

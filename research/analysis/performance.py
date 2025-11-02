"""
성능 측정 클래스
SRP: 성능 측정만 담당
"""

from typing import Tuple
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class PerformanceMetrics:
    """
    모델 성능 지표 계산 클래스

    추론 시간, 학습 시간, 과적합 정도 등을 측정
    """

    @staticmethod
    def measure_inference_time(
        model: nn.Module, data_loader: DataLoader, device: str, num_batches: int = 100
    ) -> Tuple[float, float]:
        """
        추론 시간 측정

        Args:
            model: PyTorch 모델
            data_loader: 데이터 로더
            device: 디바이스 ('cuda' 또는 'cpu')
            num_batches: 측정할 배치 수

        Returns:
            (평균 추론 시간, 표준편차)
        """
        model.eval()
        times = []

        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                inputs = inputs.to(device)

                start_time = time.time()
                _ = model(inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                times.append(end_time - start_time)

        return np.mean(times), np.std(times)

    @staticmethod
    def calculate_overfitting_gap(train_metric: float, test_metric: float) -> float:
        """
        과적합 정도 계산

        Args:
            train_metric: 학습 메트릭 값
            test_metric: 테스트 메트릭 값

        Returns:
            과적합 갭 (train - test)
        """
        return train_metric - test_metric

    @staticmethod
    def measure_training_speed(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        num_batches: int = 100,
    ) -> float:
        """
        학습 속도 측정

        Args:
            model: PyTorch 모델
            data_loader: 데이터 로더
            criterion: 손실 함수
            optimizer: 옵티마이저
            device: 디바이스
            num_batches: 측정할 배치 수

        Returns:
            배치당 평균 학습 시간 (초)
        """
        model.train()
        times = []

        for i, (inputs, labels) in enumerate(data_loader):
            if i >= num_batches:
                break

            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

        return np.mean(times)

    @staticmethod
    def calculate_model_efficiency(
        params: int, metric_value: float, is_higher_better: bool = True
    ) -> float:
        """
        모델 효율성 계산 (파라미터 대비 성능)

        Args:
            params: 파라미터 수
            metric_value: 메트릭 값
            is_higher_better: 높을수록 좋은 메트릭인지

        Returns:
            효율성 점수
        """
        if params == 0:
            return 0.0

        # 파라미터 수의 로그를 사용하여 정규화
        log_params = np.log10(params + 1)

        if is_higher_better:
            efficiency = metric_value / log_params
        else:
            efficiency = log_params / (metric_value + 1e-10)

        return efficiency

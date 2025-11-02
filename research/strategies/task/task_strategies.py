"""
구체적인 Task 전략 구현
"""

from typing import Optional
import torch
import torch.nn as nn
from research.strategies.task.base import TaskStrategy


class MultiClassStrategy(TaskStrategy):
    """
    다중분류 전략

    10개 이상의 클래스를 분류하는 태스크에 사용
    (예: MNIST, CIFAR-10, ImageNet)
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: 클래스 개수
        """
        self.num_classes = num_classes

    def get_criterion(self) -> nn.Module:
        """CrossEntropyLoss 반환 (내부적으로 Softmax 포함)"""
        return nn.CrossEntropyLoss()

    def get_output_activation(self) -> Optional[nn.Module]:
        """CrossEntropyLoss가 Softmax를 포함하므로 None"""
        return None

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """정확도 계산"""
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return 100.0 * correct / total

    def get_metric_name(self) -> str:
        """메트릭 이름"""
        return "Accuracy"

    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """레이블 전처리 (다중분류는 변경 불필요)"""
        return labels


class BinaryClassificationStrategy(TaskStrategy):
    """
    이진분류 전략

    2개의 클래스를 분류하는 태스크에 사용
    (예: 스팸 필터링, 감정 분석)
    """

    def get_criterion(self) -> nn.Module:
        """BCEWithLogitsLoss 반환 (Sigmoid + BCE 결합)"""
        return nn.BCEWithLogitsLoss()

    def get_output_activation(self) -> Optional[nn.Module]:
        """추론 시 사용할 Sigmoid"""
        return nn.Sigmoid()

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """정확도 계산"""
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return 100.0 * correct / total

    def get_metric_name(self) -> str:
        """메트릭 이름"""
        return "Accuracy"

    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """레이블을 float으로 변환하고 shape 조정"""
        return labels.float().view(-1, 1)


class RegressionStrategy(TaskStrategy):
    """
    회귀 전략

    연속적인 값을 예측하는 태스크에 사용
    (예: 주택 가격 예측, 주식 가격 예측)
    """

    def get_criterion(self) -> nn.Module:
        """MSE Loss 반환"""
        return nn.MSELoss()

    def get_output_activation(self) -> Optional[nn.Module]:
        """회귀는 활성화 함수 불필요"""
        return None

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """MSE 계산"""
        mse = nn.MSELoss()(outputs, labels)
        return mse.item()

    def get_metric_name(self) -> str:
        """메트릭 이름"""
        return "MSE"

    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """레이블을 float으로 변환하고 shape 조정"""
        return labels.float().view(-1, 1)

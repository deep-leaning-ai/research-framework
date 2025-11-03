"""
Convolutional Neural Network 모델
"""

from typing import Dict, Any
import torch.nn as nn
from research.models.simple.base import BaseModel


class CNN(BaseModel):
    """
    Convolutional Neural Network

    MNIST와 같은 이미지 분류 태스크에 적합한 CNN 아키텍처
    """

    # Architecture constants - avoid hardcoding
    INPUT_CHANNELS = 1
    CONV1_OUT_CHANNELS = 16
    CONV2_OUT_CHANNELS = 32

    KERNEL_SIZE = 5
    STRIDE = 1
    PADDING = 2
    POOL_SIZE = 2

    # MNIST input: 28x28
    INPUT_SIZE = 28
    AFTER_POOL1_SIZE = 14  # 28 / 2
    AFTER_POOL2_SIZE = 7   # 14 / 2
    FC_INPUT_SIZE = 32 * 7 * 7  # 1568

    DEFAULT_OUTPUT_DIM = 10

    def __init__(self, output_dim: int = None, task_strategy: "TaskStrategy" = None):
        """
        Args:
            output_dim: 출력 차원 (기본값 DEFAULT_OUTPUT_DIM - MNIST)
            task_strategy: Task 전략 (None이면 기본 MultiClass 사용)
        """
        # Use default if not provided
        output_dim = output_dim or self.DEFAULT_OUTPUT_DIM

        # task_strategy가 None이면 나중에 설정됨 (순환 import 방지)
        if task_strategy is None:
            from research.strategies import MultiClassStrategy

            task_strategy = MultiClassStrategy(num_classes=output_dim)

        super().__init__(output_dim, task_strategy)

        # Convolutional Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                self.INPUT_CHANNELS,
                self.CONV1_OUT_CHANNELS,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=self.POOL_SIZE, stride=self.POOL_SIZE),
        )

        # Convolutional Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                self.CONV1_OUT_CHANNELS,
                self.CONV2_OUT_CHANNELS,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE,
                padding=self.PADDING
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=self.POOL_SIZE, stride=self.POOL_SIZE),
        )

        # Fully Connected Layer
        self.fc = nn.Linear(self.FC_INPUT_SIZE, self.output_dim, bias=True)

        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, 1, 28, 28)

        Returns:
            출력 텐서 (batch_size, output_dim)
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out

    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return "CNN"

    def get_model_info(self) -> Dict[str, Any]:
        """모델 상세 정보 반환"""
        return {
            "name": self.get_model_name(),
            "type": "Convolutional Neural Network",
            "output_dim": self.output_dim,
            "task": self.task_strategy.__class__.__name__,
            "features": [
                "Local Connectivity",
                "Parameter Sharing",
                "Translation Invariance",
            ],
            "architecture": {
                "layer1": "Conv2d(1->16) + ReLU + AvgPool2d",
                "layer2": "Conv2d(16->32) + ReLU + AvgPool2d",
                "fc": f"Linear(1568->{self.output_dim})",
            },
        }

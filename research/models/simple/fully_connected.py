"""
Fully Connected Neural Network 모델
"""

from typing import Dict, Any
import torch.nn as nn
from research.models.simple.base import BaseModel


class FullyConnectedNN(BaseModel):
    """
    Fully Connected Neural Network

    3-layer Fully Connected 아키텍처
    """

    # Architecture constants - avoid hardcoding
    INPUT_SIZE = 784  # 28x28
    DEFAULT_HIDDEN_SIZE = 128
    DEFAULT_OUTPUT_DIM = 10

    # Layer configuration
    NUM_HIDDEN_LAYERS = 2

    def __init__(
        self,
        hidden_size: int = None,
        output_dim: int = None,
        task_strategy: "TaskStrategy" = None,
    ):
        """
        Args:
            hidden_size: 은닉층 크기
            output_dim: 출력 차원
            task_strategy: Task 전략
        """
        # Use defaults if not provided
        hidden_size = hidden_size or self.DEFAULT_HIDDEN_SIZE
        output_dim = output_dim or self.DEFAULT_OUTPUT_DIM

        if task_strategy is None:
            from research.strategies import MultiClassStrategy

            task_strategy = MultiClassStrategy(num_classes=output_dim)

        super().__init__(output_dim, task_strategy)

        self.hidden_size = hidden_size

        # INPUT_SIZE input for MNIST (28x28 = 784)
        self.fc1 = nn.Linear(self.INPUT_SIZE, hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, self.output_dim)

        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 텐서 (batch_size, 1, 28, 28) 또는 (batch_size, 784)

        Returns:
            출력 텐서 (batch_size, output_dim)
        """
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_model_name(self) -> str:
        """모델 이름 반환"""
        return f"FullyConnected_h{self.hidden_size}"

    def get_model_info(self) -> Dict[str, Any]:
        """모델 상세 정보 반환"""
        return {
            "name": self.get_model_name(),
            "type": "Fully Connected Neural Network",
            "hidden_size": self.hidden_size,
            "output_dim": self.output_dim,
            "task": self.task_strategy.__class__.__name__,
            "features": [
                "Global Connectivity",
                "No Parameter Sharing",
                "Simple Architecture",
            ],
            "architecture": {
                "fc1": f"Linear(784->{self.hidden_size}) + ReLU",
                "fc2": f"Linear({self.hidden_size}->{self.hidden_size}) + ReLU",
                "fc3": f"Linear({self.hidden_size}->{self.output_dim})",
            },
        }

"""
모델 베이스 클래스
SRP: 모델의 기본 인터페이스만 정의
OCP: 새로운 모델 추가 시 기존 코드 수정 불필요
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    모델 추상 베이스 클래스

    모든 모델은 이 클래스를 상속받아야 하며,
    get_model_name과 get_model_info 메서드를 구현해야 함
    """

    def __init__(self, output_dim: int, task_strategy: "TaskStrategy"):
        """
        Args:
            output_dim: 출력 차원 (클래스 수 또는 회귀 출력 수)
            task_strategy: Task 전략 객체
        """
        super().__init__()
        self.output_dim = output_dim
        self.task_strategy = task_strategy

    @abstractmethod
    def get_model_name(self) -> str:
        """모델 이름 반환"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        pass

    @abstractmethod
    def forward(self, x):
        """순전파"""
        pass

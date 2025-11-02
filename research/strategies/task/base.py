"""
Task 전략 베이스 클래스
Strategy Pattern: 다양한 Task 타입을 지원하기 위한 전략 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class TaskStrategy(ABC):
    """
    Task별 전략을 정의하는 추상 클래스

    OCP: 새로운 Task 타입 추가 시 기존 코드 수정 불필요
    SRP: Task에 필요한 로직만 담당
    """

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        """
        Task에 맞는 손실 함수 반환

        Returns:
            손실 함수 (nn.Module)
        """
        pass

    @abstractmethod
    def get_output_activation(self) -> Optional[nn.Module]:
        """
        출력층 활성화 함수 반환

        Returns:
            활성화 함수 또는 None
        """
        pass

    @abstractmethod
    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        기본 평가 지표 계산

        Args:
            outputs: 모델 출력
            labels: 정답 레이블

        Returns:
            메트릭 값
        """
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        기본 메트릭 이름 반환

        Returns:
            메트릭 이름
        """
        pass

    @abstractmethod
    def prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Task에 맞게 레이블 전처리

        Args:
            labels: 원본 레이블

        Returns:
            전처리된 레이블
        """
        pass

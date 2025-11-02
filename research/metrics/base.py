"""
메트릭 계산 베이스 클래스
Strategy Pattern: 다양한 메트릭을 지원하기 위한 전략 인터페이스
"""

from abc import ABC, abstractmethod
import torch


class MetricCalculator(ABC):
    """
    메트릭 계산 추상 클래스

    OCP: 새로운 메트릭 추가 시 기존 코드 수정 불필요
    SRP: 메트릭 계산만 담당
    """

    @abstractmethod
    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        메트릭 계산

        Args:
            outputs: 모델 출력
            labels: 정답 레이블

        Returns:
            계산된 메트릭 값
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        메트릭 이름 반환

        Returns:
            메트릭 이름
        """
        pass

    @property
    def name(self) -> str:
        """
        메트릭 이름 프로퍼티 (get_name()의 편의 래퍼)

        Returns:
            메트릭 이름
        """
        return self.get_name()

    @abstractmethod
    def is_higher_better(self) -> bool:
        """
        높을수록 좋은 메트릭인지 여부

        Returns:
            True: 높을수록 좋음 (예: Accuracy)
            False: 낮을수록 좋음 (예: Loss, MSE)
        """
        pass

    def get_display_format(self) -> str:
        """
        표시 형식 반환

        Returns:
            포맷 문자열 (예: ".2f", ".4f")
        """
        return ".2f"

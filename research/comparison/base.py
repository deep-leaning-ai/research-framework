"""
모델 비교 베이스 클래스
Strategy Pattern: 다양한 비교 방식을 지원
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelComparator(ABC):
    """
    모델 비교 추상 클래스

    OCP: 새로운 비교 방식 추가 시 기존 코드 수정 불필요
    SRP: 모델 비교만 담당
    """

    @abstractmethod
    def compare(self, results: Dict[str, "ExperimentResult"]) -> Dict[str, Any]:
        """
        모델들을 비교하고 결과 반환

        Args:
            results: 모델 이름을 키로 하는 ExperimentResult 딕셔너리

        Returns:
            비교 결과 딕셔너리
        """
        pass

    @abstractmethod
    def get_comparison_name(self) -> str:
        """
        비교 방식 이름 반환

        Returns:
            비교 방식 이름
        """
        pass

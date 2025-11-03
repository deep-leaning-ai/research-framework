"""
모델 비교 베이스 클래스

이 모듈은 다양한 모델 비교 방식을 위한 추상 인터페이스를 제공합니다.
Strategy Pattern을 활용하여 새로운 비교 방식을 쉽게 추가할 수 있습니다.

사용 예시:
    >>> from research.comparison.comparators import PerformanceComparator
    >>> comparator = PerformanceComparator(metric_name='accuracy', higher_is_better=True)
    >>> rankings = comparator.compare(experiment_results)
    >>> print(rankings['best_model'])

상속 예시:
    >>> class CustomComparator(ModelComparator):
    ...     def compare(self, results):
    ...         # 커스텀 비교 로직
    ...         return comparison_results
    ...     def get_comparison_name(self):
    ...         return "Custom Comparison"
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelComparator(ABC):
    """
    모델 비교를 위한 추상 베이스 클래스

    이 클래스는 다양한 모델 비교 전략의 인터페이스를 정의합니다.
    구현체는 compare()와 get_comparison_name() 메서드를 반드시 구현해야 합니다.

    Design Patterns:
        - Strategy Pattern: 다양한 비교 방식을 교체 가능
        - Template Method Pattern: 공통 비교 로직 제공 가능

    SOLID Principles:
        - OCP (Open-Closed Principle): 새로운 비교 방식 추가 시 기존 코드 수정 불필요
        - SRP (Single Responsibility Principle): 모델 비교 로직만 담당
        - DIP (Dependency Inversion Principle): 구체 클래스가 아닌 추상 인터페이스에 의존

    Attributes:
        None (추상 클래스)

    Example:
        구현체 예시는 comparators.py의 PerformanceComparator, EfficiencyComparator,
        SpeedComparator를 참조하세요.
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

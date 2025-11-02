"""
메트릭 추적 클래스
SRP: 여러 메트릭의 히스토리 관리만 담당
"""

from typing import Dict, List
import torch
from research.metrics.base import MetricCalculator


class MetricTracker:
    """
    여러 메트릭을 동시에 추적하고 관리

    여러 메트릭의 계산 결과를 에폭별로 기록하고 관리하는 클래스
    """

    def __init__(self, metrics: List[MetricCalculator]):
        """
        Args:
            metrics: 추적할 메트릭 리스트
        """
        self.metrics = metrics
        self.history: Dict[str, List[float]] = {
            metric.get_name(): [] for metric in metrics
        }

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        모든 메트릭 계산 및 기록

        Args:
            outputs: 모델 출력
            labels: 정답 레이블

        Returns:
            계산된 메트릭들의 딕셔너리
        """
        results = {}
        for metric in self.metrics:
            value = metric.calculate(outputs, labels)
            self.history[metric.get_name()].append(value)
            results[metric.get_name()] = value
        return results

    def get_history(self, metric_name: str) -> List[float]:
        """
        특정 메트릭의 히스토리 반환

        Args:
            metric_name: 메트릭 이름

        Returns:
            메트릭 값의 리스트
        """
        return self.history.get(metric_name, [])

    def get_all_history(self) -> Dict[str, List[float]]:
        """
        모든 메트릭 히스토리 반환

        Returns:
            메트릭 이름을 키로 하는 딕셔너리
        """
        return self.history

    def get_latest(self) -> Dict[str, float]:
        """
        각 메트릭의 최신 값 반환

        Returns:
            메트릭 이름을 키로 하는 딕셔너리
        """
        return {name: values[-1] for name, values in self.history.items() if values}

    def get_best(self, metric: MetricCalculator) -> float:
        """
        특정 메트릭의 최고/최저 값 반환

        Args:
            metric: 메트릭 객체

        Returns:
            최고 또는 최저 값
        """
        values = self.history[metric.get_name()]
        if not values:
            return 0.0
        return max(values) if metric.is_higher_better() else min(values)

    def reset(self):
        """히스토리 초기화"""
        for key in self.history:
            self.history[key] = []

    def summary(self) -> str:
        """
        메트릭 요약 문자열 반환

        Returns:
            요약 문자열
        """
        summary_lines = []
        for metric in self.metrics:
            name = metric.get_name()
            if name in self.history and self.history[name]:
                latest = self.history[name][-1]
                best = self.get_best(metric)
                fmt = metric.get_display_format()
                summary_lines.append(
                    f"{name}: Latest={latest:{fmt}}, Best={best:{fmt}}"
                )
        return "\n".join(summary_lines)

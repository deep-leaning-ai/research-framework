"""
실험 결과 데이터 클래스
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentResult:
    """
    실험 결과를 담는 데이터 클래스

    다중 메트릭을 지원하며 Task-agnostic 설계
    """

    model_name: str
    """모델 이름"""

    task_type: str
    """Task 타입 (MultiClassStrategy, BinaryClassificationStrategy 등)"""

    parameters: int
    """모델 파라미터 수"""

    train_metrics: Dict[str, List[float]]
    """학습 메트릭 히스토리 {"Accuracy": [...], "F1-Score": [...]}"""

    val_metrics: Dict[str, List[float]]
    """검증 메트릭 히스토리 {"Accuracy": [...], "F1-Score": [...]}"""

    test_metrics: Dict[str, List[float]]
    """테스트 메트릭 히스토리 {"Accuracy": [...], "F1-Score": [...]}"""

    train_loss: List[float]
    """학습 손실 히스토리"""

    val_loss: List[float]
    """검증 손실 히스토리"""

    test_loss: List[float]
    """테스트 손실 히스토리"""

    epoch_times: List[float]
    """에폭별 학습 시간 (초)"""

    inference_time: float
    """평균 추론 시간 (초)"""

    primary_metric_name: str
    """주요 메트릭 이름"""

    best_test_metric: float
    """테스트에서의 최고 메트릭 값"""

    final_overfitting_gap: Optional[float] = None
    """최종 과적합 갭 (분류에만 해당)"""

    additional_info: Optional[Dict[str, Any]] = None
    """추가 정보"""

    def get_final_train_metric(self, metric_name: str) -> Optional[float]:
        """
        특정 메트릭의 최종 학습 값 반환

        Args:
            metric_name: 메트릭 이름

        Returns:
            최종 값 또는 None
        """
        if metric_name in self.train_metrics and self.train_metrics[metric_name]:
            return self.train_metrics[metric_name][-1]
        return None

    def get_final_val_metric(self, metric_name: str) -> Optional[float]:
        """
        특정 메트릭의 최종 검증 값 반환

        Args:
            metric_name: 메트릭 이름

        Returns:
            최종 값 또는 None
        """
        if metric_name in self.val_metrics and self.val_metrics[metric_name]:
            return self.val_metrics[metric_name][-1]
        return None

    def get_final_test_metric(self, metric_name: str) -> Optional[float]:
        """
        특정 메트릭의 최종 테스트 값 반환

        Args:
            metric_name: 메트릭 이름

        Returns:
            최종 값 또는 None
        """
        if metric_name in self.test_metrics and self.test_metrics[metric_name]:
            return self.test_metrics[metric_name][-1]
        return None

    def get_best_test_metric_for(
        self, metric_name: str, higher_better: bool = True
    ) -> Optional[float]:
        """
        특정 메트릭의 최고 테스트 값 반환

        Args:
            metric_name: 메트릭 이름
            higher_better: 높을수록 좋은 메트릭인지

        Returns:
            최고 값 또는 None
        """
        if metric_name in self.test_metrics and self.test_metrics[metric_name]:
            values = self.test_metrics[metric_name]
            return max(values) if higher_better else min(values)
        return None

    def summary(self) -> str:
        """
        결과 요약 문자열 반환

        Returns:
            요약 문자열
        """
        lines = [
            f"Model: {self.model_name}",
            f"Task: {self.task_type}",
            f"Parameters: {self.parameters:,}",
            f"Primary Metric: {self.primary_metric_name} = {self.best_test_metric:.4f}",
            f"Inference Time: {self.inference_time*1000:.2f}ms",
        ]

        if self.final_overfitting_gap is not None:
            lines.append(f"Overfitting Gap: {self.final_overfitting_gap:.2f}")

        return "\n".join(lines)

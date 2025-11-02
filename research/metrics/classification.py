"""
분류 태스크 메트릭들
"""

import torch
from research.metrics.base import MetricCalculator
from sklearn.metrics import precision_recall_fscore_support


class AccuracyMetric(MetricCalculator):
    """정확도 메트릭"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """정확도 계산"""
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return 100.0 * correct / total

    def get_name(self) -> str:
        return "Accuracy"

    def is_higher_better(self) -> bool:
        return True


class PrecisionMetric(MetricCalculator):
    """정밀도 메트릭"""

    def __init__(self, average: str = "macro"):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
        """
        self.average = average

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """정밀도 계산"""
        _, predicted = outputs.max(1)
        pred_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()

        precision, _, _, _ = precision_recall_fscore_support(
            labels_np, pred_np, average=self.average, zero_division=0
        )
        return precision * 100

    def get_name(self) -> str:
        return f"Precision ({self.average})"

    def is_higher_better(self) -> bool:
        return True


class RecallMetric(MetricCalculator):
    """재현율 메트릭"""

    def __init__(self, average: str = "macro"):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
        """
        self.average = average

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """재현율 계산"""
        _, predicted = outputs.max(1)
        pred_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()

        _, recall, _, _ = precision_recall_fscore_support(
            labels_np, pred_np, average=self.average, zero_division=0
        )
        return recall * 100

    def get_name(self) -> str:
        return f"Recall ({self.average})"

    def is_higher_better(self) -> bool:
        return True


class F1ScoreMetric(MetricCalculator):
    """F1 Score 메트릭"""

    def __init__(self, average: str = "macro"):
        """
        Args:
            average: 평균 방식 ('macro', 'micro', 'weighted')
        """
        self.average = average

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """F1 Score 계산"""
        _, predicted = outputs.max(1)
        pred_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()

        _, _, f1, _ = precision_recall_fscore_support(
            labels_np, pred_np, average=self.average, zero_division=0
        )
        return f1 * 100

    def get_name(self) -> str:
        return f"F1-Score ({self.average})"

    def is_higher_better(self) -> bool:
        return True


class Top5AccuracyMetric(MetricCalculator):
    """Top-5 정확도 메트릭 (다중분류)"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Top-5 정확도 계산"""
        _, top5_pred = outputs.topk(5, 1, True, True)
        correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        total = labels.size(0)
        return 100.0 * correct / total

    def get_name(self) -> str:
        return "Top-5 Accuracy"

    def is_higher_better(self) -> bool:
        return True

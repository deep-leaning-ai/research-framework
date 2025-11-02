"""
회귀 태스크 메트릭들
"""

import torch
from research.metrics.base import MetricCalculator


class MSEMetric(MetricCalculator):
    """평균 제곱 오차 (Mean Squared Error)"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """MSE 계산"""
        mse = torch.mean((outputs - labels) ** 2)
        return mse.item()

    def get_name(self) -> str:
        return "MSE"

    def is_higher_better(self) -> bool:
        return False

    def get_display_format(self) -> str:
        return ".4f"


class MAEMetric(MetricCalculator):
    """평균 절대 오차 (Mean Absolute Error)"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """MAE 계산"""
        mae = torch.mean(torch.abs(outputs - labels))
        return mae.item()

    def get_name(self) -> str:
        return "MAE"

    def is_higher_better(self) -> bool:
        return False

    def get_display_format(self) -> str:
        return ".4f"


class RMSEMetric(MetricCalculator):
    """평균 제곱근 오차 (Root Mean Squared Error)"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """RMSE 계산"""
        mse = torch.mean((outputs - labels) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()

    def get_name(self) -> str:
        return "RMSE"

    def is_higher_better(self) -> bool:
        return False

    def get_display_format(self) -> str:
        return ".4f"


class R2ScoreMetric(MetricCalculator):
    """결정 계수 (R² Score)"""

    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """R² Score 계산"""
        ss_res = torch.sum((labels - outputs) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - ss_res / ss_tot
        return r2.item()

    def get_name(self) -> str:
        return "R² Score"

    def is_higher_better(self) -> bool:
        return True

    def get_display_format(self) -> str:
        return ".4f"

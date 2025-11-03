"""
Base classes for metrics

Strategy Pattern: Supports various metrics through a common interface.
"""

from abc import ABC, abstractmethod
import torch


class MetricCalculator(ABC):
    """Base abstract class for metric calculation

    OCP: New metrics can be added without modifying existing code.
    SRP: Responsible only for metric calculation.
    """

    @abstractmethod
    def calculate(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate metric

        Args:
            outputs: Model outputs
            labels: Ground truth labels

        Returns:
            Calculated metric value
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get metric name

        Returns:
            Metric name
        """
        pass

    @property
    def name(self) -> str:
        """Metric name property (convenience wrapper for get_name())

        Returns:
            Metric name
        """
        return self.get_name()

    @abstractmethod
    def is_higher_better(self) -> bool:
        """Check if higher values are better

        Returns:
            True: Higher is better (e.g., Accuracy)
            False: Lower is better (e.g., Loss, MSE)
        """
        pass

    def get_display_format(self) -> str:
        """Get display format

        Returns:
            Format string (e.g., ".2f", ".4f")
        """
        return ".2f"


# BaseMetric is an alias for MetricCalculator
# This provides a more intuitive name for the new API
BaseMetric = MetricCalculator

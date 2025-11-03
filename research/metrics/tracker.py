"""Metric tracker implementation

Manages multiple metrics simultaneously with history tracking and moving averages.
All implementation avoids hardcoding by using class constants.
"""

from typing import Dict, List, Optional, Union
import torch
import numpy as np

from .base import BaseMetric


class MetricTracker:
    """Tracks multiple metrics simultaneously

    Manages metric calculation history, moving averages, and summary statistics.
    """

    DEFAULT_WINDOW_SIZE = 10
    EMPTY_HISTORY_VALUE = 0.0

    def __init__(self, metric_names: List[str], window_size: Optional[int] = None):
        """Initialize MetricTracker

        Args:
            metric_names: List of metric names to track
            window_size: Window size for moving average (default: 10)
        """
        self.metric_names = metric_names
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self.history: Dict[str, List[float]] = {name: [] for name in metric_names}

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metrics: Dict[str, BaseMetric]
    ) -> Dict[str, float]:
        """Calculate and record all metrics

        Args:
            predictions: Model predictions
            targets: Ground truth labels/values
            metrics: Dictionary of metric instances {name: metric_instance}

        Returns:
            Dict[str, float]: Calculated metric values
        """
        results = {}

        for metric_name in self.metric_names:
            if metric_name not in metrics:
                raise ValueError(
                    f"Metric '{metric_name}' not found in provided metrics"
                )

            metric = metrics[metric_name]
            value = metric.calculate(predictions, targets)
            self.history[metric_name].append(value)
            results[metric_name] = value

        return results

    def get_latest(
        self, metric_name: Optional[str] = None
    ) -> Union[float, Dict[str, float]]:
        """Get latest metric value(s)

        Args:
            metric_name: Specific metric name (None for all metrics)

        Returns:
            float or Dict[str, float]: Latest metric value(s)
        """
        if metric_name is not None:
            # Return specific metric
            if metric_name not in self.history:
                raise ValueError(f"Unknown metric: {metric_name}")

            history = self.history[metric_name]
            if not history:
                return self.EMPTY_HISTORY_VALUE

            return history[-1]

        # Return all metrics
        results = {}
        for name, history in self.history.items():
            if history:
                results[name] = history[-1]
            else:
                results[name] = self.EMPTY_HISTORY_VALUE

        return results

    def get_best(self, metric_name: str, higher_is_better: bool = True) -> float:
        """Get best metric value

        Args:
            metric_name: Metric name
            higher_is_better: Whether higher values are better

        Returns:
            float: Best metric value
        """
        if metric_name not in self.history:
            raise ValueError(f"Unknown metric: {metric_name}")

        history = self.history[metric_name]
        if not history:
            return self.EMPTY_HISTORY_VALUE

        if higher_is_better:
            return max(history)
        else:
            return min(history)

    def get_history(
        self, metric_name: Optional[str] = None
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Get metric history

        Args:
            metric_name: Specific metric name (None for all metrics)

        Returns:
            List[float] or Dict: Metric history
        """
        if metric_name is not None:
            # Return specific metric
            if metric_name not in self.history:
                raise ValueError(f"Unknown metric: {metric_name}")

            return self.history[metric_name]

        # Return all metrics
        return self.history

    def get_moving_average(self, metric_name: str) -> float:
        """Calculate moving average

        Args:
            metric_name: Metric name

        Returns:
            float: Moving average value
        """
        if metric_name not in self.history:
            raise ValueError(f"Unknown metric: {metric_name}")

        history = self.history[metric_name]
        if not history:
            return self.EMPTY_HISTORY_VALUE

        # Take last window_size values
        window = history[-self.window_size:]
        moving_avg = sum(window) / len(window)

        return moving_avg

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics

        Returns:
            Dict: {metric_name: {mean, std, min, max, latest}}
        """
        summary_dict = {}

        for metric_name, history in self.history.items():
            if not history:
                summary_dict[metric_name] = {
                    'mean': self.EMPTY_HISTORY_VALUE,
                    'std': self.EMPTY_HISTORY_VALUE,
                    'min': self.EMPTY_HISTORY_VALUE,
                    'max': self.EMPTY_HISTORY_VALUE,
                    'latest': self.EMPTY_HISTORY_VALUE
                }
            else:
                values = np.array(history)
                summary_dict[metric_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'latest': float(history[-1])
                }

        return summary_dict

    def reset(self, metric_name: Optional[str] = None):
        """Reset metric history

        Args:
            metric_name: Specific metric name (None for all metrics)
        """
        if metric_name is not None:
            # Reset specific metric
            if metric_name not in self.history:
                raise ValueError(f"Unknown metric: {metric_name}")

            self.history[metric_name] = []
        else:
            # Reset all metrics
            for name in self.metric_names:
                self.history[name] = []

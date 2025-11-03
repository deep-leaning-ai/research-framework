"""Regression metrics implementation

All metrics avoid hardcoding by using class constants and configuration parameters.
"""

import torch

from .base import BaseMetric


class MSEMetric(BaseMetric):
    """Mean Squared Error metric for regression tasks

    Calculates the average of squared differences between predictions and targets.
    """

    METRIC_NAME = "mse"
    HIGHER_IS_BETTER = False
    EXPONENT = 2

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate MSE

        Args:
            predictions: Model predictions
                        Shape: (batch_size,) or (batch_size, 1)
            targets: Ground truth values
                    Shape: (batch_size,) or (batch_size, 1)

        Returns:
            float: MSE value (0.0 or higher)
        """
        # Ensure same shape
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate MSE
        squared_diff = (predictions - targets) ** self.EXPONENT
        mse = squared_diff.mean().item()

        return mse

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class MAEMetric(BaseMetric):
    """Mean Absolute Error metric for regression tasks

    Calculates the average of absolute differences between predictions and targets.
    """

    METRIC_NAME = "mae"
    HIGHER_IS_BETTER = False

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate MAE

        Args:
            predictions: Model predictions
                        Shape: (batch_size,) or (batch_size, 1)
            targets: Ground truth values
                    Shape: (batch_size,) or (batch_size, 1)

        Returns:
            float: MAE value (0.0 or higher)
        """
        # Ensure same shape
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate MAE
        abs_diff = torch.abs(predictions - targets)
        mae = abs_diff.mean().item()

        return mae

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class R2Metric(BaseMetric):
    """R-squared (Coefficient of Determination) metric for regression tasks

    Calculates the proportion of variance in the target explained by predictions.
    """

    METRIC_NAME = "r2_score"
    HIGHER_IS_BETTER = True
    PERFECT_SCORE = 1.0
    ZERO_VARIANCE_SCORE = 0.0
    EXPONENT = 2

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate R² score

        Args:
            predictions: Model predictions
                        Shape: (batch_size,) or (batch_size, 1)
            targets: Ground truth values
                    Shape: (batch_size,) or (batch_size, 1)

        Returns:
            float: R² value (-inf ~ 1.0, where 1.0 is perfect)
        """
        # Ensure same shape
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate residual sum of squares
        ss_res = torch.sum((targets - predictions) ** self.EXPONENT)

        # Calculate total sum of squares
        targets_mean = targets.mean()
        ss_tot = torch.sum((targets - targets_mean) ** self.EXPONENT)

        # Handle zero variance case
        if ss_tot == 0:
            return self.ZERO_VARIANCE_SCORE

        # Calculate R²
        r2 = self.PERFECT_SCORE - (ss_res / ss_tot)

        return r2.item()

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER

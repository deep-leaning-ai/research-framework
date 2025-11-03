"""Classification metrics implementation

All metrics avoid hardcoding by using class constants and configuration parameters.
"""

import torch
from typing import Optional
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from .base import BaseMetric


class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks

    Calculates the proportion of correctly classified samples.
    """

    METRIC_NAME = "accuracy"
    HIGHER_IS_BETTER = True

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy

        Args:
            predictions: Model predictions (logits or class indices)
                        Shape: (batch_size,) or (batch_size, num_classes)
            targets: Ground truth labels (class indices)
                    Shape: (batch_size,)

        Returns:
            float: Accuracy score (0.0 ~ 1.0)
        """
        # Handle 2D logits input
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        # Calculate accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()

        return accuracy

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class PrecisionMetric(BaseMetric):
    """Precision metric for classification tasks

    Calculates the proportion of true positives among positive predictions.
    """

    METRIC_NAME = "precision"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted", "binary"]
    ZERO_DIVISION_VALUE = 0.0

    def __init__(self, average: Optional[str] = None, num_classes: Optional[int] = None):
        """Initialize PrecisionMetric

        Args:
            average: Averaging method ('macro', 'micro', 'weighted', 'binary')
            num_classes: Number of classes

        Raises:
            ValueError: If average is not in VALID_AVERAGES
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(
                f"average must be one of {self.VALID_AVERAGES}, got '{self.average}'"
            )

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate precision

        Args:
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels

        Returns:
            float: Precision score (0.0 ~ 1.0)
        """
        # Handle 2D logits input
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        # Convert to numpy for sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate precision
        precision, _, _, _ = precision_recall_fscore_support(
            targets_np,
            preds_np,
            average=self.average,
            zero_division=self.ZERO_DIVISION_VALUE
        )

        return float(precision)

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class RecallMetric(BaseMetric):
    """Recall metric for classification tasks

    Calculates the proportion of true positives among actual positives.
    """

    METRIC_NAME = "recall"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted", "binary"]
    ZERO_DIVISION_VALUE = 0.0

    def __init__(self, average: Optional[str] = None, num_classes: Optional[int] = None):
        """Initialize RecallMetric

        Args:
            average: Averaging method ('macro', 'micro', 'weighted', 'binary')
            num_classes: Number of classes

        Raises:
            ValueError: If average is not in VALID_AVERAGES
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(
                f"average must be one of {self.VALID_AVERAGES}, got '{self.average}'"
            )

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate recall

        Args:
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels

        Returns:
            float: Recall score (0.0 ~ 1.0)
        """
        # Handle 2D logits input
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        # Convert to numpy for sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate recall
        _, recall, _, _ = precision_recall_fscore_support(
            targets_np,
            preds_np,
            average=self.average,
            zero_division=self.ZERO_DIVISION_VALUE
        )

        return float(recall)

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class F1ScoreMetric(BaseMetric):
    """F1 Score metric for classification tasks

    Calculates the harmonic mean of precision and recall.
    """

    METRIC_NAME = "f1_score"
    HIGHER_IS_BETTER = True
    DEFAULT_AVERAGE = "macro"
    VALID_AVERAGES = ["macro", "micro", "weighted", "binary"]

    def __init__(self, average: Optional[str] = None, num_classes: Optional[int] = None):
        """Initialize F1ScoreMetric

        Args:
            average: Averaging method ('macro', 'micro', 'weighted', 'binary')
            num_classes: Number of classes

        Raises:
            ValueError: If average is not in VALID_AVERAGES
        """
        self.average = average or self.DEFAULT_AVERAGE
        self.num_classes = num_classes

        if self.average not in self.VALID_AVERAGES:
            raise ValueError(
                f"average must be one of {self.VALID_AVERAGES}, got '{self.average}'"
            )

        # Create precision and recall metrics for reuse
        self.precision_metric = PrecisionMetric(average, num_classes)
        self.recall_metric = RecallMetric(average, num_classes)

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate F1 score

        Args:
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels

        Returns:
            float: F1 score (0.0 ~ 1.0)
        """
        # Handle 2D logits input
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)

        # Convert to numpy for sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate F1 score using sklearn
        _, _, f1, _ = precision_recall_fscore_support(
            targets_np,
            preds_np,
            average=self.average,
            zero_division=0.0
        )

        return float(f1)

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class Top5AccuracyMetric(BaseMetric):
    """Top-5 Accuracy metric for classification tasks

    Calculates the proportion of samples where the correct class
    is in the top-5 predictions.
    """

    METRIC_NAME = "top5_accuracy"
    HIGHER_IS_BETTER = True
    TOP_K = 5

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Top-5 accuracy

        Args:
            predictions: Model logits
                        Shape: (batch_size, num_classes)
            targets: Ground truth labels
                    Shape: (batch_size,)

        Returns:
            float: Top-5 accuracy (0.0 ~ 1.0)
        """
        # Predictions must be 2D (logits)
        if predictions.dim() == 1:
            raise ValueError(
                "Top5AccuracyMetric requires 2D logits input, "
                f"got {predictions.dim()}D"
            )

        batch_size, num_classes = predictions.shape

        # If num_classes < TOP_K, use num_classes
        k = min(self.TOP_K, num_classes)

        # Get top-k predictions
        _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True)

        # Check if targets are in top-k
        targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=1).float()

        # Calculate accuracy
        top5_accuracy = correct.mean().item()

        return top5_accuracy

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER


class AUCMetric(BaseMetric):
    """AUC (Area Under ROC Curve) metric for binary classification

    Calculates the area under the ROC curve.
    """

    METRIC_NAME = "auc"
    HIGHER_IS_BETTER = True

    def calculate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate AUC score

        Args:
            predictions: Probability predictions (0.0 ~ 1.0)
                        Shape: (batch_size,)
            targets: Ground truth labels (0 or 1)
                    Shape: (batch_size,)

        Returns:
            float: AUC score (0.0 ~ 1.0)
        """
        # Convert to numpy for sklearn
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate AUC using sklearn
        auc_score = roc_auc_score(targets_np, preds_np)

        return float(auc_score)

    def get_name(self) -> str:
        """Get metric name"""
        return self.METRIC_NAME

    def is_higher_better(self) -> bool:
        """Check if higher values are better"""
        return self.HIGHER_IS_BETTER

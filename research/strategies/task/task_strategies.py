"""
Task strategy implementations

All strategies avoid hardcoding by using class constants.
"""

from typing import Optional
import torch
import torch.nn as nn
from research.strategies.task.base import TaskStrategy


class MultiClassStrategy(TaskStrategy):
    """Multi-class classification strategy

    Used for tasks with N classes (N >= 2).
    Examples: MNIST, CIFAR-10, ImageNet
    """

    # Class constants - avoid hardcoding
    TASK_TYPE = "multiclass"
    DEFAULT_NUM_CLASSES = 10
    SOFTMAX_DIM = 1

    def __init__(self, num_classes: Optional[int] = None):
        """Initialize strategy

        Args:
            num_classes: Number of classes (default: 10)
        """
        self.num_classes = num_classes or self.DEFAULT_NUM_CLASSES

    def get_criterion(self) -> nn.Module:
        """Get CrossEntropyLoss (includes Softmax internally)"""
        return nn.CrossEntropyLoss()

    def get_activation(self) -> nn.Module:
        """Get Softmax activation for inference"""
        return nn.Softmax(dim=self.SOFTMAX_DIM)

    def get_output_activation(self) -> Optional[nn.Module]:
        """Legacy method - CrossEntropyLoss includes Softmax"""
        return None

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy

        Args:
            outputs: Model logits (batch_size, num_classes)
            labels: True labels (batch_size,)

        Returns:
            float: Accuracy (0.0 ~ 1.0)
        """
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def get_metric_name(self) -> str:
        """Get metric name"""
        return "Accuracy"

    def prepare_labels(self, labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
        """Prepare labels for multi-class classification

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for multi-class)

        Returns:
            torch.Tensor: Labels as Long tensor
        """
        return labels.long()

    def get_task_type(self) -> str:
        """Get task type"""
        return self.TASK_TYPE


class BinaryClassificationStrategy(TaskStrategy):
    """Binary classification strategy

    Used for tasks with 2 classes (0 or 1).
    Examples: Spam filtering, Sentiment analysis
    """

    # Class constants - avoid hardcoding
    TASK_TYPE = "binary"
    THRESHOLD = 0.5

    def get_criterion(self) -> nn.Module:
        """Get BCEWithLogitsLoss (Sigmoid + BCE combined)"""
        return nn.BCEWithLogitsLoss()

    def get_activation(self) -> nn.Module:
        """Get Sigmoid activation for inference"""
        return nn.Sigmoid()

    def get_output_activation(self) -> Optional[nn.Module]:
        """Legacy method - returns Sigmoid"""
        return nn.Sigmoid()

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy

        Args:
            outputs: Model logits (batch_size, 1) or (batch_size,)
            labels: True labels (batch_size, 1) or (batch_size,) - 0 or 1

        Returns:
            float: Accuracy (0.0 ~ 1.0)
        """
        # Apply sigmoid and threshold
        predicted = (torch.sigmoid(outputs) > self.THRESHOLD).float()

        # Flatten both for comparison
        predicted = predicted.view(-1)
        labels = labels.view(-1)

        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def get_metric_name(self) -> str:
        """Get metric name"""
        return "Accuracy"

    def prepare_labels(self, labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
        """Prepare labels for binary classification

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for binary)

        Returns:
            torch.Tensor: Labels as Float tensor with shape matching model output
        """
        labels = labels.float()
        # Ensure labels have shape [batch_size, 1] to match model output
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        return labels

    def get_task_type(self) -> str:
        """Get task type"""
        return self.TASK_TYPE


class RegressionStrategy(TaskStrategy):
    """Regression strategy

    Used for tasks predicting continuous values.
    Examples: House price prediction, Stock price prediction
    """

    # Class constants - avoid hardcoding
    TASK_TYPE = "regression"

    def get_criterion(self) -> nn.Module:
        """Get MSELoss"""
        return nn.MSELoss()

    def get_activation(self) -> Optional[nn.Module]:
        """Get activation (None for regression - linear output)"""
        return None

    def get_output_activation(self) -> Optional[nn.Module]:
        """Legacy method - regression doesn't need activation"""
        return None

    def calculate_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate MSE

        Args:
            outputs: Model predictions (batch_size, 1) or (batch_size,)
            labels: True values (batch_size,) or (batch_size, 1)

        Returns:
            float: MSE value
        """
        # Flatten tensors for consistent calculation
        outputs = outputs.view(-1)
        labels = labels.view(-1)

        mse = nn.MSELoss()(outputs, labels)
        return mse.item()

    def get_metric_name(self) -> str:
        """Get metric name"""
        return "MSE"

    def prepare_labels(self, labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
        """Prepare labels for regression

        Args:
            labels: Input labels
            num_classes: Number of classes (unused for regression)

        Returns:
            torch.Tensor: Labels as Float tensor
        """
        return labels.float()

    def get_task_type(self) -> str:
        """Get task type"""
        return self.TASK_TYPE

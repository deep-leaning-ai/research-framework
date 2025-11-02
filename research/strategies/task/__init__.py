"""
Task 전략 패키지
"""

from research.strategies.task.base import TaskStrategy
from research.strategies.task.task_strategies import (
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy
)

__all__ = [
    'TaskStrategy',
    'MultiClassStrategy',
    'BinaryClassificationStrategy',
    'RegressionStrategy',
]

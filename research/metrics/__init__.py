"""
Metrics package

Provides classification and regression metrics with unified interface.
"""

from research.metrics.base import BaseMetric, MetricCalculator
from research.metrics.classification import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    Top5AccuracyMetric,
    AUCMetric
)
from research.metrics.regression import (
    MSEMetric,
    MAEMetric,
    R2Metric
)
from research.metrics.tracker import MetricTracker

__all__ = [
    # Base classes
    'BaseMetric',
    'MetricCalculator',  # Legacy compatibility

    # Classification metrics
    'AccuracyMetric',
    'PrecisionMetric',
    'RecallMetric',
    'F1ScoreMetric',
    'Top5AccuracyMetric',
    'AUCMetric',

    # Regression metrics
    'MSEMetric',
    'MAEMetric',
    'R2Metric',

    # Tracker
    'MetricTracker',
]

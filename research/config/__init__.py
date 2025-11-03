"""Configuration package

Provides dataclass-based configuration management with validation.
All configurations avoid hardcoding by using class constants.
"""

from .base import BaseConfig
from .model import ModelConfig
from .training import TrainingConfig
from .experiment import ExperimentConfig

__all__ = [
    'BaseConfig',
    'ModelConfig',
    'TrainingConfig',
    'ExperimentConfig',
]

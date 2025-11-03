"""Base configuration classes

All configurations use dataclasses and avoid hardcoding by using class constants.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """Base configuration for all experiments

    Provides common configuration options like device selection.
    """

    # Class constants - avoid hardcoding
    DEFAULT_DEVICE = "auto"
    VALID_DEVICES = ["auto", "cuda", "cpu"]

    device: str = DEFAULT_DEVICE

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.device not in self.VALID_DEVICES:
            raise ValueError(
                f"Device must be one of {self.VALID_DEVICES}, got '{self.device}'"
            )

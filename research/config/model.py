"""Model configuration

Defines model-specific hyperparameters like num_classes and in_channels.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Model configuration

    Attributes:
        num_classes: Number of output classes
        in_channels: Number of input channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        device: Device to use ('auto', 'cuda', 'cpu')
    """

    # Class constants - avoid hardcoding
    DEFAULT_NUM_CLASSES = 10
    DEFAULT_IN_CHANNELS = 3
    VALID_IN_CHANNELS = [1, 3, 4]  # Grayscale, RGB, RGBA

    num_classes: int = DEFAULT_NUM_CLASSES
    in_channels: int = DEFAULT_IN_CHANNELS

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Call parent validation first
        super().__post_init__()

        # Validate in_channels
        if self.in_channels not in self.VALID_IN_CHANNELS:
            raise ValueError(
                f"in_channels must be one of {self.VALID_IN_CHANNELS}, "
                f"got {self.in_channels}"
            )

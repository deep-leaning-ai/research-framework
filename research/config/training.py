"""Training configuration

Defines training-specific hyperparameters like learning rate, epochs, and optimizer.
"""

from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration

    Attributes:
        learning_rate: Learning rate for optimizer
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        device: Device to use ('auto', 'cuda', 'cpu')
    """

    # Class constants - avoid hardcoding
    DEFAULT_LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6
    MAX_LEARNING_RATE = 1e-2

    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 32

    DEFAULT_OPTIMIZER = "adam"
    VALID_OPTIMIZERS = ["adam", "adamw", "sgd"]

    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    optimizer: str = DEFAULT_OPTIMIZER

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Call parent validation first
        super().__post_init__()

        # Validate learning rate
        if not self.MIN_LEARNING_RATE <= self.learning_rate <= self.MAX_LEARNING_RATE:
            raise ValueError(
                f"learning_rate must be between {self.MIN_LEARNING_RATE} "
                f"and {self.MAX_LEARNING_RATE}, got {self.learning_rate}"
            )

        # Validate optimizer
        if self.optimizer not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {self.VALID_OPTIMIZERS}, "
                f"got '{self.optimizer}'"
            )

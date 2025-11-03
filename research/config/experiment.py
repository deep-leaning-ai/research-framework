"""Experiment configuration

Combines model and training configurations into a single experiment config.
Provides backward compatibility with dictionary-based configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any

from .model import ModelConfig
from .training import TrainingConfig


@dataclass
class ExperimentConfig:
    """Complete experiment configuration

    Combines ModelConfig and TrainingConfig for a complete experiment setup.

    Attributes:
        model: Model configuration
        training: Training configuration
    """

    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary

        Provides backward compatibility with existing dictionary-based API.

        Args:
            config_dict: Configuration dictionary with keys:
                - num_classes: Number of output classes (optional)
                - in_channels: Number of input channels (optional)
                - learning_rate: Learning rate (optional)
                - max_epochs: Maximum epochs (optional)
                - batch_size: Batch size (optional)
                - optimizer: Optimizer type (optional)
                - device: Device to use (optional)

        Returns:
            ExperimentConfig: Configured experiment

        Example:
            >>> config_dict = {'num_classes': 10, 'learning_rate': 1e-4}
            >>> config = ExperimentConfig.from_dict(config_dict)
        """
        # Extract model parameters
        model_config = ModelConfig(
            num_classes=config_dict.get('num_classes', ModelConfig.DEFAULT_NUM_CLASSES),
            in_channels=config_dict.get('in_channels', ModelConfig.DEFAULT_IN_CHANNELS),
            device=config_dict.get('device', ModelConfig.DEFAULT_DEVICE)
        )

        # Extract training parameters
        training_config = TrainingConfig(
            learning_rate=config_dict.get('learning_rate', TrainingConfig.DEFAULT_LEARNING_RATE),
            max_epochs=config_dict.get('max_epochs', TrainingConfig.DEFAULT_MAX_EPOCHS),
            batch_size=config_dict.get('batch_size', TrainingConfig.DEFAULT_BATCH_SIZE),
            optimizer=config_dict.get('optimizer', TrainingConfig.DEFAULT_OPTIMIZER),
            device=config_dict.get('device', TrainingConfig.DEFAULT_DEVICE)
        )

        return cls(model=model_config, training=training_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary

        Returns:
            Dict: Configuration as dictionary

        Example:
            >>> config = ExperimentConfig(model=ModelConfig(), training=TrainingConfig())
            >>> config_dict = config.to_dict()
        """
        return {
            # Model parameters
            'num_classes': self.model.num_classes,
            'in_channels': self.model.in_channels,
            # Training parameters
            'learning_rate': self.training.learning_rate,
            'max_epochs': self.training.max_epochs,
            'batch_size': self.training.batch_size,
            'optimizer': self.training.optimizer,
            # Common parameters
            'device': self.model.device
        }

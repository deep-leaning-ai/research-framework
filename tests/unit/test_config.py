"""Unit tests for config module

All tests follow the Given-When-Then pattern and avoid hardcoding.
"""

import pytest

# Test constants
DEFAULT_NUM_CLASSES = 10
DEFAULT_IN_CHANNELS = 3
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = "adam"
DEFAULT_DEVICE = "auto"

INVALID_LEARNING_RATE_TOO_HIGH = 1.0
INVALID_LEARNING_RATE_TOO_LOW = 1e-10
INVALID_OPTIMIZER = "invalid_optimizer"
INVALID_DEVICE = "invalid_device"
INVALID_IN_CHANNELS = 2


# ============================================================================
# BaseConfig Tests
# ============================================================================

@pytest.mark.unit
class TestBaseConfig:
    """BaseConfig tests"""

    def test_default_device(self):
        """Given: BaseConfig without device parameter
        When: Create instance
        Then: Use default device"""
        # Given & When
        from research.config import BaseConfig
        config = BaseConfig()

        # Then
        assert config.device == BaseConfig.DEFAULT_DEVICE

    def test_custom_device(self):
        """Given: Custom device parameter
        When: Create BaseConfig
        Then: Use custom device"""
        # Given
        from research.config import BaseConfig
        custom_device = "cuda"

        # When
        config = BaseConfig(device=custom_device)

        # Then
        assert config.device == custom_device

    def test_invalid_device_raises_error(self):
        """Given: Invalid device parameter
        When: Create BaseConfig
        Then: Raise ValueError"""
        # Given
        from research.config import BaseConfig

        # When & Then
        with pytest.raises(ValueError, match="Device must be one of"):
            BaseConfig(device=INVALID_DEVICE)

    def test_valid_devices_constant(self):
        """Given: BaseConfig class
        When: Check VALID_DEVICES
        Then: Contains expected devices"""
        # Given & When
        from research.config import BaseConfig

        # Then
        assert "auto" in BaseConfig.VALID_DEVICES
        assert "cuda" in BaseConfig.VALID_DEVICES
        assert "cpu" in BaseConfig.VALID_DEVICES


# ============================================================================
# ModelConfig Tests
# ============================================================================

@pytest.mark.unit
class TestModelConfig:
    """ModelConfig tests"""

    def test_default_values(self):
        """Given: ModelConfig without parameters
        When: Create instance
        Then: Use default values"""
        # Given & When
        from research.config import ModelConfig
        config = ModelConfig()

        # Then
        assert config.num_classes == ModelConfig.DEFAULT_NUM_CLASSES
        assert config.in_channels == ModelConfig.DEFAULT_IN_CHANNELS
        assert config.device == ModelConfig.DEFAULT_DEVICE

    def test_custom_num_classes(self):
        """Given: Custom num_classes
        When: Create ModelConfig
        Then: Use custom num_classes"""
        # Given
        from research.config import ModelConfig
        custom_num_classes = 100

        # When
        config = ModelConfig(num_classes=custom_num_classes)

        # Then
        assert config.num_classes == custom_num_classes

    def test_custom_in_channels(self):
        """Given: Custom in_channels (1 for grayscale)
        When: Create ModelConfig
        Then: Use custom in_channels"""
        # Given
        from research.config import ModelConfig
        grayscale_channels = 1

        # When
        config = ModelConfig(in_channels=grayscale_channels)

        # Then
        assert config.in_channels == grayscale_channels

    def test_invalid_in_channels_raises_error(self):
        """Given: Invalid in_channels
        When: Create ModelConfig
        Then: Raise ValueError"""
        # Given
        from research.config import ModelConfig

        # When & Then
        with pytest.raises(ValueError, match="in_channels must be one of"):
            ModelConfig(in_channels=INVALID_IN_CHANNELS)

    def test_valid_in_channels_constant(self):
        """Given: ModelConfig class
        When: Check VALID_IN_CHANNELS
        Then: Contains 1, 3, 4"""
        # Given & When
        from research.config import ModelConfig

        # Then
        assert 1 in ModelConfig.VALID_IN_CHANNELS
        assert 3 in ModelConfig.VALID_IN_CHANNELS
        assert 4 in ModelConfig.VALID_IN_CHANNELS


# ============================================================================
# TrainingConfig Tests
# ============================================================================

@pytest.mark.unit
class TestTrainingConfig:
    """TrainingConfig tests"""

    def test_default_values(self):
        """Given: TrainingConfig without parameters
        When: Create instance
        Then: Use default values"""
        # Given & When
        from research.config import TrainingConfig
        config = TrainingConfig()

        # Then
        assert config.learning_rate == TrainingConfig.DEFAULT_LEARNING_RATE
        assert config.max_epochs == TrainingConfig.DEFAULT_MAX_EPOCHS
        assert config.batch_size == TrainingConfig.DEFAULT_BATCH_SIZE
        assert config.optimizer == TrainingConfig.DEFAULT_OPTIMIZER

    def test_custom_learning_rate(self):
        """Given: Custom learning rate
        When: Create TrainingConfig
        Then: Use custom learning rate"""
        # Given
        from research.config import TrainingConfig
        custom_lr = 1e-3

        # When
        config = TrainingConfig(learning_rate=custom_lr)

        # Then
        assert config.learning_rate == custom_lr

    def test_invalid_learning_rate_too_high_raises_error(self):
        """Given: Learning rate too high
        When: Create TrainingConfig
        Then: Raise ValueError"""
        # Given
        from research.config import TrainingConfig

        # When & Then
        with pytest.raises(ValueError, match="learning_rate must be between"):
            TrainingConfig(learning_rate=INVALID_LEARNING_RATE_TOO_HIGH)

    def test_invalid_learning_rate_too_low_raises_error(self):
        """Given: Learning rate too low
        When: Create TrainingConfig
        Then: Raise ValueError"""
        # Given
        from research.config import TrainingConfig

        # When & Then
        with pytest.raises(ValueError, match="learning_rate must be between"):
            TrainingConfig(learning_rate=INVALID_LEARNING_RATE_TOO_LOW)

    def test_custom_optimizer(self):
        """Given: Custom optimizer
        When: Create TrainingConfig
        Then: Use custom optimizer"""
        # Given
        from research.config import TrainingConfig
        custom_optimizer = "adamw"

        # When
        config = TrainingConfig(optimizer=custom_optimizer)

        # Then
        assert config.optimizer == custom_optimizer

    def test_invalid_optimizer_raises_error(self):
        """Given: Invalid optimizer
        When: Create TrainingConfig
        Then: Raise ValueError"""
        # Given
        from research.config import TrainingConfig

        # When & Then
        with pytest.raises(ValueError, match="optimizer must be one of"):
            TrainingConfig(optimizer=INVALID_OPTIMIZER)

    def test_valid_optimizers_constant(self):
        """Given: TrainingConfig class
        When: Check VALID_OPTIMIZERS
        Then: Contains adam, adamw, sgd"""
        # Given & When
        from research.config import TrainingConfig

        # Then
        assert "adam" in TrainingConfig.VALID_OPTIMIZERS
        assert "adamw" in TrainingConfig.VALID_OPTIMIZERS
        assert "sgd" in TrainingConfig.VALID_OPTIMIZERS

    def test_custom_batch_size(self):
        """Given: Custom batch size
        When: Create TrainingConfig
        Then: Use custom batch size"""
        # Given
        from research.config import TrainingConfig
        custom_batch_size = 64

        # When
        config = TrainingConfig(batch_size=custom_batch_size)

        # Then
        assert config.batch_size == custom_batch_size

    def test_custom_max_epochs(self):
        """Given: Custom max epochs
        When: Create TrainingConfig
        Then: Use custom max epochs"""
        # Given
        from research.config import TrainingConfig
        custom_epochs = 50

        # When
        config = TrainingConfig(max_epochs=custom_epochs)

        # Then
        assert config.max_epochs == custom_epochs


# ============================================================================
# ExperimentConfig Tests
# ============================================================================

@pytest.mark.unit
class TestExperimentConfig:
    """ExperimentConfig tests"""

    def test_create_with_sub_configs(self):
        """Given: ModelConfig and TrainingConfig
        When: Create ExperimentConfig
        Then: Store both configs"""
        # Given
        from research.config import ExperimentConfig, ModelConfig, TrainingConfig
        model_config = ModelConfig(num_classes=10)
        training_config = TrainingConfig(learning_rate=1e-3)

        # When
        config = ExperimentConfig(model=model_config, training=training_config)

        # Then
        assert config.model == model_config
        assert config.training == training_config

    def test_from_dict_with_all_parameters(self):
        """Given: Dictionary with all parameters
        When: Create ExperimentConfig from dict
        Then: All values are set correctly"""
        # Given
        from research.config import ExperimentConfig
        config_dict = {
            'num_classes': 10,
            'in_channels': 3,
            'learning_rate': 1e-4,
            'max_epochs': 20,
            'batch_size': 32,
            'optimizer': 'adam',
            'device': 'cuda'
        }

        # When
        config = ExperimentConfig.from_dict(config_dict)

        # Then
        assert config.model.num_classes == 10
        assert config.model.in_channels == 3
        assert config.model.device == 'cuda'
        assert config.training.learning_rate == 1e-4
        assert config.training.max_epochs == 20
        assert config.training.batch_size == 32
        assert config.training.optimizer == 'adam'

    def test_from_dict_with_partial_parameters(self):
        """Given: Dictionary with only required parameters
        When: Create ExperimentConfig from dict
        Then: Use defaults for missing values"""
        # Given
        from research.config import ExperimentConfig, ModelConfig, TrainingConfig
        config_dict = {
            'num_classes': 10,
            'learning_rate': 1e-4
        }

        # When
        config = ExperimentConfig.from_dict(config_dict)

        # Then
        assert config.model.num_classes == 10
        assert config.model.in_channels == ModelConfig.DEFAULT_IN_CHANNELS
        assert config.training.learning_rate == 1e-4
        assert config.training.max_epochs == TrainingConfig.DEFAULT_MAX_EPOCHS

    def test_from_dict_with_empty_dict(self):
        """Given: Empty dictionary
        When: Create ExperimentConfig from dict
        Then: Use all default values"""
        # Given
        from research.config import ExperimentConfig, ModelConfig, TrainingConfig
        config_dict = {}

        # When
        config = ExperimentConfig.from_dict(config_dict)

        # Then
        assert config.model.num_classes == ModelConfig.DEFAULT_NUM_CLASSES
        assert config.model.in_channels == ModelConfig.DEFAULT_IN_CHANNELS
        assert config.training.learning_rate == TrainingConfig.DEFAULT_LEARNING_RATE
        assert config.training.max_epochs == TrainingConfig.DEFAULT_MAX_EPOCHS

    def test_from_dict_with_invalid_value(self):
        """Given: Dictionary with invalid value
        When: Create ExperimentConfig from dict
        Then: Raise ValueError"""
        # Given
        from research.config import ExperimentConfig
        config_dict = {
            'learning_rate': INVALID_LEARNING_RATE_TOO_HIGH
        }

        # When & Then
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Given: ExperimentConfig instance
        When: Convert to dict
        Then: Return dictionary with all values"""
        # Given
        from research.config import ExperimentConfig, ModelConfig, TrainingConfig
        config = ExperimentConfig(
            model=ModelConfig(num_classes=10, in_channels=3),
            training=TrainingConfig(learning_rate=1e-4, max_epochs=20)
        )

        # When
        config_dict = config.to_dict()

        # Then
        assert config_dict['num_classes'] == 10
        assert config_dict['in_channels'] == 3
        assert config_dict['learning_rate'] == 1e-4
        assert config_dict['max_epochs'] == 20
        assert isinstance(config_dict, dict)

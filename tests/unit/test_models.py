"""Unit tests for model system

All tests follow the Given-When-Then pattern and avoid hardcoding.
"""

import pytest
import torch
import torch.nn as nn

# Test constants
NUM_CLASSES = 10
BATCH_SIZE = 4
RGB_CHANNELS = 3
GRAY_CHANNELS = 1
IMAGE_SIZE_224 = 224
IMAGE_SIZE_28 = 28
HIDDEN_SIZE = 128

# ResNet variants
RESNET_VARIANTS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
RESNET18 = 'resnet18'
RESNET50 = 'resnet50'

# VGG variants
VGG_VARIANTS = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
VGG16 = 'vgg16'
VGG16_BN = 'vgg16_bn'

# Training flags
PRETRAINED_TRUE = True
PRETRAINED_FALSE = False

# Freeze/unfreeze validation
REQUIRES_GRAD_TRUE = True
REQUIRES_GRAD_FALSE = False

# Block counts
MIN_BLOCKS = 1
MAX_BLOCKS = 5
TWO_BLOCKS = 2


@pytest.mark.unit
class TestResNetModel:
    """ResNetModel tests"""

    def test_resnet_model_creation(self):
        """Given: ResNet18 variant
        When: ResNetModel created
        Then: Model instance created with correct num_classes"""
        # Given
        from research.models.pretrained import ResNetModel

        # When
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # Then
        assert model is not None
        assert model.num_classes == NUM_CLASSES
        assert model.variant == RESNET18

    def test_resnet_freeze_backbone(self):
        """Given: ResNet model
        When: freeze_backbone() called
        Then: Backbone parameters have requires_grad=False"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        model.freeze_backbone()

        # Then
        for param in model.get_backbone_params():
            assert param.requires_grad == REQUIRES_GRAD_FALSE

    def test_resnet_unfreeze_all(self):
        """Given: ResNet model with frozen backbone
        When: unfreeze_all() called
        Then: All parameters have requires_grad=True"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        model.freeze_backbone()

        # When
        model.unfreeze_all()

        # Then
        for param in model.model.parameters():
            assert param.requires_grad == REQUIRES_GRAD_TRUE

    def test_resnet_1_channel_input(self):
        """Given: in_channels=1
        When: ResNet model created
        Then: conv1 input channels modified to 1"""
        # Given
        from research.models.pretrained import ResNetModel

        # When
        model = ResNetModel(
            variant=RESNET18,
            num_classes=NUM_CLASSES,
            in_channels=GRAY_CHANNELS,
            pretrained=PRETRAINED_FALSE
        )

        # Then
        assert model.model.conv1.in_channels == GRAY_CHANNELS

    def test_resnet_forward_pass(self):
        """Given: ResNet model and input tensor
        When: forward() called
        Then: Correct output shape returned"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        input_tensor = torch.randn(BATCH_SIZE, RGB_CHANNELS, IMAGE_SIZE_224, IMAGE_SIZE_224)

        # When
        output = model.forward(input_tensor)

        # Then
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_resnet_forward_1_channel(self):
        """Given: ResNet model with 1 channel and grayscale input
        When: forward() called
        Then: Correct output shape returned"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(
            variant=RESNET18,
            num_classes=NUM_CLASSES,
            in_channels=GRAY_CHANNELS,
            pretrained=PRETRAINED_FALSE
        )
        input_tensor = torch.randn(BATCH_SIZE, GRAY_CHANNELS, IMAGE_SIZE_224, IMAGE_SIZE_224)

        # When
        output = model.forward(input_tensor)

        # Then
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_resnet_all_variants(self):
        """Given: All ResNet variants
        When: Each variant model created
        Then: All variants work correctly"""
        # Given
        from research.models.pretrained import ResNetModel

        # When & Then
        for variant in RESNET_VARIANTS:
            model = ResNetModel(variant=variant, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
            assert model.variant == variant
            assert model.num_classes == NUM_CLASSES

    def test_resnet_get_model_info(self):
        """Given: ResNet model
        When: get_model_info() called
        Then: Dictionary with model information returned"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        info = model.get_model_info()

        # Then
        assert 'model_name' in info
        assert 'num_classes' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['num_classes'] == NUM_CLASSES

    def test_resnet_freeze_all(self):
        """Given: ResNet model
        When: freeze_all() called
        Then: All parameters have requires_grad=False"""
        # Given
        from research.models.pretrained import ResNetModel
        model = ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        model.freeze_all()

        # Then
        for param in model.model.parameters():
            assert param.requires_grad == REQUIRES_GRAD_FALSE

    def test_resnet_classifier_output_dim(self):
        """Given: ResNet model with custom num_classes
        When: Check fc layer output features
        Then: fc.out_features equals num_classes"""
        # Given
        from research.models.pretrained import ResNetModel
        custom_classes = 100

        # When
        model = ResNetModel(variant=RESNET18, num_classes=custom_classes, pretrained=PRETRAINED_FALSE)

        # Then
        assert model.model.fc.out_features == custom_classes


@pytest.mark.unit
class TestVGGModel:
    """VGGModel tests"""

    def test_vgg_model_creation(self):
        """Given: VGG16 variant
        When: VGGModel created
        Then: Model instance created with correct num_classes"""
        # Given
        from research.models.pretrained import VGGModel

        # When
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # Then
        assert model is not None
        assert model.num_classes == NUM_CLASSES
        assert model.variant == VGG16

    def test_vgg_freeze_backbone(self):
        """Given: VGG model
        When: freeze_backbone() called
        Then: Features parameters have requires_grad=False"""
        # Given
        from research.models.pretrained import VGGModel
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        model.freeze_backbone()

        # Then
        for param in model.get_backbone_params():
            assert param.requires_grad == REQUIRES_GRAD_FALSE

    def test_vgg_unfreeze_all(self):
        """Given: VGG model with frozen features
        When: unfreeze_all() called
        Then: All parameters have requires_grad=True"""
        # Given
        from research.models.pretrained import VGGModel
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        model.freeze_backbone()

        # When
        model.unfreeze_all()

        # Then
        for param in model.model.parameters():
            assert param.requires_grad == REQUIRES_GRAD_TRUE

    def test_vgg_partial_unfreeze_features(self):
        """Given: VGG model
        When: partial_unfreeze_features(2) called
        Then: Last 2 blocks have requires_grad=True, others False"""
        # Given
        from research.models.pretrained import VGGModel
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        model.partial_unfreeze_features(TWO_BLOCKS)

        # Then
        # At least some parameters should be trainable (classifier + last 2 blocks)
        trainable_count = sum(1 for p in model.model.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in model.model.parameters() if not p.requires_grad)

        assert trainable_count > 0
        assert frozen_count > 0  # Some early blocks should be frozen

    def test_vgg_forward_pass(self):
        """Given: VGG model and input tensor
        When: forward() called
        Then: Correct output shape returned"""
        # Given
        from research.models.pretrained import VGGModel
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        input_tensor = torch.randn(BATCH_SIZE, RGB_CHANNELS, IMAGE_SIZE_224, IMAGE_SIZE_224)

        # When
        output = model.forward(input_tensor)

        # Then
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_vgg_1_channel_input(self):
        """Given: in_channels=1
        When: VGG model created
        Then: First conv input channels modified to 1"""
        # Given
        from research.models.pretrained import VGGModel

        # When
        model = VGGModel(
            variant=VGG16,
            num_classes=NUM_CLASSES,
            in_channels=GRAY_CHANNELS,
            pretrained=PRETRAINED_FALSE
        )

        # Then
        assert model.model.features[0].in_channels == GRAY_CHANNELS

    def test_vgg_all_variants(self):
        """Given: All VGG variants
        When: Each variant model created
        Then: All variants work correctly"""
        # Given
        from research.models.pretrained import VGGModel

        # When & Then
        for variant in VGG_VARIANTS:
            model = VGGModel(variant=variant, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
            assert model.variant == variant
            assert model.num_classes == NUM_CLASSES

    def test_vgg_get_model_info(self):
        """Given: VGG model
        When: get_model_info() called
        Then: Dictionary with model information returned"""
        # Given
        from research.models.pretrained import VGGModel
        model = VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # When
        info = model.get_model_info()

        # Then
        assert 'model_name' in info
        assert 'num_classes' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['num_classes'] == NUM_CLASSES

    def test_vgg_classifier_output_dim(self):
        """Given: VGG model with custom num_classes
        When: Check classifier[6] output features
        Then: classifier[6].out_features equals num_classes"""
        # Given
        from research.models.pretrained import VGGModel
        custom_classes = 100

        # When
        model = VGGModel(variant=VGG16, num_classes=custom_classes, pretrained=PRETRAINED_FALSE)

        # Then
        assert model.model.classifier[6].out_features == custom_classes


@pytest.mark.unit
class TestCNN:
    """CNN tests"""

    def test_cnn_creation(self):
        """Given: CNN parameters
        When: CNN created
        Then: Model instance created"""
        # Given
        from research.models.simple import CNN

        # When
        model = CNN(output_dim=NUM_CLASSES)

        # Then
        assert model is not None
        assert model.output_dim == NUM_CLASSES

    def test_cnn_forward_pass(self):
        """Given: CNN and MNIST-shaped input
        When: forward() called
        Then: Correct output shape returned"""
        # Given
        from research.models.simple import CNN
        model = CNN(output_dim=NUM_CLASSES)
        input_tensor = torch.randn(BATCH_SIZE, GRAY_CHANNELS, IMAGE_SIZE_28, IMAGE_SIZE_28)

        # When
        output = model.forward(input_tensor)

        # Then
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_cnn_get_model_info(self):
        """Given: CNN model
        When: get_model_info() called
        Then: Dictionary with model information returned"""
        # Given
        from research.models.simple import CNN
        model = CNN(output_dim=NUM_CLASSES)

        # When
        info = model.get_model_info()

        # Then
        assert 'name' in info
        assert 'type' in info
        assert 'output_dim' in info
        assert info['output_dim'] == NUM_CLASSES

    def test_cnn_architecture_layers(self):
        """Given: CNN model
        When: Check layer existence
        Then: layer1, layer2, fc exist"""
        # Given
        from research.models.simple import CNN
        model = CNN(output_dim=NUM_CLASSES)

        # When & Then
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'fc')


@pytest.mark.unit
class TestFullyConnectedNN:
    """FullyConnectedNN tests"""

    def test_fc_creation(self):
        """Given: FullyConnectedNN parameters
        When: FullyConnectedNN created
        Then: Model instance created"""
        # Given
        from research.models.simple import FullyConnectedNN

        # When
        model = FullyConnectedNN(hidden_size=HIDDEN_SIZE, output_dim=NUM_CLASSES)

        # Then
        assert model is not None
        assert model.output_dim == NUM_CLASSES
        assert model.hidden_size == HIDDEN_SIZE

    def test_fc_forward_pass(self):
        """Given: FullyConnectedNN and input
        When: forward() called
        Then: Correct output shape returned"""
        # Given
        from research.models.simple import FullyConnectedNN
        model = FullyConnectedNN(hidden_size=HIDDEN_SIZE, output_dim=NUM_CLASSES)
        input_tensor = torch.randn(BATCH_SIZE, GRAY_CHANNELS, IMAGE_SIZE_28, IMAGE_SIZE_28)

        # When
        output = model.forward(input_tensor)

        # Then
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_fc_get_model_info(self):
        """Given: FullyConnectedNN model
        When: get_model_info() called
        Then: Dictionary with model information returned"""
        # Given
        from research.models.simple import FullyConnectedNN
        model = FullyConnectedNN(hidden_size=HIDDEN_SIZE, output_dim=NUM_CLASSES)

        # When
        info = model.get_model_info()

        # Then
        assert 'name' in info
        assert 'type' in info
        assert 'output_dim' in info
        assert 'hidden_size' in info
        assert info['output_dim'] == NUM_CLASSES
        assert info['hidden_size'] == HIDDEN_SIZE

    def test_fc_architecture_layers(self):
        """Given: FullyConnectedNN model
        When: Check layer existence
        Then: fc1, fc2, fc3 exist"""
        # Given
        from research.models.simple import FullyConnectedNN
        model = FullyConnectedNN(hidden_size=HIDDEN_SIZE, output_dim=NUM_CLASSES)

        # When & Then
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')


@pytest.mark.unit
class TestModelRegistry:
    """ModelRegistry tests"""

    def test_registry_create_resnet(self):
        """Given: 'resnet18' registered
        When: ModelRegistry.create('resnet18') called
        Then: ResNetModel instance returned"""
        # Given
        from research.models.pretrained import ModelRegistry, ResNetModel

        # When
        model = ModelRegistry.create(RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # Then
        assert isinstance(model, ResNetModel)
        assert model.num_classes == NUM_CLASSES

    def test_registry_create_vgg(self):
        """Given: 'vgg16' registered
        When: ModelRegistry.create('vgg16') called
        Then: VGGModel instance returned"""
        # Given
        from research.models.pretrained import ModelRegistry, VGGModel

        # When
        model = ModelRegistry.create(VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)

        # Then
        assert isinstance(model, VGGModel)
        assert model.num_classes == NUM_CLASSES

    def test_registry_list_models(self):
        """Given: Multiple models registered
        When: list_models() called
        Then: Sorted model name list returned"""
        # Given
        from research.models.pretrained import ModelRegistry

        # When
        models = ModelRegistry.list_models()

        # Then
        assert isinstance(models, list)
        assert len(models) > 0
        assert RESNET18 in models
        assert VGG16 in models
        # Check sorted
        assert models == sorted(models)

    def test_registry_unregistered_model(self):
        """Given: Unregistered model name
        When: create('unknown') called
        Then: ValueError raised"""
        # Given
        from research.models.pretrained import ModelRegistry
        unknown_model = 'unknown_model_xyz'

        # When & Then
        with pytest.raises(ValueError) as exc_info:
            ModelRegistry.create(unknown_model)

        assert unknown_model in str(exc_info.value)

    def test_registry_is_registered(self):
        """Given: Registered and unregistered models
        When: is_registered() called
        Then: Correct boolean returned"""
        # Given
        from research.models.pretrained import ModelRegistry
        unknown_model = 'unknown_model_xyz'

        # When & Then
        assert ModelRegistry.is_registered(RESNET18) is True
        assert ModelRegistry.is_registered(VGG16) is True
        assert ModelRegistry.is_registered(unknown_model) is False

    def test_registry_get_model_info(self):
        """Given: Registered model
        When: get_model_info() called
        Then: Model info dictionary returned"""
        # Given
        from research.models.pretrained import ModelRegistry

        # When
        info = ModelRegistry.get_model_info(RESNET18)

        # Then
        assert isinstance(info, dict)
        assert 'name' in info
        assert 'class' in info
        assert info['name'] == RESNET18


@pytest.mark.unit
class TestModelInterchangeability:
    """Test model interchangeability (LSP)"""

    def test_all_pretrained_models_have_same_interface(self):
        """Given: All pretrained model classes
        When: Check methods
        Then: All have same interface"""
        # Given
        from research.models.pretrained import ResNetModel, VGGModel

        models = [
            ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE),
            VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        ]

        # When & Then
        for model in models:
            assert hasattr(model, 'freeze_backbone')
            assert hasattr(model, 'unfreeze_all')
            assert hasattr(model, 'freeze_all')
            assert hasattr(model, 'get_backbone_params')
            assert hasattr(model, 'get_model_info')
            assert hasattr(model, 'forward')

    def test_all_simple_models_have_forward(self):
        """Given: All simple model classes
        When: Check forward method
        Then: All have forward method"""
        # Given
        from research.models.simple import CNN, FullyConnectedNN

        models = [
            CNN(output_dim=NUM_CLASSES),
            FullyConnectedNN(hidden_size=HIDDEN_SIZE, output_dim=NUM_CLASSES)
        ]

        # When & Then
        for model in models:
            assert hasattr(model, 'forward')
            assert hasattr(model, 'get_model_info')

    def test_all_pretrained_models_return_correct_shape(self):
        """Given: All pretrained models
        When: forward() with same input
        Then: All return same output shape"""
        # Given
        from research.models.pretrained import ResNetModel, VGGModel

        models = [
            ResNetModel(variant=RESNET18, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE),
            VGGModel(variant=VGG16, num_classes=NUM_CLASSES, pretrained=PRETRAINED_FALSE)
        ]
        input_tensor = torch.randn(BATCH_SIZE, RGB_CHANNELS, IMAGE_SIZE_224, IMAGE_SIZE_224)

        # When & Then
        for model in models:
            output = model.forward(input_tensor)
            assert output.shape == (BATCH_SIZE, NUM_CLASSES)

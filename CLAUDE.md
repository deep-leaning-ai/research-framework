# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**KTB ML Framework** (`research` package) is a unified machine learning experiment framework for transfer learning and general ML tasks. It merges functionality from two previous frameworks (ktb_dl_research and ml_framework) into a comprehensive, SOLID-principle-based architecture.

Key characteristics:
- No hardcoding philosophy: All configuration values defined as class constants
- SOLID principles with Strategy Pattern, Template Method, Facade Pattern, and Dependency Injection
- TDD approach with Given-When-Then pattern
- Task-agnostic design supporting multiple ML tasks through strategy abstractions

## Common Commands

### Installation
```bash
pip install -e .              # Editable install
pip install -e ".[dev]"       # With dev dependencies
pip install -e ".[all]"       # All optional dependencies
```

### Testing
```bash
pytest                        # Run all tests
pytest -v                     # Verbose output
pytest -m unit                # Only unit tests
pytest -m "not slow"          # Skip slow tests
pytest --cov=research         # With coverage report
```

### Code Quality
```bash
black research tests --line-length 100    # Format code
isort research tests --profile black      # Sort imports
flake8 research                           # Lint
mypy research                             # Type check
```

### Running Examples
```bash
python examples/quickstart.py                  # Quick start demo
python examples/test_metric_system.py          # Metric system demo
python examples/test_transfer_learning.py      # Transfer learning demo
```

## Architecture

### Module Organization

```
research/
├── core/              # Core abstractions (BaseModel, Experiment, base strategies)
├── models/            # Model implementations
│   ├── pretrained/    # Transfer learning models (ResNet, VGG) with ModelRegistry
│   └── simple/        # Simple models (CNN, FullyConnected)
├── strategies/        # Strategy Pattern implementations
│   ├── training/      # Training strategies (VanillaTrainingStrategy)
│   ├── logging/       # Logging strategies (Simple, WandB)
│   └── task/          # Task strategies (MultiClass, Binary, Regression)
├── metrics/           # Metric calculators (Classification, Regression) + MetricTracker
├── experiment/        # ExperimentRunner, ExperimentResult, ExperimentRecorder
├── comparison/        # Model comparison system (comparators + ComparisonManager)
├── visualization/     # ExperimentVisualizer with 8-panel charts
├── data/              # Data modules (CIFAR10DataModule, DataLoaderFactory)
└── config/            # Configuration classes using dataclasses
```

### Key Design Patterns

1. **Template Method Pattern**: `BaseModel` defines common flow (setup_model, forward, train_step, etc.), subclasses implement specific details
2. **Strategy Pattern**: Used for tasks (loss/activation/metrics per task type), training, logging, metrics, and comparison
3. **Facade Pattern**: `Experiment` class provides simple API hiding complex orchestration
4. **Registry Pattern**: `ModelRegistry` for pretrained model management with automatic registration
5. **Dependency Injection**: Components depend on abstractions (base classes), not concrete implementations

### Key Abstractions

1. **TaskStrategy**: Defines loss function, activation, and metric calculation per task type
   - `MultiClassStrategy` (CrossEntropyLoss, no activation for logits)
   - `BinaryClassificationStrategy` (BCELoss, Sigmoid)
   - `RegressionStrategy` (MSELoss, no activation)

2. **MetricCalculator**: Base for all metrics with `calculate()`, `get_name()`, `is_higher_better()`
   - Classification: Accuracy, Precision, Recall, F1Score, Top5Accuracy, AUC
   - Regression: MSE, MAE, R2

3. **TrainingStrategy**: Handles training loop execution
   - `VanillaTrainingStrategy` (basic training with validation)

4. **LoggingStrategy**: Handles experiment tracking
   - `SimpleLoggingStrategy` (console output)
   - `WandBLoggingStrategy` (Weights & Biases integration)

### Core Workflow

```python
# 1. Configure
config = {'num_classes': 10, 'learning_rate': 1e-4, 'max_epochs': 2}
exp = Experiment(config)

# 2. Setup
exp.setup(
    model_name='resnet18',
    data_module=CIFAR10DataModule(),
    training_strategy=VanillaTrainingStrategy(task_strategy=MultiClassStrategy()),
    logging_strategy=SimpleLoggingStrategy()
)

# 3. Run single strategy
result = exp.run(strategy='fine_tuning', run_name='demo')

# 4. Compare strategies
comparison = exp.compare_strategies(['feature_extraction', 'fine_tuning'])

# 5. Visualize
ExperimentVisualizer.plot_comparison(recorder, save_path='comparison.png')
```

### Transfer Learning Strategies

- `feature_extraction`: Freeze backbone, train only classifier
- `fine_tuning`: Unfreeze all layers for full training
- `inference`: Freeze all layers for evaluation only

## Testing Conventions

### Test Organization
- `tests/conftest.py`: Comprehensive fixtures with TDD constants (PERFECT_ACCURACY, NUM_SAMPLES, etc.)
- `tests/unit/`: 14 unit test files covering all modules
- `tests/integration/`: End-to-end integration tests

### Pytest Markers
Use markers to filter tests:
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow-running tests (skip with `-m "not slow"`)
- `@pytest.mark.gpu`: GPU-required tests

### Test Pattern
Follow Given-When-Then pattern:
```python
def test_metric_calculation(perfect_predictions, imperfect_predictions):
    # Given
    calculator = AccuracyCalculator()

    # When
    accuracy = calculator.calculate(perfect_predictions, perfect_predictions)

    # Then
    assert accuracy == PERFECT_ACCURACY
```

All test constants defined in conftest.py fixtures, no hardcoded values in tests.

## Important Architectural Decisions

1. **Defensive Imports**: All imports in `__init__.py` wrapped in try-except to prevent package-wide import failures if optional dependencies missing

2. **No Hardcoding**: All magic numbers, thresholds, and configuration values defined as class constants (e.g., `DEFAULT_LEARNING_RATE`, `NUM_CLASSES`)

3. **Multi-channel Support**: Models support both 1-channel inputs (grayscale/mel-spectrogram) and 3-channel inputs (RGB) through dynamic channel adaptation

4. **Backwards Compatibility**: Maintains API compatibility with original ktb_dl_research framework

5. **Strategy Composition**: Task strategies compose with training strategies, allowing flexible experiment configurations without class explosion

## Implemented Features Index

This section maps user intentions to specific code locations for quick reference when requesting implementations or modifications.

### 1. Data Loading & Processing

**CIFAR10 Data Module**
- File: `research/data/cifar10.py`
- Class: `CIFAR10DataModule`
- Key Methods: `prepare_data()`, `setup()`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `get_class_names()`
- Features: ImageNet/CIFAR-10 normalization, automatic train/val split (80/20), persistent workers optimization

**Data Loader Factory**
- File: `research/data/loaders.py`
- Class: `DataLoaderFactory`
- Key Method: `create_loaders()` (static)
- Features: Custom dataset support, configurable train_ratio, automatic splitting, reproducible random seed

### 2. Models

**Pretrained Models (ResNet)**
- File: `research/models/pretrained/resnet.py`
- Class: `ResNetModel`
- Variants: resnet18, resnet34, resnet50, resnet101, resnet152
- Key Methods: `freeze_backbone()`, `unfreeze_all()`, `freeze_until_layer()`, `get_layer_groups()`
- Features: Multi-channel input support (1, 3, 4 channels), automatic weight averaging for grayscale

**Pretrained Models (VGG)**
- File: `research/models/pretrained/vgg.py`
- Class: `VGGModel`
- Variants: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
- Key Methods: `freeze_features()`, `unfreeze_classifier_only()`, `partial_unfreeze_features()`, `get_architecture_info()`
- Features: Block-wise unfreezing, multi-channel input support

**Model Registry**
- File: `research/models/pretrained/registry.py`
- Class: `ModelRegistry`
- Key Methods: `register()` (decorator), `create()`, `list_models()`, `is_registered()`, `get_model_info()`
- Registered Models: 13 total (5 ResNet + 8 VGG variants)

**Simple Models**
- CNN: `research/models/simple/cnn.py` - For MNIST-like 28x28 images (2 Conv + AvgPool + FC)
- FullyConnectedNN: `research/models/simple/fully_connected.py` - 3 FC layers with configurable hidden size

### 3. Training Strategies

**Vanilla Training**
- File: `research/strategies/training/vanilla_strategy.py`
- Class: `VanillaTrainingStrategy`
- Key Methods: `train()`, `evaluate()`
- Features: Task-agnostic, optimizer support (Adam/AdamW/SGD), learning rate scheduling (ReduceLROnPlateau), early stopping, gradient clipping

**Transfer Learning Modes** (via Experiment facade)
- `feature_extraction`: Freeze backbone, train classifier only
- `fine_tuning`: Train entire network
- `inference`: Freeze all, evaluation only

### 4. Task Strategies

All in `research/strategies/task/task_strategies.py`:

- **MultiClassStrategy**: CrossEntropyLoss, Softmax, Accuracy metric
- **BinaryClassificationStrategy**: BCEWithLogitsLoss, Sigmoid, Accuracy (threshold=0.5)
- **RegressionStrategy**: MSELoss, no activation, MSE metric

### 5. Metrics

**Classification Metrics** (`research/metrics/classification.py`)
- `AccuracyMetric`: Proportion of correct predictions
- `PrecisionMetric`: True positives / positive predictions (supports macro/micro/weighted/binary)
- `RecallMetric`: True positives / actual positives (supports averaging)
- `F1ScoreMetric`: Harmonic mean of precision and recall (supports averaging)
- `Top5AccuracyMetric`: Correct class in top-5 predictions
- `AUCMetric`: Area under ROC curve (binary classification)

**Regression Metrics** (`research/metrics/regression.py`)
- `MSEMetric`: Mean Squared Error
- `MAEMetric`: Mean Absolute Error
- `R2Metric`: R-squared (coefficient of determination)

**Metric Tracking**
- File: `research/metrics/tracker.py`
- Class: `MetricTracker`
- Key Methods: `update()`, `get_latest()`, `get_best()`, `get_history()`, `get_moving_average()`, `summary()`, `reset()`
- Features: Multiple metrics tracking, moving averages (window_size=10)

### 6. Logging Strategies

**Simple Logging**
- File: `research/strategies/logging/simple_strategy.py`
- Class: `SimpleLoggingStrategy`
- Features: Console output, no dependencies

**Weights & Biases Logging**
- File: `research/strategies/logging/wandb_strategy.py`
- Class: `WandBLoggingStrategy`
- Features: WandB integration, automatic fallback if not installed

Both implement: `init_run()`, `log_metrics()`, `log_hyperparams()`, `log_artifact()`, `finish()`, `get_history()`

### 7. Experiment Management

**Experiment Facade**
- File: `research/core/experiment.py`
- Class: `Experiment`
- Key Methods: `setup()`, `run()`, `compare_strategies()`, `evaluate_pretrained()`, `get_history()`, `save_results()`
- Features: Model state management, automatic reset, history tracking

**Experiment Runner**
- File: `research/experiment/runner.py`
- Class: `ExperimentRunner`
- Key Methods: `run_single_experiment()`, `run_multiple_experiments()`, `get_recorder()`
- Features: Task-agnostic design, metric tracking, inference time measurement, overfitting gap calculation

**Experiment Recorder**
- File: `research/experiment/recorder.py`
- Class: `ExperimentRecorder`
- Key Methods: `add_result()`, `get_result()`, `get_all_results()`, `print_summary()`, `save_to_file()`, `get_best_model()`
- Features: Memory management (max 100 results), version control, auto-save every 10 results

**Experiment Result**
- File: `research/experiment/result.py`
- Class: `ExperimentResult` (dataclass)
- Attributes: model_name, task_type, parameters, train/val/test metrics and losses, epoch_times, inference_time, primary_metric_name
- Key Methods: `get_final_train_metric()`, `get_final_val_metric()`, `get_final_test_metric()`, `get_best_test_metric_for()`, `summary()`

### 8. Comparison & Analysis

**Comparators** (`research/comparison/comparators.py`)
- `PerformanceComparator`: Ranks by metric performance
- `EfficiencyComparator`: Calculates performance / log10(parameters) ratio
- `SpeedComparator`: Compares inference and training time

**Comparison Manager**
- File: `research/comparison/manager.py`
- Class: `ComparisonManager`
- Key Methods: `add_comparator()`, `compare()`, `run_all_comparisons()`, `export_comparison_report()`, `generate_report()`, `print_summary()`
- Features: Multiple comparator orchestration, automatic report generation

### 9. Visualization

**Experiment Visualizer**
- File: `research/visualization/visualizer.py`
- Class: `ExperimentVisualizer`
- Key Methods: `plot_comparison()` (8-panel), `plot_metric_comparison()` (2-panel)
- Features: Dynamic color generation for 20+ models, high-resolution output (300 DPI)

**Available Chart Types:**
1. Training Progress (Train vs Val Loss)
2. Final Test Performance (Test Loss)
3. Primary Metric Comparison (Train/Val/Test)
4. Best Performance Bar Chart
5. Parameter Efficiency Scatter Plot
6. Average Training Time per Epoch
7. Average Inference Time
8. Overfitting Gap (Train-Val difference)

### 10. Configuration

**Config Classes** (all using dataclasses)
- `ModelConfig`: `research/config/model.py` - num_classes, in_channels, device
- `TrainingConfig`: `research/config/training.py` - learning_rate, max_epochs, batch_size, optimizer
- `ExperimentConfig`: `research/config/experiment.py` - Combines ModelConfig + TrainingConfig, has `from_dict()` and `to_dict()`

## Quick Reference

### When you want to...

**Load CIFAR-10 data:**
```python
from research.data.cifar10 import CIFAR10DataModule
dm = CIFAR10DataModule(data_dir='./data', batch_size=32)
dm.prepare_data()
dm.setup()
```

**Load custom data:**
```python
from research.data.loaders import DataLoaderFactory
train_loader, val_loader, test_loader = DataLoaderFactory.create_loaders(
    train_dataset=your_train_dataset,
    test_dataset=your_test_dataset,
    batch_size=32
)
```

**Create a pretrained model:**
```python
from research.models.pretrained.registry import ModelRegistry
# List available models
models = ModelRegistry.list_models()  # ['resnet18', 'resnet34', ..., 'vgg19_bn']
# Create model
model = ModelRegistry.create('resnet50', num_classes=10)
```

**Create a simple model:**
```python
from research.models.simple.cnn import CNN
from research.models.simple.fully_connected import FullyConnectedNN

cnn = CNN(output_dim=10)  # For MNIST-like tasks
fcnn = FullyConnectedNN(input_dim=784, hidden_dim=128, output_dim=10)
```

**Setup a training strategy:**
```python
from research.strategies.training.vanilla_strategy import VanillaTrainingStrategy
from research.strategies.task.task_strategies import MultiClassStrategy

task_strategy = MultiClassStrategy(num_classes=10)
training_strategy = VanillaTrainingStrategy(
    task_strategy=task_strategy,
    optimizer_name='adam',
    learning_rate=1e-4,
    use_scheduler=True,
    early_stopping_patience=5
)
```

**Run an experiment (easy way):**
```python
from research.core.experiment import Experiment

config = {'num_classes': 10, 'learning_rate': 1e-4, 'max_epochs': 10}
exp = Experiment(config)
exp.setup(
    model_name='resnet18',
    data_module=CIFAR10DataModule(),
    training_strategy=training_strategy,
    logging_strategy=SimpleLoggingStrategy()
)

# Run single strategy
result = exp.run(strategy='fine_tuning', run_name='my_experiment')

# Compare multiple strategies
comparison = exp.compare_strategies(['feature_extraction', 'fine_tuning'])
```

**Run experiment (advanced way):**
```python
from research.experiment.runner import ExperimentRunner

runner = ExperimentRunner(training_strategy=training_strategy, logging_strategy=logging_strategy)
result = runner.run_single_experiment(
    model=model,
    data_module=data_module,
    config=config,
    experiment_name='my_experiment'
)
```

**Compare models:**
```python
from research.comparison.manager import ComparisonManager
from research.comparison.comparators import PerformanceComparator, EfficiencyComparator

manager = ComparisonManager()
manager.add_comparator(PerformanceComparator(metric_name='accuracy', higher_better=True))
manager.add_comparator(EfficiencyComparator(metric_name='accuracy'))

# Compare all experiments
comparison_results = manager.run_all_comparisons(experiment_results)
manager.print_summary()
```

**Visualize results:**
```python
from research.visualization.visualizer import ExperimentVisualizer

# 8-panel comprehensive comparison
ExperimentVisualizer.plot_comparison(
    recorder=recorder,
    save_path='comparison.png',
    primary_metric='accuracy'
)

# Focused 2-panel metric comparison
ExperimentVisualizer.plot_metric_comparison(
    recorder=recorder,
    metric_name='accuracy',
    save_path='accuracy_comparison.png'
)
```

**Track custom metrics:**
```python
from research.metrics.tracker import MetricTracker
from research.metrics.classification import AccuracyMetric, F1ScoreMetric

tracker = MetricTracker(['accuracy', 'f1_score'])
metrics = {
    'accuracy': AccuracyMetric(),
    'f1_score': F1ScoreMetric(average='weighted')
}

# During training loop
tracker.update(predictions, targets, metrics)
latest = tracker.get_latest()
best = tracker.get_best('accuracy')
summary = tracker.summary()
```

**Use different task types:**
```python
from research.strategies.task.task_strategies import (
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy
)

# Multi-class classification (e.g., CIFAR-10)
task = MultiClassStrategy(num_classes=10)

# Binary classification
task = BinaryClassificationStrategy()

# Regression
task = RegressionStrategy()
```

## Development Guidelines

- No emojis in code or comments
- Use TDD with Given-When-Then pattern for all tests
- Define all configuration values as class constants, never hardcode
- Follow SOLID principles when extending functionality
- Add type hints for better IDE support
- Write comprehensive docstrings with usage examples
- Use defensive programming with input validation

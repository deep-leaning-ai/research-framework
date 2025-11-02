# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KTB ML Framework is a unified machine learning experimentation framework that integrates transfer learning (pretrained models like ResNet, VGG) with general ML tasks (multiclass, binary, regression). Built with SOLID principles and design patterns (Strategy, Factory, Template Method, Facade, Observer).

**Package Name**: `research` (installed as `research`, legacy name was `ktb_dl_research`)

## Development Commands

### Installation & Setup

```bash
# Basic installation (development mode)
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install development dependencies only
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_metrics.py

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=research --cov-report=term-missing

# Run tests by marker
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m slow
pytest tests/ -m gpu
```

### Code Quality

```bash
# Format code with black
black research/

# Sort imports with isort
isort research/

# Lint with flake8
flake8 research/

# Type checking with mypy
mypy research/
```

### Running Examples

```bash
# Quick start example (full workflow)
python3 examples/quickstart.py

# Metric system test
python3 examples/test_metric_system.py

# Visualization test
python3 examples/test_visualization.py

# Task strategies test
python3 examples/test_task_strategies.py

# Comparison system test
python3 examples/test_comparison_system.py
```

## Architecture Overview

### Package Structure

```
research/
├── core/              # Abstract base classes (BaseModel, Experiment)
├── models/
│   ├── pretrained/    # Transfer learning models (ResNet, VGG) + ModelRegistry
│   └── simple/        # Simple models (CNN, FullyConnectedNN)
├── strategies/
│   ├── training/      # Training strategies (VanillaTrainingStrategy)
│   ├── logging/       # Logging strategies (Simple, WandB)
│   ├── task/          # Task strategies (MultiClass, Binary, Regression)
│   └── optimization/  # Future optimization strategies
├── metrics/           # Metric system (MetricTracker, individual metrics)
├── experiment/        # Experiment management (Runner, Recorder, Result)
├── comparison/        # Model comparison (Performance, Efficiency, Speed comparators)
├── visualization/     # Visualization tools (8-panel charts, confusion matrix)
├── analysis/          # Analysis tools (legacy + new analyzers)
├── data/              # Data modules (CIFAR10DataModule, DataLoaderFactory)
├── utils/             # Utility functions
└── compat/            # Backward compatibility layer
```

### Key Design Patterns

1. **Strategy Pattern**: Used extensively for task strategies, training strategies, logging strategies, and comparators. New strategies can be added without modifying existing code.

2. **Factory + Registry Pattern**: `ModelRegistry` uses decorators to automatically register pretrained models. Models are created via `ModelRegistry.create(model_name, ...)`.

3. **Template Method Pattern**: `BaseModel` provides the template for transfer learning models. Subclasses implement `_load_pretrained()`, `_modify_classifier()`, and `get_backbone_params()`.

4. **Facade Pattern**: `Experiment` class provides a simplified interface for complex workflows (setup, training, comparison).

5. **Observer Pattern**: `ExperimentRecorder` automatically collects and manages experiment results.

### Core Components

**Transfer Learning System**:
- Supports ResNet (18/34/50/101/152) and VGG (11/13/16/19 + BN variants)
- Three modes: `feature_extraction` (freeze backbone), `fine_tuning` (train all), `inference` (eval only)
- Methods: `freeze_backbone()`, `unfreeze_all()`, `freeze_all()`, `partial_unfreeze()`

**Task Strategies**:
- `MultiClassStrategy`: CrossEntropyLoss, Softmax activation, accuracy metric
- `BinaryClassificationStrategy`: BCEWithLogitsLoss, Sigmoid activation
- `RegressionStrategy`: MSELoss, no activation function

**Metrics System**:
- `MetricTracker`: Manages multiple metrics simultaneously with history tracking
- Classification: AccuracyMetric, PrecisionMetric, RecallMetric, F1ScoreMetric
- Regression: MSEMetric, MAEMetric, R2Metric
- Custom metrics: Inherit from `BaseMetric` and implement `calculate(predictions, targets)`

**Comparison System**:
- `PerformanceComparator`: Ranks by metric values
- `EfficiencyComparator`: Ranks by parameter efficiency (performance / log10(params))
- `SpeedComparator`: Compares training and inference speed
- `ComparisonManager`: Orchestrates multiple comparators and generates reports

**Visualization**:
- `ExperimentVisualizer.plot_comparison()`: Creates 8-panel comprehensive charts
- Includes: train/val loss, test loss, metric comparison, best performance, efficiency scatter, epoch time, inference time, overfitting gap
- Legacy functions: `visualize_samples()`, `plot_confusion_matrix()`, `plot_comprehensive_comparison()`, `plot_accuracy_improvement()`

### Main Entry Points

The package exports a unified API through `research/__init__.py`. All imports use defensive try-except blocks to prevent import failures.

Main classes to use:
- `Experiment`: High-level experiment orchestration
- `ExperimentRunner`: Lower-level experiment execution
- `ExperimentRecorder`: Collects and manages multiple experiment results
- `MetricTracker`: Tracks multiple metrics during training
- `ComparisonManager`: Compares models across different dimensions

## Common Development Tasks

### Adding a New Pretrained Model

1. Create model class in `research/models/pretrained/`
2. Inherit from `BaseModel`
3. Implement: `_load_pretrained()`, `_modify_classifier()`, `get_backbone_params()`
4. Register with decorator: `@ModelRegistry.register('model_name', variant='model_variant')`

### Adding a New Metric

1. Create metric class in `research/metrics/classification.py` or `regression.py`
2. Inherit from `BaseMetric` (from `research/metrics/base.py`)
3. Implement `calculate(self, predictions, targets)` method
4. Add to relevant exports in `research/metrics/__init__.py`

### Adding a New Task Strategy

1. Create strategy in `research/strategies/task/`
2. Inherit from `TaskStrategy`
3. Implement: `get_criterion()`, `calculate_metric()`, `prepare_labels()`
4. Add to exports in `research/strategies/task/__init__.py`

### Adding a New Comparator

1. Create comparator in `research/comparison/`
2. Inherit from `ModelComparator` (define in base.py or use existing interface)
3. Implement: `get_comparison_name()`, `compare(results)`
4. Add to exports in `research/comparison/__init__.py`

## Testing Guidelines

- All tests use pytest framework
- Fixtures defined in `tests/conftest.py` include: device, dummy data for all task types, metric trackers, experiment results
- Use markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.gpu`
- Tests should be fast and deterministic (use `seed_everything` fixture)
- Coverage target: aim for high coverage on core modules

## Important Notes

### Backward Compatibility

- The package maintains 100% backward compatibility with legacy `ktb_dl_research` API
- Legacy wrapper available in `ktb_dl_research.py` at project root
- All original APIs are re-exported through `research/__init__.py`

### Defensive Imports

- All imports in `__init__.py` use try-except blocks
- Missing dependencies won't crash the entire package
- Only successfully imported modules appear in `__all__`

### Code Style

- Black formatter with 100-character line length
- isort with black profile
- Type hints preferred but not strictly enforced
- Docstrings in Google style

### GPU Usage

- Framework automatically detects GPU availability
- Explicit device specification: `VanillaTrainingStrategy(device='cuda')`
- All strategies handle CPU/GPU seamlessly

### Input Channels

- Default: 3 channels (RGB images)
- For 1-channel data (grayscale, mel-spectrograms): set `in_channels: 1` in config
- Framework automatically adapts first convolutional layer
- For pretrained models with 1 channel: RGB weights are averaged to single channel

**Example for 1-channel data:**
```python
config = {
    'num_classes': 10,
    'in_channels': 1,  # For mel-spectrograms or grayscale
    'learning_rate': 1e-4,
    'max_epochs': 20
}
exp = Experiment(config)
exp.setup(model_name='resnet18', data_module=mel_dm, ...)
```

## Typical Workflow

1. Create data module (e.g., `CIFAR10DataModule`) or custom DataLoader
2. Define config dict with hyperparameters (including `in_channels` if not RGB)
3. Create `Experiment` instance and call `setup()` with model, strategies
4. Run experiment with `exp.run(strategy='fine_tuning')` or compare with `exp.compare_strategies(['feature_extraction', 'fine_tuning'])`
5. Use `ExperimentRecorder` to collect results from multiple experiments
6. Generate visualizations with `ExperimentVisualizer.plot_comparison()`
7. Compare models with `ComparisonManager`

## Version Information

- Current version: 0.1.0
- Python compatibility: 3.8+
- Main dependencies: PyTorch 2.0+, torchvision, numpy, matplotlib, scikit-learn, pandas
- Optional: wandb (for WandB logging), jupyter (for notebooks)

## Additional Documentation

- README.md: Project overview and quick start
- QUICKSTART.md: 5-minute tutorial
- ARCHITECTURE.md: Detailed architecture and design patterns
- examples/README.md: Example code descriptions
- research/visualization/VISUALIZATION_FEATURES.md: Visualization API specification

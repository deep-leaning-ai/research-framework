"""
KTB ML Framework - 통합 머신러닝 실험 프레임워크

기존 ktb_dl_research와 ml_framework를 통합하여 만든 범용 ML 실험 프레임워크입니다.
- 전이학습 (Pretrained Models: ResNet, VGG)
- 단순 모델 (CNN, FullyConnected)
- 다양한 태스크 전략 (MultiClass, Binary, Regression)
- 고급 메트릭 추적 및 비교 시스템
- 종합 시각화 도구

완전한 하위 호환성을 제공합니다.
"""

__version__ = '0.1.0'
__author__ = 'KTB AI Research Team'

# ============================================================================
# Core Components (기존 ktb_dl_research API 유지)
# ============================================================================

try:
    from .core.experiment import Experiment
except ImportError:
    Experiment = None

try:
    from .core.base_model import BaseModel
except ImportError:
    BaseModel = None

try:
    from .models.pretrained.registry import ModelRegistry
except ImportError:
    ModelRegistry = None

# ============================================================================
# Pretrained Models (전이학습용)
# ============================================================================

try:
    from .models.pretrained.resnet import ResNetModel
except ImportError:
    ResNetModel = None

try:
    from .models.pretrained.vgg import VGGModel
except ImportError:
    VGGModel = None

# ============================================================================
# Simple Models (ml_framework에서 추가)
# ============================================================================

try:
    from .models.simple.base import BaseModel as SimpleBaseModel
except ImportError:
    SimpleBaseModel = None

try:
    from .models.simple.cnn import CNN
except ImportError:
    CNN = None

try:
    from .models.simple.fully_connected import FullyConnectedNN
except ImportError:
    FullyConnectedNN = None

# ============================================================================
# Data Modules
# ============================================================================

try:
    from .data.cifar10 import CIFAR10DataModule
except ImportError:
    CIFAR10DataModule = None

try:
    from .data.loaders import DataLoaderFactory
except ImportError:
    DataLoaderFactory = None

# ============================================================================
# Training & Logging Strategies (기존 유지)
# ============================================================================

try:
    from .strategies.training.vanilla_strategy import VanillaTrainingStrategy
except ImportError:
    VanillaTrainingStrategy = None

try:
    from .strategies.logging.simple_strategy import SimpleLoggingStrategy
except ImportError:
    SimpleLoggingStrategy = None

try:
    from .strategies.logging.wandb_strategy import WandBLoggingStrategy
except ImportError:
    WandBLoggingStrategy = None

# ============================================================================
# Task Strategies (ml_framework에서 추가)
# ============================================================================

try:
    from .strategies.task.base import TaskStrategy
except ImportError:
    TaskStrategy = None

try:
    from .strategies.task.task_strategies import (
        MultiClassStrategy,
        BinaryClassificationStrategy,
        RegressionStrategy
    )
except ImportError:
    MultiClassStrategy = None
    BinaryClassificationStrategy = None
    RegressionStrategy = None

# ============================================================================
# Metrics System (ml_framework의 고급 메트릭 시스템)
# ============================================================================

try:
    from .metrics.base import MetricCalculator
except ImportError:
    MetricCalculator = None

try:
    from .metrics.classification import (
        AccuracyMetric,
        PrecisionMetric,
        RecallMetric,
        F1ScoreMetric,
        Top5AccuracyMetric,
        AUCMetric
    )
except ImportError:
    AccuracyMetric = None
    PrecisionMetric = None
    RecallMetric = None
    F1ScoreMetric = None
    Top5AccuracyMetric = None
    AUCMetric = None

try:
    from .metrics.regression import (
        MSEMetric,
        MAEMetric,
        R2Metric
    )
except ImportError:
    MSEMetric = None
    MAEMetric = None
    R2Metric = None

try:
    from .metrics.tracker import MetricTracker
except ImportError:
    MetricTracker = None

# ============================================================================
# Experiment Tools (ml_framework에서 추가)
# ============================================================================

try:
    from .experiment.runner import ExperimentRunner
except ImportError:
    ExperimentRunner = None

try:
    from .experiment.result import ExperimentResult
except ImportError:
    ExperimentResult = None

try:
    from .experiment.recorder import ExperimentRecorder
except ImportError:
    ExperimentRecorder = None

# ============================================================================
# Comparison System (ml_framework의 강화된 비교 시스템)
# ============================================================================

try:
    from .comparison.manager import ComparisonManager
except ImportError:
    ComparisonManager = None

try:
    from .comparison.comparators import (
        PerformanceComparator,
        EfficiencyComparator,
        SpeedComparator
    )
except ImportError:
    PerformanceComparator = None
    EfficiencyComparator = None
    SpeedComparator = None

# ============================================================================
# Visualization (통합 시각화 도구)
# ============================================================================

try:
    from .visualization.visualizer import ExperimentVisualizer
except ImportError:
    ExperimentVisualizer = None

# ============================================================================
# Analysis Tools removed - Legacy code is no longer supported
# Use new metrics, comparison, and visualization systems instead:
# - metrics: research.metrics.*
# - comparison: research.comparison.*
# - visualization: research.visualization.*
# ============================================================================


# ============================================================================
# Public API
# ============================================================================

# Dynamically build __all__ list - only include successfully imported modules
_all_exports = [
    # Core
    'Experiment',
    'BaseModel',
    'ModelRegistry',

    # Pretrained Models
    'ResNetModel',
    'VGGModel',

    # Simple Models
    'SimpleBaseModel',
    'CNN',
    'FullyConnectedNN',

    # Data
    'CIFAR10DataModule',
    'DataLoaderFactory',

    # Training & Logging Strategies
    'VanillaTrainingStrategy',
    'SimpleLoggingStrategy',
    'WandBLoggingStrategy',

    # Task Strategies
    'TaskStrategy',
    'MultiClassStrategy',
    'BinaryClassificationStrategy',
    'RegressionStrategy',

    # Metrics
    'MetricCalculator',
    'AccuracyMetric',
    'PrecisionMetric',
    'RecallMetric',
    'F1ScoreMetric',
    'Top5AccuracyMetric',
    'AUCMetric',
    'MSEMetric',
    'MAEMetric',
    'R2Metric',
    'MetricTracker',

    # Experiment Tools
    'ExperimentRunner',
    'ExperimentResult',
    'ExperimentRecorder',

    # Comparison
    'ComparisonManager',
    'PerformanceComparator',
    'EfficiencyComparator',
    'SpeedComparator',

    # Visualization
    'ExperimentVisualizer',
]

# Filter out None values (failed imports)
__all__ = [name for name in _all_exports if globals().get(name) is not None]

# ============================================================================
# Utility Functions
# ============================================================================

def list_models():
    """사용 가능한 pretrained 모델 목록 출력"""
    models = ModelRegistry.list_models()
    print(f"Available Pretrained Models ({len(models)}):")
    for model in models:
        print(f"  - {model}")
    return models

def list_simple_models():
    """사용 가능한 단순 모델 목록 출력"""
    simple_models = ['CNN', 'FullyConnectedNN']
    print(f"Available Simple Models ({len(simple_models)}):")
    for model in simple_models:
        print(f"  - {model}")
    return simple_models

def get_version():
    """프레임워크 버전 반환"""
    return __version__

def print_info():
    """프레임워크 정보 출력"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               KTB ML Framework v{__version__}                    ║
╚══════════════════════════════════════════════════════════════╝

통합 머신러닝 실험 프레임워크

주요 기능:
  - 전이학습 (ResNet, VGG 등 pretrained 모델)
  - 다양한 태스크 (MultiClass, Binary, Regression)
  - 고급 메트릭 추적 (실시간 다중 메트릭)
  - 모델 비교 시스템 (성능, 효율성, 속도)
  - 종합 시각화 (8-panel charts)

사용 예시:
  >>> import research
  >>> research.list_models()          # Pretrained 모델 목록
  >>> research.list_simple_models()   # 단순 모델 목록
  >>> exp = research.Experiment(config)  # 실험 생성

문서: 추후 제공 예정
GitHub: 추후 제공 예정
    """)

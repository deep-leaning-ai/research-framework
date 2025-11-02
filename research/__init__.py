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

from .core.experiment import Experiment
from .core.base_model import BaseModel
from .models.pretrained.registry import ModelRegistry

# ============================================================================
# Pretrained Models (전이학습용)
# ============================================================================

from .models.pretrained.resnet import ResNetModel
from .models.pretrained.vgg import VGGModel

# ============================================================================
# Simple Models (ml_framework에서 추가)
# ============================================================================

from .models.simple.base import BaseModel as SimpleBaseModel
from .models.simple.cnn import CNN
from .models.simple.fully_connected import FullyConnectedNN

# ============================================================================
# Data Modules
# ============================================================================

from .data.cifar10 import CIFAR10DataModule
from .data.loaders import DataLoaderFactory

# ============================================================================
# Training & Logging Strategies (기존 유지)
# ============================================================================

from .strategies.training.vanilla_strategy import VanillaTrainingStrategy
from .strategies.logging.simple_strategy import SimpleLoggingStrategy
from .strategies.logging.wandb_strategy import WandBLoggingStrategy

# ============================================================================
# Task Strategies (ml_framework에서 추가)
# ============================================================================

from .strategies.task.base import TaskStrategy
from .strategies.task.task_strategies import (
    MultiClassStrategy,
    BinaryClassificationStrategy,
    RegressionStrategy
)

# ============================================================================
# Metrics System (ml_framework의 고급 메트릭 시스템)
# ============================================================================

from .metrics.base import MetricCalculator
from .metrics.classification import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric
)
from .metrics.regression import (
    MSEMetric,
    MAEMetric
)
from .metrics.tracker import MetricTracker

# ============================================================================
# Experiment Tools (ml_framework에서 추가)
# ============================================================================

from .experiment.runner import ExperimentRunner
from .experiment.result import ExperimentResult
from .experiment.recorder import ExperimentRecorder

# ============================================================================
# Comparison System (ml_framework의 강화된 비교 시스템)
# ============================================================================

from .comparison.manager import ComparisonManager
from .comparison.comparators import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator
)

# ============================================================================
# Visualization (통합 시각화 도구)
# ============================================================================

from .visualization.visualizer import ExperimentVisualizer

# ============================================================================
# Analysis Tools (기존 ktb_dl_research 분석 도구 유지)
# ============================================================================

# 기존 visualizer 함수들 import (legacy 지원)
try:
    from .analysis.visualizer import (
        visualize_samples,
        plot_confusion_matrix,
        plot_comprehensive_comparison,
        plot_accuracy_improvement
    )
except ImportError:
    # visualizer가 없으면 스킵
    visualize_samples = None
    plot_confusion_matrix = None
    plot_comprehensive_comparison = None
    plot_accuracy_improvement = None

try:
    from .analysis.metrics import calculate_metrics
except ImportError:
    calculate_metrics = None

try:
    from .analysis.comparator import ModelComparator
except ImportError:
    ModelComparator = None

# ml_framework 분석 도구
try:
    from .analysis.model_analyzer import ModelAnalyzer
    from .analysis.performance import PerformanceAnalyzer
except ImportError:
    ModelAnalyzer = None
    PerformanceAnalyzer = None


# ============================================================================
# Public API
# ============================================================================

__all__ = [
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
    'MSEMetric',
    'MAEMetric',
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

    # Analysis (Legacy from ktb_dl_research)
    'ModelComparator',
    'visualize_samples',
    'plot_confusion_matrix',
    'plot_comprehensive_comparison',
    'plot_accuracy_improvement',
    'calculate_metrics',

    # Analysis (from ml_framework)
    'ModelAnalyzer',
    'PerformanceAnalyzer',
]

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

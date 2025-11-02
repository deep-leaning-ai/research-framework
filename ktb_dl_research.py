"""
KTB DL Research - Compatibility Wrapper

이 모듈은 research 패키지의 하위 호환성을 위한 래퍼입니다.
기존 ktb_dl_research API를 그대로 유지하면서 새로운 research 패키지를 사용합니다.

사용법:
    기존 코드:
    from ktb_dl_research import Experiment, ResNetModel, VGGModel

    새로운 코드 (권장):
    from research import Experiment, ResNetModel, VGGModel

    두 방식 모두 작동합니다!
"""

import warnings

# 하위 호환성 경고 (선택적으로 활성화)
_SHOW_DEPRECATION_WARNING = False

if _SHOW_DEPRECATION_WARNING:
    warnings.warn(
        "ktb_dl_research는 research 패키지로 통합되었습니다. "
        "'from research import ...' 사용을 권장합니다. "
        "기존 코드는 계속 작동하지만, 향후 버전에서 제거될 수 있습니다.",
        DeprecationWarning,
        stacklevel=2
    )

# research 패키지의 모든 공개 API를 re-export
from research import (
    # Version info
    __version__,
    __author__,
    
    # Core components
    Experiment,
    BaseModel,
    ModelRegistry,
    
    # Pretrained Models
    ResNetModel,
    VGGModel,
    
    # Data modules
    CIFAR10DataModule,
    
    # Strategies
    VanillaTrainingStrategy,
    SimpleLoggingStrategy,
    WandBLoggingStrategy,
    
    # Analysis tools
    ModelComparator,
    visualize_samples,
    plot_confusion_matrix,
    plot_comprehensive_comparison,
    plot_accuracy_improvement,
    calculate_metrics,
    
    # Utility functions
    list_models,
    get_version,
)

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Core
    'Experiment',
    'BaseModel',
    'ModelRegistry',
    
    # Models
    'ResNetModel',
    'VGGModel',
    
    # Data
    'CIFAR10DataModule',
    
    # Strategies
    'VanillaTrainingStrategy',
    'SimpleLoggingStrategy',
    'WandBLoggingStrategy',
    
    # Analysis
    'ModelComparator',
    'visualize_samples',
    'plot_confusion_matrix',
    'plot_comprehensive_comparison',
    'plot_accuracy_improvement',
    'calculate_metrics',
    
    # Functions
    'list_models',
    'get_version',
]

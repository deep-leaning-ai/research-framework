"""Logging strategies package"""

try:
    from .simple_strategy import SimpleLoggingStrategy
    __all__ = ['SimpleLoggingStrategy']
except ImportError:
    __all__ = []

try:
    from .wandb_strategy import WandBLoggingStrategy
    __all__.append('WandBLoggingStrategy')
except ImportError:
    pass

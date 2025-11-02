"""
모델 패키지
"""

from .base import BaseModel
from .cnn import CNN
from .fully_connected import FullyConnectedNN

__all__ = [
    "BaseModel",
    "CNN",
    "FullyConnectedNN",
]

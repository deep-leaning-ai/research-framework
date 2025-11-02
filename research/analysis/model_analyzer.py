"""
모델 분석기
"""

import torch.nn as nn
from typing import Dict, Any


class ModelAnalyzer:
    """모델 구조 및 파라미터 분석을 위한 클래스"""

    @staticmethod
    def print_analysis(model: nn.Module) -> None:
        """
        모델의 구조와 파라미터 정보를 출력합니다.

        Args:
            model: 분석할 PyTorch 모델
        """
        print(f"Model: {model.__class__.__name__}")
        print(f"Total Parameters: {ModelAnalyzer.count_parameters(model):,}")
        print(
            f"Trainable Parameters: {ModelAnalyzer.count_trainable_parameters(model):,}"
        )
        print("-" * 50)
        print(model)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """
        모델의 전체 파라미터 수를 계산합니다.

        Args:
            model: 파라미터를 계산할 모델

        Returns:
            전체 파라미터 수
        """
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def count_trainable_parameters(model: nn.Module) -> int:
        """
        훈련 가능한 파라미터 수를 계산합니다.

        Args:
            model: 파라미터를 계산할 모델

        Returns:
            훈련 가능한 파라미터 수
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        모델의 상세 정보를 딕셔너리로 반환합니다.

        Args:
            model: 분석할 모델

        Returns:
            모델 정보가 담긴 딕셔너리
        """
        return {
            "model_name": model.__class__.__name__,
            "total_parameters": ModelAnalyzer.count_parameters(model),
            "trainable_parameters": ModelAnalyzer.count_trainable_parameters(model),
            "model_size_mb": ModelAnalyzer.count_parameters(model)
            * 4
            / (1024**2),  # float32 기준
        }

    @staticmethod
    def analyze_layer_sizes(model: nn.Module) -> Dict[str, int]:
        """
        각 레이어별 파라미터 수를 분석합니다.

        Args:
            model: 분석할 모델

        Returns:
            레이어별 파라미터 수 딕셔너리
        """
        layer_info = {}
        for name, param in model.named_parameters():
            layer_info[name] = param.numel()
        return layer_info

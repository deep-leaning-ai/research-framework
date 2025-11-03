"""
구체적인 모델 비교 전략들
"""

from typing import Dict, Any
import numpy as np
from research.comparison.base import ModelComparator


class PerformanceComparator(ModelComparator):
    """
    성능 기반 비교

    특정 메트릭 기준으로 모델 성능 순위 매기기
    """

    def __init__(self, metric_name: str, higher_better: bool = True):
        """
        Args:
            metric_name: 비교할 메트릭 이름
            higher_better: 높을수록 좋은 메트릭인지
        """
        self.metric_name = metric_name
        self.higher_better = higher_better

    def compare(self, results: Dict[str, "ExperimentResult"]) -> Dict[str, Any]:
        """메트릭 기준으로 모델 순위 매기기"""
        model_scores = {}

        for model_name, result in results.items():
            if self.metric_name in result.test_metrics:
                # 최고 성능 추출
                values = result.test_metrics[self.metric_name]
                best_score = max(values) if self.higher_better else min(values)
                model_scores[model_name] = best_score

        # 정렬
        sorted_models = sorted(
            model_scores.items(), key=lambda x: x[1], reverse=self.higher_better
        )

        return {
            "ranking": sorted_models,
            "best_model": sorted_models[0][0] if sorted_models else None,
            "best_score": sorted_models[0][1] if sorted_models else None,
            "metric": self.metric_name,
        }

    def get_comparison_name(self) -> str:
        return f"Performance ({self.metric_name})"


class EfficiencyComparator(ModelComparator):
    """
    효율성 기반 비교

    파라미터 대비 성능 효율성 계산

    공식: efficiency = performance / log₁₀(parameters + EPSILON)
    - log₁₀: 파라미터 수의 영향을 감소시킴
    - EPSILON: log(0) 방지 및 최소값 보장
    """

    # 클래스 상수
    LOG_BASE = 10
    EPSILON = 1

    def __init__(self, metric_name: str):
        """
        Args:
            metric_name: 비교할 메트릭 이름
        """
        self.metric_name = metric_name

    def compare(self, results: Dict[str, "ExperimentResult"]) -> Dict[str, Any]:
        """파라미터 대비 성능 효율성 계산"""
        efficiency_scores = {}

        for model_name, result in results.items():
            if self.metric_name in result.test_metrics:
                best_metric = max(result.test_metrics[self.metric_name])
                params = result.parameters
                # 효율성 = 성능 / log₁₀(파라미터 수 + EPSILON)
                efficiency = (
                    best_metric / np.log10(params + self.EPSILON)
                    if params > 0
                    else 0
                )
                efficiency_scores[model_name] = {
                    "efficiency": efficiency,
                    "performance": best_metric,
                    "parameters": params,
                }

        sorted_models = sorted(
            efficiency_scores.items(), key=lambda x: x[1]["efficiency"], reverse=True
        )

        return {
            "ranking": sorted_models,
            "best_model": sorted_models[0][0] if sorted_models else None,
            "metric": self.metric_name,
        }

    def get_comparison_name(self) -> str:
        return f"Efficiency ({self.metric_name})"


class SpeedComparator(ModelComparator):
    """
    속도 기반 비교

    추론 속도와 학습 속도 비교
    """

    def compare(self, results: Dict[str, "ExperimentResult"]) -> Dict[str, Any]:
        """추론 속도 기준 비교"""
        speed_scores = {}

        for model_name, result in results.items():
            speed_scores[model_name] = {
                "inference_time": result.inference_time,
                "avg_epoch_time": (
                    np.mean(result.epoch_times) if result.epoch_times else 0
                ),
            }

        # 추론 시간 기준 정렬 (낮을수록 좋음)
        sorted_models = sorted(
            speed_scores.items(), key=lambda x: x[1]["inference_time"]
        )

        return {
            "ranking": sorted_models,
            "fastest_model": sorted_models[0][0] if sorted_models else None,
            "fastest_time": (
                sorted_models[0][1]["inference_time"] if sorted_models else None
            ),
        }

    def get_comparison_name(self) -> str:
        return "Speed Comparison"

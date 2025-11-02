"""
실험 결과 기록 및 관리
SRP: 실험 결과 기록만 담당
"""

from typing import Dict, Optional
from research.experiment.result import ExperimentResult


class ExperimentRecorder:
    """
    실험 결과 기록 및 관리 클래스

    여러 모델의 실험 결과를 저장하고 관리
    """

    def __init__(self):
        """실험 기록기 초기화"""
        self.results: Dict[str, ExperimentResult] = {}

    def add_result(self, result: ExperimentResult):
        """
        실험 결과 추가

        Args:
            result: 추가할 실험 결과
        """
        self.results[result.model_name] = result

    def get_result(self, model_name: str) -> Optional[ExperimentResult]:
        """
        특정 모델의 결과 조회

        Args:
            model_name: 모델 이름

        Returns:
            실험 결과 또는 None
        """
        return self.results.get(model_name)

    def get_all_results(self) -> Dict[str, ExperimentResult]:
        """
        모든 결과 조회

        Returns:
            모든 실험 결과 딕셔너리
        """
        return self.results

    def print_summary(self):
        """결과 요약 출력"""
        print(f"\n{'='*100}")
        print(f"{'실험 결과 요약':^100}")
        print(f"{'='*100}")
        print(
            f"{'모델':<25} | {'파라미터':>15} | {'주요 메트릭':>12} | "
            f"{'최고 성능':>12} | {'추론시간(ms)':>12}"
        )
        print(f"{'-'*100}")

        for model_name, result in self.results.items():
            print(
                f"{model_name:<25} | "
                f"{result.parameters:>15,} | "
                f"{result.primary_metric_name:>12} | "
                f"{result.best_test_metric:>12.2f} | "
                f"{result.inference_time*1000:>12.2f}"
            )

        print(f"{'='*100}\n")

    def clear(self):
        """모든 결과 삭제"""
        self.results.clear()

    def save_to_file(self, filepath: str):
        """
        결과를 파일로 저장

        Args:
            filepath: 저장 경로
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("실험 결과 리포트\n")
            f.write("=" * 100 + "\n\n")

            for model_name, result in self.results.items():
                f.write(f"\n## {model_name}\n")
                f.write("-" * 100 + "\n")
                f.write(result.summary())
                f.write("\n\n")

                # 메트릭 상세 정보
                f.write("Test Metrics:\n")
                for metric_name, values in result.test_metrics.items():
                    if values:
                        best_val = max(values)
                        final_val = values[-1]
                        f.write(
                            f"  {metric_name}: Best={best_val:.4f}, Final={final_val:.4f}\n"
                        )

                f.write("\n")

        print(f"실험 결과가 '{filepath}'에 저장되었습니다.")

    def get_best_model(
        self, metric_name: str, higher_better: bool = True
    ) -> Optional[str]:
        """
        특정 메트릭 기준 최고 모델 반환

        Args:
            metric_name: 메트릭 이름
            higher_better: 높을수록 좋은 메트릭인지

        Returns:
            최고 모델 이름 또는 None
        """
        if not self.results:
            return None

        best_model = None
        best_value = float("-inf") if higher_better else float("inf")

        for model_name, result in self.results.items():
            value = result.get_best_test_metric_for(metric_name, higher_better)
            if value is not None:
                if higher_better:
                    if value > best_value:
                        best_value = value
                        best_model = model_name
                else:
                    if value < best_value:
                        best_value = value
                        best_model = model_name

        return best_model
